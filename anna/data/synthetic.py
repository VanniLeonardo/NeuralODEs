from __future__ import annotations

from dataclasses import dataclass
from email import generator
from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from config import LatentODEConfig


@dataclass
class SampleTensors:
    observed_context: Tensor
    context_values: Tensor
    context_times: Tensor
    context_mask: Tensor
    interp_mask: Tensor
    full_times: Tensor
    ground_truth: Tensor
    future_mask: Tensor


class TimeSeriesDataset(Dataset[Dict[str, Tensor]]):
    """Synthetic irregular sine data for interpolation and extrapolation."""

    def __init__(self, config: LatentODEConfig):
        self.config = config
        self.num_context = config.seq_len // 2
        self.num_future = config.seq_len - self.num_context
        self._samples = self._build_samples()

    def _sample_times(self, generator: torch.Generator) -> tuple[Tensor, Tensor]:
        eps = 1e-4

        n_context_inner = self.num_context - 1
        n_future_inner = self.num_future - 1

        # Context times in (0, train_horizon), strictly increasing.
        # Anchor t0 = 0; sample the rest uniformly in (eps, train_horizon - eps),
        # then use float64 for sort+scale to avoid float32 collisions.
        if n_context_inner > 0:
            context_inner = torch.rand(
                n_context_inner, generator=generator, dtype=torch.float64
            )
            context_inner = torch.sort(context_inner).values
            context_inner = context_inner * (self.config.train_horizon - 2 * eps) + eps
            # Tiebreaker: add a tiny monotonically increasing offset to break any
            # rounding-induced ties. The offset is small enough that the last
            # element stays strictly below train_horizon - eps + eps = train_horizon.
            tie = 1e-9 * torch.arange(n_context_inner, dtype=torch.float64)
            context_inner = context_inner + tie
            context_times = torch.cat(
                [torch.zeros(1, dtype=torch.float64), context_inner], dim=0
            ).to(torch.float32)
        else:
            context_times = torch.zeros(1)

        # Future times in (train_horizon, extrap_horizon), strictly increasing,
        # with the final endpoint fixed at extrap_horizon.
        if n_future_inner > 0:
            future_inner = torch.rand(
                n_future_inner, generator=generator, dtype=torch.float64
            )
            future_inner = torch.sort(future_inner).values
            future_inner = future_inner * (
                self.config.extrap_horizon - self.config.train_horizon - 2 * eps
            ) + (self.config.train_horizon + eps)
            tie = 1e-9 * torch.arange(n_future_inner, dtype=torch.float64)
            future_inner = future_inner + tie
            future_times = torch.cat(
                [future_inner, torch.tensor([self.config.extrap_horizon], dtype=torch.float64)],
                dim=0,
            ).to(torch.float32)
        else:
            future_times = torch.tensor([self.config.extrap_horizon])

        assert torch.all(context_times[1:] > context_times[:-1]), "context_times not strictly increasing"
        assert torch.all(future_times[1:] > future_times[:-1]), "future_times not strictly increasing"
        assert future_times[0] > context_times[-1], "future must start after context ends"

        return context_times, future_times
    
    def _burst_mask(self, generator: torch.Generator) -> Tensor:
        """
        Generate a bursty missingness mask: all points observed except for one
        contiguous burst of length sampled uniformly from burst_length_range.
        The burst is placed at a uniformly random position within the context,
        but cannot cover t_0 (index 0) or the final context point.

        Returns: (num_context, 1) float tensor, 1 = observed, 0 = missing.
        """
        mask = torch.ones(self.num_context, 1)

        lo, hi = self.config.burst_length_range
        # Clamp so a burst can always fit strictly between indices 1 and num_context - 2
        max_burst = min(hi, self.num_context - 3)
        if max_burst < lo:
            return mask  # context too short for any burst; fall back to fully observed

        burst_len = int(
            torch.randint(lo, max_burst + 1, (1,), generator=generator).item()
        )
        # Start index must leave room for the burst and keep index 0 and last observed
        max_start = self.num_context - 1 - burst_len   # last valid start
        start = int(
            torch.randint(1, max_start, (1,), generator=generator).item()
        )
        mask[start : start + burst_len, 0] = 0.0
        return mask

    def _build_samples(self) -> list[SampleTensors]:
        generator = torch.Generator().manual_seed(self.config.seed)
        samples: list[SampleTensors] = []

        for _ in range(self.config.num_samples):
            context_times, future_times = self._sample_times(generator)
            full_times = torch.cat([context_times, future_times], dim=0)

            ground_truth = self._generate_signal(full_times, generator)
            noisy_values = ground_truth + self.config.noise_std * torch.randn(
                ground_truth.size(0), ground_truth.size(1), generator=generator
            )

            context_values = noisy_values[: self.num_context]

            if getattr(self.config, "burst_missing", False):
                context_keep = self._burst_mask(generator)
            else:
                context_keep = (
                    torch.rand(self.num_context, 1, generator=generator)
                    > self.config.missing_pct
                ).float()

            # Anchor t0 as observed so the encoder always sees the initial condition.
            context_keep[0] = 1.0

            # Enforce at least 2 observed points (encoder needs nontrivial info)
            # and at least 1 masked point (so interpolation MSE has a target).
            n_kept = int(context_keep.sum().item())
            if n_kept < 2 and self.num_context > 1:
                context_keep[1] = 1.0
                n_kept = int(context_keep.sum().item())
            if n_kept == self.num_context and self.num_context > 1:
                context_keep[-1] = 0.0


            # Broadcast the (num_context, 1) mask across input_dim so that
            # multi-dimensional signals get the same mask on every channel
            # (entire timesteps drop together, not individual dimensions).
            context_keep = context_keep.expand(-1, self.config.input_dim).contiguous()

            observed_context = context_values * context_keep
            interp_mask = 1.0 - context_keep
            future_mask = torch.ones(self.num_future, self.config.input_dim)

            samples.append(
                SampleTensors(
                    observed_context=observed_context,
                    context_values=context_values,
                    context_times=context_times,
                    context_mask=context_keep,
                    interp_mask=interp_mask,
                    full_times=full_times,
                    ground_truth=ground_truth,
                    future_mask=future_mask,
                )
            )

        return samples

    def _generate_signal(self, t: Tensor, generator: torch.Generator) -> Tensor:
        """
        Generate the clean target signal on time grid t.
        Returns shape (T, input_dim).
        """
        signal_type = getattr(self.config, "signal_type", "sine")

        if signal_type == "sine" and self.config.input_dim == 1:
            # Original 1D case
            phase = 2.0 * torch.pi * torch.rand(1, generator=generator)
            return torch.sin(t.unsqueeze(-1) + phase)

        if signal_type == "spiral":
            # 2D decaying spiral: x = r(t) cos(ωt + φ), y = r(t) sin(ωt + φ)
            # r(t) = r0 * exp(-α t). Each sample has random r0, ω, φ, α.
            assert self.config.input_dim == 2, "spiral requires input_dim=2"
            omega = 0.5 + 1.5 * torch.rand(1, generator=generator)   # angular freq in [0.5, 2.0]
            phase = 2.0 * torch.pi * torch.rand(1, generator=generator)
            alpha = 0.05 + 0.10 * torch.rand(1, generator=generator) # decay in [0.05, 0.15]
            r0 = 0.8 + 0.4 * torch.rand(1, generator=generator)      # amplitude in [0.8, 1.2]
            r = r0 * torch.exp(-alpha * t)
            x = r * torch.cos(omega * t + phase)
            y = r * torch.sin(omega * t + phase)
            return torch.stack([x, y], dim=-1)

        if signal_type == "damped":
            # 2D coupled oscillator: two independent damped sines with related freqs.
            assert self.config.input_dim == 2, "damped requires input_dim=2"
            omega1 = 0.5 + 1.5 * torch.rand(1, generator=generator)
            omega2 = omega1 * (1.0 + 0.3 * torch.rand(1, generator=generator))   # related but not equal
            phase1 = 2.0 * torch.pi * torch.rand(1, generator=generator)
            phase2 = 2.0 * torch.pi * torch.rand(1, generator=generator)
            alpha = 0.05 + 0.05 * torch.rand(1, generator=generator)
            damp = torch.exp(-alpha * t)
            x = damp * torch.sin(omega1 * t + phase1)
            y = damp * torch.sin(omega2 * t + phase2)
            return torch.stack([x, y], dim=-1)

        if signal_type == "sine" and self.config.input_dim > 1:
            # Multi-D independent sines (trivial ablation)
            phases = 2.0 * torch.pi * torch.rand(self.config.input_dim, generator=generator)
            return torch.sin(t.unsqueeze(-1) + phases)

        raise ValueError(f"unknown signal_type={signal_type} for input_dim={self.config.input_dim}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        sample = self._samples[index]
        return {
            "observed_context": sample.observed_context.clone(),
            "context_values": sample.context_values.clone(),
            "context_times": sample.context_times.clone(),
            "context_mask": sample.context_mask.clone(),
            "interp_mask": sample.interp_mask.clone(),
            "full_times": sample.full_times.clone(),
            "ground_truth": sample.ground_truth.clone(),
            "future_mask": sample.future_mask.clone(),
        }
