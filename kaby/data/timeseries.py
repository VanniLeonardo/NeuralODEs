from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from kaby.config import ODEConfig


class IrregularSineWaveDataset(Dataset):
    """Synthetic irregular sine-wave dataset with Anna-style batch fields.

    Each sample contains:
    - observed_context: noisy context values, zero-filled where missing
    - context_values: noisy context values before masking
    - context_times: irregular timestamps in the context interval
    - context_mask: 1 where context is observed, 0 where missing
    - interp_mask: 1 where context is hidden and used for interpolation
    - full_times: context_times followed by future times
    - ground_truth: clean full trajectory over full_times
    - future_mask: 1 over the future horizon only
    """

    def __init__(
        self,
        num_samples: int,
        context_start: float = 0.0,
        context_end: float = 5.0,
        future_end: float = 10.0,
        n_context_points: int = 20,
        n_future_points: int = 20,
        min_observed_context_points: int = 2,
        observation_prob: float = 0.7,
        noise_std: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if not 0.0 <= context_start < context_end < future_end:
            raise ValueError("Expected context_start < context_end < future_end.")
        if n_context_points < min_observed_context_points + 1:
            raise ValueError(
                "n_context_points must exceed min_observed_context_points so "
                "interpolation targets exist."
            )
        if n_future_points <= 0:
            raise ValueError("n_future_points must be positive.")
        if not 0.0 < observation_prob < 1.0:
            raise ValueError("observation_prob must lie strictly between 0 and 1.")

        self.num_samples = num_samples
        self.context_start = context_start
        self.context_end = context_end
        self.future_end = future_end
        self.n_context_points = n_context_points
        self.n_future_points = n_future_points
        self.min_observed_context_points = min_observed_context_points
        self.observation_prob = observation_prob
        self.noise_std = noise_std

        generator = torch.Generator().manual_seed(seed)
        self.samples = [self._generate_sample(generator) for _ in range(num_samples)]

    def _sample_context_and_future_times(
        self,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples strictly increasing context/future times.

        To be a bit closer to Anna's benchmark style:
        - the first context time is anchored at context_start
        - the last future time is anchored at future_end
        """
        eps = 1e-4

        n_context_inner = self.n_context_points - 1
        if n_context_inner > 0:
            context_inner = self.context_start + eps + (
                self.context_end - self.context_start - 2 * eps
            ) * torch.rand(n_context_inner, generator=generator)
            context_inner = torch.sort(context_inner).values
            context_times = torch.cat(
                [torch.tensor([self.context_start]), context_inner], dim=0
            ).to(torch.float32)
        else:
            context_times = torch.tensor([self.context_start], dtype=torch.float32)

        n_future_inner = self.n_future_points - 1
        if n_future_inner > 0:
            future_inner = self.context_end + eps + (
                self.future_end - self.context_end - 2 * eps
            ) * torch.rand(n_future_inner, generator=generator)
            future_inner = torch.sort(future_inner).values
            future_times = torch.cat(
                [future_inner, torch.tensor([self.future_end])], dim=0
            ).to(torch.float32)
        else:
            future_times = torch.tensor([self.future_end], dtype=torch.float32)

        return context_times, future_times

    def _sample_observation_mask(self, generator: torch.Generator) -> torch.Tensor:
        """Samples a context mask and guarantees a usable interpolation task."""
        mask = torch.rand(self.n_context_points, generator=generator) < self.observation_prob

        
        mask[0] = True

        observed_count = int(mask.sum().item())
        if observed_count < self.min_observed_context_points:
            candidate_indices = torch.randperm(self.n_context_points, generator=generator)
            for index in candidate_indices.tolist():
                mask[index] = True
                observed_count += 1
                if observed_count >= self.min_observed_context_points:
                    break

        # Ensure at least one missing point remains in the context interval.
        if int((~mask).sum().item()) == 0 and self.n_context_points > 1:
            mask[-1] = False

        return mask

    def _generate_sample(self, generator: torch.Generator) -> Dict[str, torch.Tensor]:
        """Generates one noisy sine-wave trajectory with random phase."""
        phase = 2.0 * math.pi * torch.rand(1, generator=generator).item()

        context_times, future_times = self._sample_context_and_future_times(generator)
        full_times = torch.cat([context_times, future_times], dim=0)

        ground_truth = torch.sin(full_times + phase).unsqueeze(-1)

        context_values = ground_truth[: self.n_context_points] + self.noise_std * torch.randn(
            self.n_context_points,
            1,
            generator=generator,
        )

        context_keep = self._sample_observation_mask(generator=generator)

        observed_context = torch.where(
            context_keep.unsqueeze(-1),
            context_values,
            torch.zeros_like(context_values),
        )

        context_mask = context_keep.unsqueeze(-1).float()
        interp_mask = (~context_keep).unsqueeze(-1).float()
        future_mask = torch.ones(self.n_future_points, 1, dtype=torch.float32)

        return {
            "observed_context": observed_context,
            "context_values": context_values,
            "context_times": context_times,
            "context_mask": context_mask,
            "interp_mask": interp_mask,
            "full_times": full_times,
            "ground_truth": ground_truth,
            "future_mask": future_mask,
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {key: value.clone() for key, value in sample.items()}


def get_irregular_sine_dataloaders(
    config: ODEConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds train/val/test dataloaders for the synthetic sine benchmark."""
    train_dataset = IrregularSineWaveDataset(
        num_samples=config.train_size,
        context_start=config.context_start,
        context_end=config.context_end,
        future_end=config.future_end,
        n_context_points=config.n_context_points,
        n_future_points=config.n_future_points,
        min_observed_context_points=config.min_observed_context_points,
        observation_prob=config.observation_prob,
        noise_std=config.noise_std,
        seed=config.seed,
    )
    val_dataset = IrregularSineWaveDataset(
        num_samples=config.val_size,
        context_start=config.context_start,
        context_end=config.context_end,
        future_end=config.future_end,
        n_context_points=config.n_context_points,
        n_future_points=config.n_future_points,
        min_observed_context_points=config.min_observed_context_points,
        observation_prob=config.observation_prob,
        noise_std=config.noise_std,
        seed=config.seed + 1,
    )
    test_dataset = IrregularSineWaveDataset(
        num_samples=config.test_size,
        context_start=config.context_start,
        context_end=config.context_end,
        future_end=config.future_end,
        n_context_points=config.n_context_points,
        n_future_points=config.n_future_points,
        min_observed_context_points=config.min_observed_context_points,
        observation_prob=config.observation_prob,
        noise_std=config.noise_std,
        seed=config.seed + 2,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader