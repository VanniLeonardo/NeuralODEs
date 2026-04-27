from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from kaby.config import ODEConfig


class IrregularSineWaveDataset(Dataset):
    """Synthetic irregular sine-wave dataset for interpolation/extrapolation.

    Each sample contains:
    - noisy observed context values
    - clean context targets
    - irregular context timestamps
    - context observation mask
    - interpolation mask
    - full query timestamps
    - full clean target trajectory
    - future mask
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

    def _sample_irregular_times(
        self,
        low: float,
        high: float,
        num_points: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Samples strictly ordered irregular timestamps."""
        times = low + (high - low) * torch.rand(num_points, generator=generator)
        return torch.sort(times).values.to(dtype=torch.float32)

    def _sample_observation_mask(self, generator: torch.Generator) -> torch.Tensor:
        """Samples a context observation mask with valid constraints."""
        mask = torch.rand(self.n_context_points, generator=generator) < self.observation_prob

        observed_count = int(mask.sum().item())
        if observed_count < self.min_observed_context_points:
            candidate_indices = torch.randperm(self.n_context_points, generator=generator)
            for index in candidate_indices.tolist():
                mask[index] = True
                observed_count += 1
                if observed_count >= self.min_observed_context_points:
                    break

        if int((~mask).sum().item()) == 0:
            observed_indices = torch.nonzero(mask, as_tuple=False).flatten()
            flip_index = int(observed_indices[-1].item())
            mask[flip_index] = False

        return mask

    def _generate_sample(self, generator: torch.Generator) -> Dict[str, torch.Tensor]:
        """Generates one noisy sine-wave trajectory with random phase."""
        phase = 2.0 * math.pi * torch.rand(1, generator=generator).item()

        context_times = self._sample_irregular_times(
            low=self.context_start,
            high=self.context_end,
            num_points=self.n_context_points,
            generator=generator,
        )
        future_times = self._sample_irregular_times(
            low=self.context_end,
            high=self.future_end,
            num_points=self.n_future_points,
            generator=generator,
        )

        context_targets = torch.sin(context_times + phase).unsqueeze(-1)
        future_targets = torch.sin(future_times + phase).unsqueeze(-1)

        context_mask = self._sample_observation_mask(generator=generator)

        context_noise = self.noise_std * torch.randn(
            self.n_context_points,
            1,
            generator=generator,
        )
        noisy_context = context_targets + context_noise

        context_observations = torch.where(
            context_mask.unsqueeze(-1),
            noisy_context,
            torch.zeros_like(noisy_context),
        )

        query_times = torch.cat([context_times, future_times], dim=0)
        query_targets = torch.cat([context_targets, future_targets], dim=0)

        future_mask = torch.zeros(
            self.n_context_points + self.n_future_points,
            dtype=torch.bool,
        )
        future_mask[self.n_context_points :] = True

        query_observed_mask = torch.zeros_like(future_mask)
        query_observed_mask[: self.n_context_points] = context_mask

        interpolation_mask = torch.zeros_like(future_mask)
        interpolation_mask[: self.n_context_points] = ~context_mask

        return {
            "context_times": context_times,
            "context_observations": context_observations,
            "context_targets": context_targets,
            "context_mask": context_mask,
            "interpolation_mask": interpolation_mask,
            "query_times": query_times,
            "query_targets": query_targets,
            "query_observed_mask": query_observed_mask,
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