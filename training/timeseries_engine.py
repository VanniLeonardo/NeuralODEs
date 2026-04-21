from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn


BatchDict = Mapping[str, torch.Tensor]


def move_batch_to_device(
    batch: BatchDict,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Moves a dictionary batch to the requested device."""
    return {key: value.to(device) for key, value in batch.items()}


def masked_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Computes MSE over a boolean mask."""
    if mask.dtype != torch.bool:
        mask = mask.bool()

    if mask.ndim == 2:
        expanded_mask = mask.unsqueeze(-1).expand_as(predictions)
    elif mask.ndim == 3:
        expanded_mask = mask.expand_as(predictions)
    else:
        raise ValueError("mask must have shape [batch, time] or [batch, time, dim].")

    if not bool(expanded_mask.any().item()):
        return predictions.new_tensor(0.0)

    squared_error = (predictions - targets).pow(2)
    return torch.masked_select(squared_error, expanded_mask).mean()


def compute_timeseries_metrics(
    predictions: torch.Tensor,
    batch: BatchDict,
) -> Dict[str, torch.Tensor]:
    """Computes reconstruction/interpolation/extrapolation metrics."""
    targets = batch["query_targets"]
    full_mask = torch.ones_like(batch["future_mask"], dtype=torch.bool)

    return {
        "loss": masked_mse(predictions, targets, full_mask),
        "observed_mse": masked_mse(
            predictions,
            targets,
            batch["query_observed_mask"],
        ),
        "interpolation_mse": masked_mse(
            predictions,
            targets,
            batch["interpolation_mask"],
        ),
        "extrapolation_mse": masked_mse(
            predictions,
            targets,
            batch["future_mask"],
        ),
    }


def train_timeseries_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Runs one training epoch for time-series forecasting."""
    model.train()

    total_loss = 0.0
    total_observed_mse = 0.0
    total_interpolation_mse = 0.0
    total_extrapolation_mse = 0.0
    total_nfe = 0.0
    total_samples = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        batch_size = batch["context_times"].size(0)

        optimizer.zero_grad()

        predictions = model(
            context_times=batch["context_times"],
            context_observations=batch["context_observations"],
            context_mask=batch["context_mask"],
            query_times=batch["query_times"],
        )
        metrics = compute_timeseries_metrics(predictions, batch)
        loss = metrics["loss"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_observed_mse += metrics["observed_mse"].item() * batch_size
        total_interpolation_mse += metrics["interpolation_mse"].item() * batch_size
        total_extrapolation_mse += metrics["extrapolation_mse"].item() * batch_size
        total_nfe += float(getattr(model, "get_nfe", lambda: 0)())
        total_samples += batch_size

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "loss": total_loss / total_samples,
        "observed_mse": total_observed_mse / total_samples,
        "interpolation_mse": total_interpolation_mse / total_samples,
        "extrapolation_mse": total_extrapolation_mse / total_samples,
        "nfe_per_sample": total_nfe / total_samples if total_samples > 0 else 0.0,
        "memory_mb": peak_memory_mb,
    }


@torch.no_grad()
def evaluate_timeseries(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates a time-series model."""
    model.eval()

    total_loss = 0.0
    total_observed_mse = 0.0
    total_interpolation_mse = 0.0
    total_extrapolation_mse = 0.0
    total_nfe = 0.0
    total_samples = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        batch_size = batch["context_times"].size(0)

        predictions = model(
            context_times=batch["context_times"],
            context_observations=batch["context_observations"],
            context_mask=batch["context_mask"],
            query_times=batch["query_times"],
        )
        metrics = compute_timeseries_metrics(predictions, batch)

        total_loss += metrics["loss"].item() * batch_size
        total_observed_mse += metrics["observed_mse"].item() * batch_size
        total_interpolation_mse += metrics["interpolation_mse"].item() * batch_size
        total_extrapolation_mse += metrics["extrapolation_mse"].item() * batch_size
        total_nfe += float(getattr(model, "get_nfe", lambda: 0)())
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "observed_mse": total_observed_mse / total_samples,
        "interpolation_mse": total_interpolation_mse / total_samples,
        "extrapolation_mse": total_extrapolation_mse / total_samples,
        "nfe_per_sample": total_nfe / total_samples if total_samples > 0 else 0.0,
    }