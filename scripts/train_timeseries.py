from __future__ import annotations

import argparse
import copy
import random
from typing import Any

import numpy as np
import torch
import wandb

from config import ODEConfig
from data.timeseries import get_irregular_sine_dataloaders
from models.ode_rnn import GRUTimeSeriesBaseline, ODERNN
from training.timeseries_engine import evaluate_timeseries, train_timeseries_epoch

try:
    from rich.console import Console
except ImportError:  # pragma: no cover
    class Console:  # type: ignore[override]
        def print(self, *args: Any, **kwargs: Any) -> None:
            print(*args)

        def rule(self, title: str) -> None:
            print(title)


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Counts trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for time-series experiments."""
    defaults = ODEConfig()

    parser = argparse.ArgumentParser(
        description="Train ODE-RNN or GRU on irregular sine-wave data.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=defaults.model_type,
        choices=["ode_rnn", "gru"],
    )
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--hidden_dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--solver", type=str, default=defaults.solver_type)
    parser.add_argument("--atol", type=float, default=defaults.atol)
    parser.add_argument("--rtol", type=float, default=defaults.rtol)

    parser.add_argument("--context_start", type=float, default=defaults.context_start)
    parser.add_argument("--context_end", type=float, default=defaults.context_end)
    parser.add_argument("--future_end", type=float, default=defaults.future_end)
    parser.add_argument(
        "--n_context_points",
        type=int,
        default=defaults.n_context_points,
    )
    parser.add_argument(
        "--n_future_points",
        type=int,
        default=defaults.n_future_points,
    )
    parser.add_argument(
        "--min_observed_context_points",
        type=int,
        default=defaults.min_observed_context_points,
    )
    parser.add_argument(
        "--observation_prob",
        type=float,
        default=defaults.observation_prob,
    )
    parser.add_argument("--noise_std", type=float, default=defaults.noise_std)

    parser.add_argument("--train_size", type=int, default=defaults.train_size)
    parser.add_argument("--val_size", type=int, default=defaults.val_size)
    parser.add_argument("--test_size", type=int, default=defaults.test_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="neural-odes-30562",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="offline",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ODEConfig:
    """Builds a dataclass config from CLI overrides."""
    config = ODEConfig()
    config.is_time_series = True
    config.in_features = 1
    config.output_dim = 1

    config.model_type = args.model_type
    config.batch_size = args.batch_size
    config.hidden_dim = args.hidden_dim
    config.lr = args.lr
    config.epochs = args.epochs
    config.solver_type = args.solver
    config.atol = args.atol
    config.rtol = args.rtol

    config.context_start = args.context_start
    config.context_end = args.context_end
    config.future_end = args.future_end
    config.n_context_points = args.n_context_points
    config.n_future_points = args.n_future_points
    config.min_observed_context_points = args.min_observed_context_points
    config.observation_prob = args.observation_prob
    config.noise_std = args.noise_std

    config.train_size = args.train_size
    config.val_size = args.val_size
    config.test_size = args.test_size
    config.seed = args.seed

    return config


def build_model(config: ODEConfig) -> torch.nn.Module:
    """Initializes the requested time-series model."""
    if config.model_type == "ode_rnn":
        return ODERNN(
            input_dim=config.in_features,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            solver_type=config.solver_type,
            atol=config.atol,
            rtol=config.rtol,
            start_time=config.context_start,
        )

    return GRUTimeSeriesBaseline(
        input_dim=config.in_features,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        start_time=config.context_start,
    )


def main() -> None:
    """Runs a full irregular time-series experiment."""
    args = parse_args()
    config = build_config(args)
    seed_everything(config.seed)

    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        mode=args.wandb_mode,
        project=args.wandb_project,
        config=config.__dict__,
    )

    train_loader, val_loader, test_loader = get_irregular_sine_dataloaders(config)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    num_parameters = count_parameters(model)

    console.rule("Irregular Time-Series Experiment")
    console.print(f"Device: {device}")
    console.print(f"Model: {config.model_type}")
    console.print(f"Trainable parameters: {num_parameters}")

    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_timeseries_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate_timeseries(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state_dict = copy.deepcopy(model.state_dict())

        wandb.log(
            {
                "epoch": epoch,
                "num_parameters": num_parameters,
                "train_loss": train_metrics["loss"],
                "train_observed_mse": train_metrics["observed_mse"],
                "train_interpolation_mse": train_metrics["interpolation_mse"],
                "train_extrapolation_mse": train_metrics["extrapolation_mse"],
                "train_forward_nfe_per_sample": train_metrics["nfe_per_sample"],
                "train_peak_memory_mb": train_metrics["memory_mb"],
                "val_loss": val_metrics["loss"],
                "val_observed_mse": val_metrics["observed_mse"],
                "val_interpolation_mse": val_metrics["interpolation_mse"],
                "val_extrapolation_mse": val_metrics["extrapolation_mse"],
                "val_forward_nfe_per_sample": val_metrics["nfe_per_sample"],
            }
        )

        console.print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_interp={val_metrics['interpolation_mse']:.6f} | "
            f"val_extra={val_metrics['extrapolation_mse']:.6f} | "
            f"nfe/sample={train_metrics['nfe_per_sample']:.3f}"
        )

    model.load_state_dict(best_state_dict)

    test_metrics = evaluate_timeseries(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    wandb.log(
        {
            "best_val_loss": best_val_loss,
            "test_loss": test_metrics["loss"],
            "test_observed_mse": test_metrics["observed_mse"],
            "test_interpolation_mse": test_metrics["interpolation_mse"],
            "test_extrapolation_mse": test_metrics["extrapolation_mse"],
            "test_forward_nfe_per_sample": test_metrics["nfe_per_sample"],
        }
    )

    console.rule("Final Test Metrics")
    console.print(test_metrics)

    wandb.finish()


if __name__ == "__main__":
    main()