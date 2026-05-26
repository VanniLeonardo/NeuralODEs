from __future__ import annotations

import random
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, random_split
import json
from pathlib import Path
import time
import torch
import argparse
try:
    import wandb
except ImportError:  
    class _WandbStub:
        def init(self, *args, **kwargs):
            return self

        def log(self, *args, **kwargs):
            return None

        def finish(self) -> None:
            return None

    wandb = _WandbStub()

from config import LatentODEConfig
from models.continuous import LatentODE
from data.synthetic import TimeSeriesDataset 

console = Console()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    denom = mask.sum().clamp_min(1.0)
    return ((pred - target) ** 2 * mask).sum() / denom


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


def move_batch_to_device(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: LatentODE, loader: DataLoader, config: LatentODEConfig) -> Dict[str, float]:
    model.eval()

    total_interp_num = 0.0
    total_interp_den = 0.0
    total_extrap_num = 0.0
    total_extrap_den = 0.0
    total_recon_num = 0.0
    total_recon_den = 0.0
    total_kl = 0.0
    total_nfe = 0.0
    total_nfe_encoder = 0.0
    total_nfe_latent = 0.0
    num_batches = 0

    context_len = config.seq_len // 2

    for batch in loader:
        batch = move_batch_to_device(batch, torch.device(config.device))
        out = model(
            x=batch["observed_context"],
            t_obs=batch["context_times"],
            mask=batch["context_mask"],
            t_query=batch["full_times"],
            sample_latent=False,
        )

        preds = out["predictions"]
        interp_pred = preds[:, :context_len, :]
        future_pred = preds[:, context_len:, :]

        interp_sqerr = ((interp_pred - batch["ground_truth"][:, :context_len, :]) ** 2) * batch["interp_mask"]
        extrap_sqerr = ((future_pred - batch["ground_truth"][:, context_len:, :]) ** 2) * batch["future_mask"]
        recon_sqerr = ((interp_pred - batch["context_values"]) ** 2) * batch["context_mask"]

        total_interp_num += interp_sqerr.sum().item()
        total_interp_den += batch["interp_mask"].sum().item()
        total_extrap_num += extrap_sqerr.sum().item()
        total_extrap_den += batch["future_mask"].sum().item()
        total_recon_num += recon_sqerr.sum().item()
        total_recon_den += batch["context_mask"].sum().item()
        total_kl += kl_divergence(out["mu"], out["logvar"]).item()
        total_nfe += out["nfe"].item()
        total_nfe_encoder += out["nfe_encoder"].item()
        total_nfe_latent += out["nfe_latent"].item()
        num_batches += 1

    return {
        "recon_mse": total_recon_num / max(total_recon_den, 1.0),
        "interpolation_mse": total_interp_num / max(total_interp_den, 1.0),
        "extrapolation_mse": total_extrap_num / max(total_extrap_den, 1.0),
        "kl": total_kl / max(num_batches, 1),
        "nfe": total_nfe / max(num_batches, 1),
        "nfe_encoder": total_nfe_encoder / max(num_batches, 1),
        "nfe_latent": total_nfe_latent / max(num_batches, 1),
    }


def train_one_epoch(
    model: LatentODE,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    config: LatentODEConfig,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_nfe = 0.0
    total_nfe_encoder = 0.0
    total_nfe_latent = 0.0
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, torch.device(config.device))
        optimizer.zero_grad(set_to_none=True)

        out = model(
            x=batch["observed_context"],
            t_obs=batch["context_times"],
            mask=batch["context_mask"],
            t_query=batch["context_times"],
            sample_latent=None,
        )

        recon_loss = masked_mse(
            pred=out["predictions"],
            target=batch["context_values"],
            mask=batch["context_mask"],
        )
        kl_loss = (
            kl_divergence(out["mu"], out["logvar"])
            if config.is_variational
            else torch.zeros((), device=torch.device(config.device))
        )
        loss = recon_loss + config.kl_coeff * kl_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_nfe += out["nfe"].item()
        total_nfe_encoder += out["nfe_encoder"].item()
        total_nfe_latent += out["nfe_latent"].item()
        num_batches += 1

    return {
        "train_loss": total_loss / max(num_batches, 1),
        "train_recon": total_recon / max(num_batches, 1),
        "train_kl": total_kl / max(num_batches, 1),
        "train_nfe": total_nfe / max(num_batches, 1),
        "train_nfe_encoder": total_nfe_encoder / max(num_batches, 1),
        "train_nfe_latent": total_nfe_latent / max(num_batches, 1),
    }


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics.items():
        table.add_row(key, f"{value:.6f}")
    console.print(table)


def _to_torch_device(device_like) -> torch.device:
    if isinstance(device_like, torch.device):
        return device_like
    return torch.device(device_like)


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-ode-rnn", type=str, default=None)
    parser.add_argument("--is-variational", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--burst-missing", type=str, default=None)
    parser.add_argument("--extrap-horizon", type=float, default=None)
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--signal-type", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-val", type=int, default=None)
    parser.add_argument("--num-test", type=int, default=None)
    parser.add_argument("--encoder-type", type=str, default=None,
                        help="one of: odernn, gru_time, gru_notime")
    return parser.parse_args()
    
@torch.no_grad()
def evaluate_persistence(loader: DataLoader, config: LatentODEConfig) -> Dict[str, float]:
    """
    Non-parametric baseline: predict each query point as the last observed
    context value (carry-forward). Serves as a floor for interpolation and
    extrapolation MSE.
    """
    total_interp_num = 0.0
    total_interp_den = 0.0
    total_extrap_num = 0.0
    total_extrap_den = 0.0

    context_len = config.seq_len // 2

    for batch in loader:
        observed = batch["observed_context"]       
        context_mask = batch["context_mask"]       
        ground_truth = batch["ground_truth"]       
        interp_mask = batch["interp_mask"]         
        future_mask = batch["future_mask"]         

        B, T_ctx, D = observed.shape
        T_full = ground_truth.shape[1]


        pred_ctx = torch.zeros_like(observed)
        last_value = torch.zeros(B, D, device=observed.device, dtype=observed.dtype)
        for i in range(T_ctx):
            is_obs = context_mask[:, i, :].bool()          
            last_value = torch.where(is_obs, observed[:, i, :], last_value)
            pred_ctx[:, i, :] = last_value

        pred_future = last_value.unsqueeze(1).expand(B, T_full - T_ctx, D)

        interp_sqerr = ((pred_ctx - ground_truth[:, :context_len, :]) ** 2) * interp_mask
        extrap_sqerr = ((pred_future - ground_truth[:, context_len:, :]) ** 2) * future_mask

        total_interp_num += interp_sqerr.sum().item()
        total_interp_den += interp_mask.sum().item()
        total_extrap_num += extrap_sqerr.sum().item()
        total_extrap_den += future_mask.sum().item()

    return {
        "persistence_interpolation_mse": total_interp_num / max(total_interp_den, 1.0),
        "persistence_extrapolation_mse": total_extrap_num / max(total_extrap_den, 1.0),
    }

    
def main() -> None:
    config = LatentODEConfig()
    args = parse_args()

    if args.use_ode_rnn is not None:
        config.use_ode_rnn = args.use_ode_rnn.lower() == "true"

    if args.is_variational is not None:
        config.is_variational = args.is_variational.lower() == "true"

    if args.eval_every is not None:
        config.eval_every = args.eval_every

    if args.skip_validation:
        config.skip_validation = True

    if args.seed is not None:
        config.seed = args.seed
    if args.method is not None:
        config.method = args.method

    if args.extrap_horizon is not None:
        config.extrap_horizon = args.extrap_horizon
    if args.burst_missing is not None:
        config.burst_missing = args.burst_missing.lower() == "true"
    if args.input_dim is not None:
        config.input_dim = args.input_dim
    if args.signal_type is not None:
        config.signal_type = args.signal_type
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.num_train is not None:
        config.num_train = args.num_train
    if args.num_val is not None:
        config.num_val = args.num_val
    if args.num_test is not None:
        config.num_test = args.num_test

    device = _to_torch_device(config.device)
    config.device = device

    if not hasattr(config, "seed"):
        config.seed = 42
    if not hasattr(config, "eval_every"):
        config.eval_every = 1
    if not hasattr(config, "skip_validation"):
        config.skip_validation = False
    if not hasattr(config, "profile_dataloader"):
        config.profile_dataloader = True
    if not hasattr(config, "profile_num_batches"):
        config.profile_num_batches = 20
    if not hasattr(config, "num_workers"):
        config.num_workers = 0
    if not hasattr(config, "pin_memory"):
        config.pin_memory = (device.type == "cuda")
    if args.encoder_type is not None:
        config.encoder_type = args.encoder_type
        config.use_ode_rnn = (args.encoder_type == "odernn")
 

    set_seed(config.seed)
    wandb.init(
        project=config.project_name,
        name=args.run_name,
        config={
            key: str(value) if isinstance(value, torch.device) else value
            for key, value in config.__dict__.items()
        },
    )

    dataset = TimeSeriesDataset(config=config)

    assert len(dataset) == config.num_samples, (
        f"dataset size {len(dataset)} != num_train+num_val+num_test = {config.num_samples}"
    )

    split_generator = torch.Generator().manual_seed(config.seed)
    train_set, val_set, test_set = random_split(
        dataset,
        [config.num_train, config.num_val, config.num_test],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    model = LatentODE(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    console.log(
            f"phase 4 faithful run | "
            f"encoder_type={getattr(config, 'encoder_type', 'odernn' if config.use_ode_rnn else 'gru_time')} | "
            f"is_variational={config.is_variational} | "
            f"device={device}"
        )
    console.log(
        f"dataset sizes | train={len(train_set)} | val={len(val_set)} | test={len(test_set)}"
    )

    for attr_name in ("num_context", "num_future"):
        if hasattr(dataset, attr_name):
            console.log(f"{attr_name}={getattr(dataset, attr_name)}")
    if hasattr(dataset, "num_context") and hasattr(dataset, "num_future"):
        console.log(f"t_full={dataset.num_context + dataset.num_future}")

    if config.profile_dataloader:
        t0 = time.time()
        for batch_idx, _ in enumerate(train_loader):
            if batch_idx + 1 >= config.profile_num_batches:
                break
        loader_time = time.time() - t0
        console.log(
            f"dataloader probe | first {config.profile_num_batches} batches in {loader_time:.2f}s"
        )

    best_val_interp = float("inf")
    best_state_dict = None

    for epoch in range(1, config.epochs + 1):
        _sync_cuda(device)
        train_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, config)
        _sync_cuda(device)
        train_time = time.time() - train_start

        val_metrics = {
            "recon_mse": float("nan"),
            "interpolation_mse": float("nan"),
            "extrapolation_mse": float("nan"),
            "kl": float("nan"),
            "nfe": float("nan"),
            "nfe_encoder": float("nan"),
            "nfe_latent": float("nan"),
        }
        val_time = 0.0

        do_validation = (not config.skip_validation) and (
            epoch % config.eval_every == 0 or epoch == 1 or epoch == config.epochs
        )

        if do_validation:
            _sync_cuda(device)
            val_start = time.time()
            val_metrics = evaluate(model, val_loader, config)
            _sync_cuda(device)
            val_time = time.time() - val_start

            if val_metrics["interpolation_mse"] < best_val_interp:
                best_val_interp = val_metrics["interpolation_mse"]
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["train_loss"]),
            "train_recon": float(train_metrics["train_recon"]),
            "train_kl": float(train_metrics["train_kl"]),
            "train_nfe": float(train_metrics["train_nfe"]),
            "train_nfe_encoder": float(train_metrics["train_nfe_encoder"]),
            "train_nfe_latent": float(train_metrics["train_nfe_latent"]),
            "train_time_sec": train_time,
            "val_recon": float(val_metrics["recon_mse"]),
            "val_interp": float(val_metrics["interpolation_mse"]),
            "val_extrap": float(val_metrics["extrapolation_mse"]),
            "val_kl": float(val_metrics["kl"]),
            "val_nfe": float(val_metrics["nfe"]),
            "val_nfe_encoder": float(val_metrics["nfe_encoder"]),
            "val_nfe_latent": float(val_metrics["nfe_latent"]),
            "val_time_sec": val_time,
        }

        wandb.log(row)

        if do_validation:
            console.log(
                f"epoch {epoch:03d} | "
                f"train_time={train_time:.2f}s | "
                f"val_time={val_time:.2f}s | "
                f"loss={row['train_loss']:.6f} | "
                f"val_interp={row['val_interp']:.6f} | "
                f"val_extrap={row['val_extrap']:.6f} | "
                f"train_encoder_nfe={row['train_nfe_encoder']:.2f} | "
                f"train_latent_nfe={row['train_nfe_latent']:.2f} | "
                f"train_total_nfe={row['train_nfe']:.2f} | "
                f"val_encoder_nfe={row['val_nfe_encoder']:.2f} | "
                f"val_latent_nfe={row['val_nfe_latent']:.2f} | "
                f"val_total_nfe={row['val_nfe']:.2f}"
            )
        else:
            console.log(
                f"epoch {epoch:03d} | "
                f"train_time={train_time:.2f}s | "
                f"loss={row['train_loss']:.6f} | "
                f"train_encoder_nfe={row['train_nfe_encoder']:.2f} | "
                f"train_latent_nfe={row['train_nfe_latent']:.2f} | "
                f"train_total_nfe={row['train_nfe']:.2f} | "
                f"validation=skipped"
                
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    _sync_cuda(device)
    test_start = time.time()
    test_metrics = evaluate(model, test_loader, config)
    _sync_cuda(device)
    test_time = time.time() - test_start

    console.log(
        f"test | "
        f"time={test_time:.2f}s | "
        f"interp={test_metrics['interpolation_mse']:.6f} | "
        f"extrap={test_metrics['extrapolation_mse']:.6f} | "
        f"recon={test_metrics['recon_mse']:.6f} | "
        f"encoder_nfe={test_metrics['nfe_encoder']:.2f} | "
        f"latent_nfe={test_metrics['nfe_latent']:.2f} | "
        f"total_nfe={test_metrics['nfe']:.2f}"
    )

    persistence_metrics = evaluate_persistence(test_loader, config)
    console.log(
        f"persistence baseline | "
        f"interp={persistence_metrics['persistence_interpolation_mse']:.6f} | "
        f"extrap={persistence_metrics['persistence_extrapolation_mse']:.6f}"
    )

    wandb.log(
            {
                "test_interp": float(test_metrics["interpolation_mse"]),
                "test_extrap": float(test_metrics["extrapolation_mse"]),
                "test_recon": float(test_metrics["recon_mse"]),
                "test_kl": float(test_metrics["kl"]),
                "test_encoder_nfe": float(test_metrics["nfe_encoder"]),
                "test_latent_nfe": float(test_metrics["nfe_latent"]),
                "test_nfe": float(test_metrics["nfe"]),
                "test_time_sec": test_time,
                "best_val_interp": best_val_interp,
                "persistence_interp": float(persistence_metrics["persistence_interpolation_mse"]),
                "persistence_extrap": float(persistence_metrics["persistence_extrapolation_mse"]),
            }
        )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    run_tag = args.run_name if args.run_name else f"seed{config.seed}_odernn{config.use_ode_rnn}_var{config.is_variational}_{config.method}"
    out_path = results_dir / f"{run_tag}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {k: str(v) for k, v in config.__dict__.items()},
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "best_val_interp": float(best_val_interp),
            "test_time_sec": float(test_time),
        }, f, indent=2)
    console.log(f"saved metrics to {out_path}")
    wandb.finish()

if __name__ == "__main__":
    main()