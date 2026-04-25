from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import torch


def _str2bool(value: str) -> bool:
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


@dataclass
class LatentODEConfig:
    input_dim: int = 1
    signal_type: str = "sine"
    extrap_horizon: float = 10.0
    latent_dim: int = 20
    nhidden: int = 40

    use_ode_rnn: bool = True   # legacy; kept for back-compat, prefer encoder_type
    encoder_type: str = "odernn"   # options: "odernn", "gru_time", "gru_notime"
    is_variational: bool = True

    method: str = "rk4"
    encoder_method: str = "euler"
    rtol: float = 1e-3
    atol: float = 1e-4

    kl_coeff: float = 0.01
    lr: float = 1e-2
    batch_size: int = 64
    # epochs: int = 100
    grad_clip_norm: float = 1.0
    # eval_every: int = 10

    num_train: int = 500       # was 2000
    num_val: int = 100         # was 400
    num_test: int = 100        # was 400
    epochs: int = 30           # was 100
    eval_every: int = 5        # was 10
    

    # num_train: int = 2000
    # num_val: int = 400
    # num_test: int = 400
    seq_len: int = 100
    noise_std: float = 0.01
    train_horizon: float = 5.0

    seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    project_name: str = "latent-ode-phase4-faithful"

    missing_pct: float = 0.2
    burst_missing: bool = False              # <-- add
    burst_length_range: tuple = (2, 8)       # <-- add: (min, max) burst length in steps

    

    @property
    def num_samples(self) -> int:
        return self.num_train + self.num_val + self.num_test

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_args(cls) -> "LatentODEConfig":
        parser = argparse.ArgumentParser()
        parser.add_argument("--input-dim", type=int, default=cls.input_dim)
        parser.add_argument("--latent-dim", type=int, default=cls.latent_dim)
        parser.add_argument("--nhidden", type=int, default=cls.nhidden)
        parser.add_argument("--use-ode-rnn", type=_str2bool, default=cls.use_ode_rnn)
        parser.add_argument("--is-variational", type=_str2bool, default=cls.is_variational)
        parser.add_argument("--method", type=str, default=cls.method)
        parser.add_argument("--encoder-method", type=str, default=cls.encoder_method)
        parser.add_argument("--rtol", type=float, default=cls.rtol)
        parser.add_argument("--atol", type=float, default=cls.atol)
        parser.add_argument("--kl-coeff", type=float, default=cls.kl_coeff)
        parser.add_argument("--lr", type=float, default=cls.lr)
        parser.add_argument("--batch-size", type=int, default=cls.batch_size)
        parser.add_argument("--epochs", type=int, default=cls.epochs)
        parser.add_argument("--grad-clip-norm", type=float, default=cls.grad_clip_norm)
        parser.add_argument("--eval-every", type=int, default=cls.eval_every)
        parser.add_argument("--num-train", type=int, default=cls.num_train)
        parser.add_argument("--num-val", type=int, default=cls.num_val)
        parser.add_argument("--num-test", type=int, default=cls.num_test)
        parser.add_argument("--seq-len", type=int, default=cls.seq_len)
        parser.add_argument("--missing-pct", type=float, default=cls.missing_pct)
        parser.add_argument("--noise-std", type=float, default=cls.noise_std)
        parser.add_argument("--train-horizon", type=float, default=cls.train_horizon)
        parser.add_argument("--extrap-horizon", type=float, default=cls.extrap_horizon)
        parser.add_argument("--seed", type=int, default=cls.seed)
        parser.add_argument("--num-workers", type=int, default=cls.num_workers)
        parser.add_argument("--device", type=str, default=cls.device)
        parser.add_argument("--project-name", type=str, default=cls.project_name)
        args = parser.parse_args()
        return cls(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            nhidden=args.nhidden,
            use_ode_rnn=args.use_ode_rnn,
            is_variational=args.is_variational,
            method=args.method,
            encoder_method=args.encoder_method,
            rtol=args.rtol,
            atol=args.atol,
            kl_coeff=args.kl_coeff,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            grad_clip_norm=args.grad_clip_norm,
            eval_every=args.eval_every,
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test,
            seq_len=args.seq_len,
            missing_pct=args.missing_pct,
            noise_std=args.noise_std,
            train_horizon=args.train_horizon,
            extrap_horizon=args.extrap_horizon,
            seed=args.seed,
            num_workers=args.num_workers,
            device=args.device,
            project_name=args.project_name,
        )
