from dataclasses import dataclass
from typing import Tuple


@dataclass
class SolverAblationConfig:
    """Configuration for the solver ablation study."""

    # Sweep parameters
    fixed_solvers: Tuple[str, ...] = ("euler", "midpoint", "rk4")
    adaptive_solvers: Tuple[str, ...] = ("bosh3", "dopri5", "dopri8")
    tolerances: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4)  # atol == rtol

    dataset: str = "circles"
    n_samples: int = 1000
    val_split: float = 0.2
    noise: float = 0.05
    hidden_dim: int = 32
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50


@dataclass
class ANODECirclesConfig:
    """Configuration for NODE vs ANODE experiments on concentric circles."""

    dataset: str = "circles"
    n_samples: int = 1000
    val_split: float = 0.2
    noise: float = 0.05

    solver_type: str = "dopri5"
    atol: float = 1e-3
    rtol: float = 1e-3

    hidden_dim: int = 2
    augment_dims: Tuple[int, ...] = (0, 1, 2, 5)

    seeds: Tuple[int, ...] = (0, 1, 2)
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 200

    project: str = "neural-odes-30562"
    results_dir: str = "results/anode"
    log_every: int = 10
    write_results: bool = True


@dataclass
class ODEConfig:
    """Central configuration for Neural ODE experiments."""

    # Data Parameters
    dataset: str = "circles"  # one of: circles, spirals, moons
    n_samples: int = 1000
    val_split: float = 0.2
    noise: float = 0.05

    # ODE Solver Parameters
    solver_type: str = "dopri5"
    atol: float = 1e-3
    rtol: float = 1e-3
    integration_time: Tuple[float, float] = (0.0, 1.0)

    # Network Architecture
    in_features: int = 2  # fixed at 2 for all synthetic 2D datasets
    hidden_dim: int = 32

    augment_dim: int = 0  # Number of zero-dimensions to append
    is_time_series: bool = False  # Flag for ODE-RNN logic

    # Training Parameters
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50

    # Time-series model options
    ode_hidden_dim: int | None = None
    gru_time_hidden_dim: int | None = None
    gru_notime_hidden_dim: int | None = None
    input_dim: int = 1
    output_dim: int = 1
    signal_type: str = "sine"
    model_type: str = "ode_rnn"

    # Time-series training options
    seed: int = 42
    train_loss_mode: str = "observed_context"

    # Time-series benchmark parameters
    context_start: float = 0.0
    context_end: float = 5.0
    future_end: float = 10.0
    n_context_points: int = 50
    n_future_points: int = 50
    min_observed_context_points: int = 2
    observation_prob: float = 0.8
    noise_std: float = 0.01

    # Time-series split sizes
    train_size: int = 500
    val_size: int = 100
    test_size: int = 100
