from dataclasses import dataclass
from typing import Tuple


@dataclass
class SolverAblationConfig:
    """Configuration for the solver ablation study."""

    # Sweep parameters
    fixed_solvers: Tuple[str, ...] = ("euler", "midpoint", "rk4")
    adaptive_solvers: Tuple[str, ...] = ("bosh3", "dopri5", "dopri8")
    tolerances: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4)  # atol == rtol

    # Shared training settings (mirrors scalar fields of ODEConfig)
    dataset: str = "circles"
    n_samples: int = 1000
    val_split: float = 0.2
    noise: float = 0.05
    hidden_dim: int = 32
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50


@dataclass
class ODEConfig:
    """Central configuration for Neural ODE experiments."""

    # Data Parameters
    dataset: str = "circles"     # one of: circles, spirals, moons
    n_samples: int = 1000
    val_split: float = 0.2
    noise: float = 0.05

    # ODE Solver Parameters
    solver_type: str = "dopri5"  # dopri5 is Runge-Kutta 4(5)
    atol: float = 1e-3
    rtol: float = 1e-3
    integration_time: Tuple[float, float] = (0.0, 1.0)

    # Network Architecture
    in_features: int = 2         # fixed at 2 for all synthetic 2D datasets
    hidden_dim: int = 32

    # Member 3 (ANODE) & Member 5 (Time-Series) overrides
    augment_dim: int = 0         # Number of zero-dimensions to append
    is_time_series: bool = False # Flag for ODE-RNN logic

    # Training Parameters
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50