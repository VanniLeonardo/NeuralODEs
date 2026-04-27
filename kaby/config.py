from dataclasses import dataclass
from typing import Tuple


@dataclass
class ODEConfig:
    """Central configuration for Neural ODE experiments."""
    
    # ODE Solver Parameters
    solver_type: str = "dopri5"  # dopri5 is Runge-Kutta 4(5)
    atol: float = 1e-3
    rtol: float = 1e-3
    integration_time: Tuple[float, float] = (0.0, 1.0)
    
    # Network Architecture
    in_features: int = 2         # e.g., 2 for 2D synthetic datasets
    hidden_dim: int = 32
    output_dim: int = 1
    
    # Member 3 (ANODE) & Member 5 (Time-Series) overrides
    augment_dim: int = 0         # Number of zero-dimensions to append
    is_time_series: bool = False # Flag for ODE-RNN logic
    model_type: str = "ode_rnn"

    # Training Parameters
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50
    seed: int = 28

    # Time-series Parameters
    context_start: float = 0.0
    context_end: float = 5.0
    future_end: float = 10.0
    n_context_points: int = 20
    n_future_points: int = 20
    min_observed_context_points: int = 2
    observation_prob: float = 0.7
    noise_std: float = 0.1
    train_size: int = 2048
    val_size: int = 256
    test_size: int = 256