from dataclasses import dataclass


@dataclass
class ODEConfig:
    """Configuration for Kaby's standalone ODE-RNN time-series experiments."""

    # ODE solver parameters
    solver_type: str = "dopri5"
    atol: float = 1e-3
    rtol: float = 1e-3

    # Model architecture
    in_features: int = 1
    hidden_dim: int = 32
    output_dim: int = 1

    # Workspace flags
    is_time_series: bool = True
    model_type: str = "ode_rnn"   # choices: "ode_rnn", "gru"

    # Training parameters
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50
    seed: int = 28

    # Time-series benchmark parameters
    context_start: float = 0.0
    context_end: float = 5.0
    future_end: float = 10.0
    n_context_points: int = 20
    n_future_points: int = 20
    min_observed_context_points: int = 2
    observation_prob: float = 0.7
    noise_std: float = 0.1

    # Dataset split sizes
    train_size: int = 2048
    val_size: int = 256
    test_size: int = 256