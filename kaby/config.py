from dataclasses import dataclass


@dataclass
class ODEConfig:
    """Configuration for standalone ODE-RNN time-series experiments."""

    # ODE solver parameters
    solver_type: str = "dopri5"
    atol: float = 1e-3
    rtol: float = 1e-3

    # Model architecture
    in_features: int = 1
    hidden_dim: int = 32
    ode_hidden_dim: int | None = None
    gru_time_hidden_dim: int | None = None
    gru_notime_hidden_dim: int | None = None
    input_dim: int = 1
    output_dim: int = 1
    signal_type: str = "sine"

    # Workspace flags
    is_time_series: bool = True
    model_type: str = "ode_rnn"   # choices: "ode_rnn", "gru_time", "gru_notime" (CLI also accepts alias "gru")
    
    # Training parameters
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50
    seed: int = 42
    train_loss_mode: str = "observed_context"  # choices: "observed_context", "full_trajectory"

    # Time-series benchmark parameters
    context_start: float = 0.0
    context_end: float = 5.0
    future_end: float = 10.0
    n_context_points: int = 50
    n_future_points: int = 50
    min_observed_context_points: int = 2
    observation_prob: float = 0.8
    noise_std: float = 0.01

    # Dataset split sizes
    train_size: int = 500
    val_size: int = 100
    test_size: int = 100