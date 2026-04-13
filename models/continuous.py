import torch
import torch.nn as nn


class ODEFunc(nn.Module):
    """Parameterizes the continuous dynamics of the hidden state.
    
    Computes the time-dependent vector field $f_\\theta(h(t), t)$ for the IVP:
    $ \\frac{dh(t)}{dt} = f_\\theta(h(t), t) $
    
    Args:
        in_features (int): Dimensionality of the hidden state $h(t)$.
        hidden_dim (int): Dimensionality of the internal hidden layers.
    """
    def __init__(self, in_features: int, hidden_dim: int):
        super().__init__()
        self.nfe = 0
        
        # The input dimension is in_features + 1 to concatenate the time scalar $t$ for ANODEs
        self.net = nn.Sequential(
            nn.Linear(in_features + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features)
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Evaluates the vector field at state $h$ and time $t$."""
        self.nfe += 1
        
        # $t$ is passed by the solver as a 0D scalar tensor.
        # We expand it to match the batch dimension of $h$.
        t_expanded = torch.ones_like(h[:, :1]) * t
        
        # Concatenate along the feature dimension
        h_time = torch.cat([h, t_expanded], dim=1)
        
        return self.net(h_time)


class ODEBlock(nn.Module):
    """Integrates the ODEFunc over time $t \\in[0, 1]$ via the adjoint method.
    
    Args:
        ode_func (nn.Module): The neural network parameterizing the vector field.
    """
    def __init__(self, ode_func: nn.Module):
        super().__init__()
        self.ode_func = ode_func
        self.integration_time = torch.tensor([0.0, 1.0]).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Solves the IVP from $t=0$ to $t=1$."""
        raise NotImplementedError("ODE integration not yet implemented.")