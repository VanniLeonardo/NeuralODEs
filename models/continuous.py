import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class ODEFunc(nn.Module):
    r"""Parameterizes the continuous dynamics of the hidden state.
    
    Computes the time-dependent vector field $f_\\theta(h(t), t)$ for the IVP:
    $ \\frac{dh(t)}{dt} = f_\\theta(h(t), t) $
    
    Args:
        in_features (int): Dimensionality of the hidden state $h(t)$.
        hidden_dim (int): Dimensionality of the internal hidden layers.
    """
    def __init__(self, in_features: int, hidden_dim: int):
        super().__init__()
        self.nfe = 0
        
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
        
        t_expanded = torch.ones_like(h[:, :1]) * t
        h_time = torch.cat([h, t_expanded], dim=1)
        
        return self.net(h_time)


class ODEBlock(nn.Module):
    r"""Integrates the ODEFunc over time $t \\in [0, 1]$ via the adjoint method.
    
    Utilizes the adjoint sensitivity method to allow backpropagation with $\mathcal{O}(1)$ 
    memory footprint.
    
    Args:
        ode_func (nn.Module): The neural network parameterizing the vector field.
        solver_type (str): The ODE solver algorithm (e.g., 'dopri5', 'rk4', 'euler').
        atol (float): Absolute error tolerance for adaptive solvers.
        rtol (float): Relative error tolerance for adaptive solvers.
    """
    def __init__(
        self, 
        ode_func: nn.Module, 
        solver_type: str = "dopri5", 
        atol: float = 1e-3, 
        rtol: float = 1e-3
    ):
        super().__init__()
        self.ode_func = ode_func
        self.solver_type = solver_type
        self.atol = atol
        self.rtol = rtol
        
        self.register_buffer("integration_time", torch.tensor([0.0, 1.0]).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Solves the IVP from $t=0$ to $t=1$."""
        
        t = self.integration_time.type_as(x)
        
        out = odeint(
            func=self.ode_func, 
            y0=x, 
            t=t, 
            rtol=self.rtol, 
            atol=self.atol, 
            method=self.solver_type
        )
        
        # out has shape (len(t), batch_size, dim)
        return out[1]