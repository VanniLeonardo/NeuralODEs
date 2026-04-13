import torch
import torch.nn as nn
from models.continuous import ODEFunc, ODEBlock


class ODENet(nn.Module):
    """Full continuous-depth model for classification tasks.
    
    Args:
        data_dim (int): Dimensionality of the input data (e.g., 2 for circles).
        hidden_dim (int): Dimensionality of the ODE hidden state.
        num_classes (int): Number of output classes (e.g., 2 for binary classification).
        solver_type (str): The ODE solver to use.
    """
    def __init__(self, data_dim: int, hidden_dim: int, num_classes: int, solver_type: str = "dopri5"):
        super().__init__()
        
        # 1. Map input data into the hidden ODE space
        self.downsampling = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 2. The continuous continuous block
        self.ode_func = ODEFunc(in_features=hidden_dim, hidden_dim=hidden_dim)
        self.ode_block = ODEBlock(ode_func=self.ode_func, solver_type=solver_type)
        
        # 3. Map the terminal ODE state to class logits
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reset NFE for tracking
        self.ode_func.nfe = 0
        
        h = self.downsampling(x)
        h_T = self.ode_block(h)
        return self.fc(h_T)


class DiscreteResNet(nn.Module):
    """Baseline discrete Residual Network for comparison.
    
    Uses standard Euler steps: $h_{t+1} = h_t + f(h_t, t)$
    """
    def __init__(self, data_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 5):
        super().__init__()
        self.num_layers = num_layers
        
        self.downsampling = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Using the exact same vector field architecture for fairness
        self.layer_func = ODEFunc(in_features=hidden_dim, hidden_dim=hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.downsampling(x)
        
        # Discrete integration (Euler method with step size dt = 1/num_layers)
        dt = 1.0 / self.num_layers
        for i in range(self.num_layers):
            # Pass a dummy time tensor to match the ODEFunc signature
            t_dummy = torch.tensor([i * dt], device=x.device, dtype=x.dtype)
            h = h + dt * self.layer_func(t_dummy, h)
            
        return self.fc(h)