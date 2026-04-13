import pytest
import torch
import torch.nn as nn
from models.continuous import ODEFunc, ODEBlock


def test_shape_consistency() -> None:
    """Ensures that the ODEBlock preserves the input dimensionality."""
    batch_size, dim = 32, 16
    x = torch.randn(batch_size, dim)
    
    ode_func = ODEFunc(in_features=dim)
    ode_block = ODEBlock(ode_func=ode_func)
    
    try:
        out = ode_block(x)
    except NotImplementedError:
        pytest.fail("ODEBlock forward pass is not implemented.")
        
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_nfe_tracking() -> None:
    """Verifies that Number of Function Evaluations (NFE) strictly increases."""
    dim = 16
    x = torch.randn(1, dim)
    
    ode_func = ODEFunc(in_features=dim)
    ode_block = ODEBlock(ode_func=ode_func)
    
    assert ode_func.nfe == 0, "NFE should initialize to 0."
    
    try:
        _ = ode_block(x)
    except NotImplementedError:
        pytest.fail("ODEBlock forward pass is not implemented.")
        
    # Solving $\frac{dh}{dt}$ takes at least a few steps, so NFE > 1
    assert ode_func.nfe > 1, f"NFE did not accumulate. NFE is {ode_func.nfe}."


def test_gradient_flow() -> None:
    """Tests the adjoint sensitivity method for valid gradient flow."""
    dim = 16
    x = torch.randn(4, dim, requires_grad=True)
    
    ode_func = ODEFunc(in_features=dim)
    ode_block = ODEBlock(ode_func=ode_func)
    
    try:
        out = ode_block(x)
    except NotImplementedError:
        pytest.fail("ODEBlock forward pass is not implemented.")
        
    loss = out.sum()
    loss.backward()
    
    # Assert gradients flowed into the neural network weights
    assert ode_func.net.weight.grad is not None, "Gradients did not flow to ODEFunc."
    # Assert gradients flowed all the way back to the input tensor
    assert x.grad is not None, "Gradients did not flow back to the input tensor."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_device_agnostic() -> None:
    """Ensures models and tensors stay on the requested GPU device."""
    device = torch.device("cuda:0")
    dim = 16
    x = torch.randn(8, dim).to(device)
    
    ode_func = ODEFunc(in_features=dim).to(device)
    ode_block = ODEBlock(ode_func=ode_func).to(device)
    
    try:
        out = ode_block(x)
    except NotImplementedError:
        pytest.fail("ODEBlock forward pass is not implemented.")
        
    assert out.device == device, "Output tensor fell back to CPU."