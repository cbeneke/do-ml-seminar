import torch
from typing import Optional, Callable
import torch.optim as optim

def centralize_gradient(x: torch.Tensor, gc_axis: int = 0) -> torch.Tensor:
    """
    Compute the centralized gradient for a tensor
    
    Args:
        x: Input gradient tensor
        gc_axis: Axis along which to compute mean (default=0)
        
    Returns:
        Centralized gradient tensor
    """
    return x - x.mean(dim=tuple(range(gc_axis)) + tuple(range(gc_axis + 1, x.dim())), keepdim=True)

def centralized_gradients_for_optimizer(optimizer_class: Callable) -> Callable:
    """
    Decorator that adds gradient centralization to an optimizer
    
    Args:
        optimizer_class: PyTorch optimizer class to modify
        
    Returns:
        Modified optimizer class with centralized gradients
    """
    def new_step(self, closure: Optional[Callable] = None):
        """Modified step function that centralizes gradients before updating"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Centralize gradients
                if len(p.grad.shape) > 1:
                    p.grad = centralize_gradient(p.grad)
        
        # Call original step function
        return self._original_step(closure)
    
    # Create new optimizer class
    class CentralizedOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._original_step = self.step
            self.step = new_step.__get__(self)
    
    return CentralizedOptimizer

# Pre-defined centralized optimizers
CentralizedAdam = centralized_gradients_for_optimizer(optim.Adam)
CentralizedSGD = centralized_gradients_for_optimizer(optim.SGD)
CentralizedLBFGS = centralized_gradients_for_optimizer(optim.LBFGS) 