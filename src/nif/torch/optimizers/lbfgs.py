import torch
import torch.optim as optim
from typing import Dict, Any, Optional

class LBFGS:
    @staticmethod
    def create(params, **kwargs):
        """
        Create an L-BFGS optimizer with sensible defaults for NIFs
        
        Args:
            params: Iterator of parameters to optimize
            **kwargs: Override default optimizer settings
        """
        defaults = {
            'lr': 1.0,
            'max_iter': 20,
            'max_eval': 25,
            'tolerance_grad': 1e-7,
            'tolerance_change': 1e-9,
            'history_size': 100,
            'line_search_fn': 'strong_wolfe'
        }
        
        defaults.update(kwargs)
        return optim.LBFGS(params, **defaults)
    
    @staticmethod
    def step(optimizer: optim.LBFGS, closure) -> Optional[torch.Tensor]:
        """
        Perform a single optimization step
        
        Args:
            optimizer: The L-BFGS optimizer
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            The loss value if the step was successful, None otherwise
        """
        return optimizer.step(closure)
    
    @staticmethod
    def get_defaults() -> Dict[str, Any]:
        """Get default optimizer settings"""
        return {
            'lr': 1.0,
            'max_iter': 20,
            'max_eval': 25,
            'tolerance_grad': 1e-7,
            'tolerance_change': 1e-9,
            'history_size': 100,
            'line_search_fn': 'strong_wolfe'
        } 