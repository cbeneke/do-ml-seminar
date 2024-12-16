import torch
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

class GTCF:
    @staticmethod
    def create(params, **kwargs):
        """
        Create a GTCF optimizer with sensible defaults for NIFs
        
        Args:
            params: Iterator of parameters to optimize
            **kwargs: Override default optimizer settings
        """
        defaults = GTCF.get_defaults()
        defaults.update(kwargs)
        return GTCFOptimizer(params, **defaults)
    
    @staticmethod
    def step(optimizer, closure) -> Optional[torch.Tensor]:
        """
        Perform a single optimization step
        
        Args:
            optimizer: The GTCF optimizer
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
            'max_trust_radius': 10.0,
            'min_trust_radius': 1e-8,
            'eta': 0.15,
            'initial_trust_radius': 1.0,
            'max_iterations': 50
        }

class GTCFOptimizer(optim.Optimizer):
    def __init__(self, params, lr=1.0, max_trust_radius=10.0, min_trust_radius=1e-8,
                 eta=0.15, initial_trust_radius=1.0, max_iterations=50):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        
        self.max_trust_radius = max_trust_radius
        self.min_trust_radius = min_trust_radius
        self.eta = eta
        self.trust_radius = initial_trust_radius
        self.max_iterations = max_iterations
        
        # Initialize state
        self.state['step'] = 0
        self.state['prev_loss'] = None
        self.state['prev_params'] = None
        self.state['prev_grads'] = None
    
    def _gather_flat_grad(self) -> torch.Tensor:
        """Gather all gradients into a single flat tensor"""
        views = []
        for p in self.param_groups[0]['params']:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)
    
    def _gather_flat_params(self) -> torch.Tensor:
        """Gather all parameters into a single flat tensor"""
        views = []
        for p in self.param_groups[0]['params']:
            views.append(p.data.view(-1))
        return torch.cat(views, 0)
    
    def _distribute_flat_params(self, flat_params: torch.Tensor):
        """Distribute flat parameters back to model parameters"""
        offset = 0
        for p in self.param_groups[0]['params']:
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view_as(p.data))
            offset += numel
    
    def _compute_quadratic_model(self, g: torch.Tensor, s: torch.Tensor, 
                               y: torch.Tensor) -> torch.Tensor:
        """Compute quadratic model approximation"""
        Bs = y
        return g.dot(s) + 0.5 * s.dot(Bs)
    
    def _conjugate_gradient_solve(self, g: torch.Tensor, H: torch.Tensor, 
                                max_iterations: int) -> torch.Tensor:
        """Solve the trust-region subproblem using conjugate gradient method"""
        x = torch.zeros_like(g)
        r = g.clone()
        p = -r.clone()
        
        for i in range(max_iterations):
            Hp = H @ p
            alpha = r.dot(r) / p.dot(Hp)
            x_new = x + alpha * p
            
            if torch.norm(x_new) > self.trust_radius:
                # Find the intersection with trust region boundary
                a = p.dot(p)
                b = 2 * x.dot(p)
                c = x.dot(x) - self.trust_radius**2
                alpha = (-b + torch.sqrt(b**2 - 4*a*c)) / (2*a)
                x = x + alpha * p
                break
                
            x = x_new
            r_new = r + alpha * Hp
            beta = r_new.dot(r_new) / r.dot(r)
            p = -r_new + beta * p
            r = r_new
            
            if torch.norm(r) < 1e-10:
                break
                
        return x
    
    def step(self, closure) -> Optional[torch.Tensor]:
        """Perform a single optimization step"""
        loss = closure()
        current_loss = loss.item()
        
        # Get current gradients and parameters
        flat_grad = self._gather_flat_grad()
        flat_params = self._gather_flat_params()
        
        # If this is the first step, just store current state
        if self.state['prev_loss'] is None:
            self.state['prev_loss'] = current_loss
            self.state['prev_params'] = flat_params.clone()
            self.state['prev_grads'] = flat_grad.clone()
            return loss
        
        # Compute step using trust-region conjugate gradient
        s = flat_params - self.state['prev_params']
        y = flat_grad - self.state['prev_grads']
        
        # Approximate Hessian using BFGS update
        H = torch.ger(y, y) / y.dot(s)
        
        # Solve trust-region subproblem
        step = self._conjugate_gradient_solve(flat_grad, H, self.max_iterations)
        
        # Try the step
        self._distribute_flat_params(flat_params + step)
        trial_loss = closure().item()
        
        # Compute actual vs predicted reduction
        actual_reduction = self.state['prev_loss'] - trial_loss
        predicted_reduction = -self._compute_quadratic_model(flat_grad, step, y)
        
        # Update trust region
        rho = actual_reduction / predicted_reduction if predicted_reduction > 0 else -1
        
        if rho < 0.25:
            self.trust_radius = max(0.25 * self.trust_radius, self.min_trust_radius)
        elif rho > 0.75 and torch.norm(step) >= 0.99 * self.trust_radius:
            self.trust_radius = min(2.0 * self.trust_radius, self.max_trust_radius)
        
        # Accept or reject step
        if rho > self.eta:
            self.state['prev_loss'] = trial_loss
            self.state['prev_params'] = flat_params + step
            self.state['prev_grads'] = self._gather_flat_grad()
        else:
            self._distribute_flat_params(flat_params)
        
        self.state['step'] += 1
        return loss 