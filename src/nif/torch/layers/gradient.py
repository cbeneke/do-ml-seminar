import torch
import torch.nn as nn

class GradientLayer(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x, create_graph=True, retain_graph=None):
        """
        Compute gradients of network outputs with respect to inputs
        
        Args:
            x: Input tensor
            create_graph: If True, graph of the derivative will be constructed
            retain_graph: If True, graph used to compute gradients will be retained
        
        Returns:
            Tensor containing gradients for each output dimension
        """
        x.requires_grad_(True)
        y = self.network(x)
        
        gradients = []
        for i in range(y.shape[-1]):
            grad = torch.autograd.grad(
                y[..., i].sum(),
                x,
                create_graph=create_graph,
                retain_graph=True if i < y.shape[-1]-1 or retain_graph else None
            )[0]
            gradients.append(grad)
            
        return torch.stack(gradients, dim=-1)
    
    def jacobian(self, x):
        """Compute full Jacobian matrix"""
        return self.forward(x, create_graph=False)
    
    def hessian(self, x):
        """Compute Hessian matrix for scalar output functions"""
        if self.network(torch.zeros_like(x)).shape[-1] != 1:
            raise ValueError("Hessian can only be computed for scalar output functions")
        
        first_grads = self.forward(x)
        hessian_rows = []
        
        for i in range(x.shape[-1]):
            grad_slice = first_grads[..., i]
            second_grads = []
            for j in range(x.shape[-1]):
                second_grad = torch.autograd.grad(
                    grad_slice.sum(),
                    x,
                    create_graph=False,
                    retain_graph=True if j < x.shape[-1]-1 else None
                )[0][..., j]
                second_grads.append(second_grad)
            hessian_rows.append(torch.stack(second_grads, dim=-1))
            
        return torch.stack(hessian_rows, dim=-1) 