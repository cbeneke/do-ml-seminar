import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[32, 32],
        activation=nn.Tanh(),
        output_activation=None,
        initialization='glorot_uniform'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        
        for i in range(len(dims)-1):
            layer = nn.Linear(dims[i], dims[i+1])
            
            # Initialize weights
            if initialization == 'glorot_uniform':
                nn.init.xavier_uniform_(layer.weight)
            elif initialization == 'glorot_normal':
                nn.init.xavier_normal_(layer.weight)
            elif initialization == 'he_uniform':
                nn.init.kaiming_uniform_(layer.weight)
            elif initialization == 'he_normal':
                nn.init.kaiming_normal_(layer.weight)
                
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            
            if i < len(dims)-2:
                layers.append(activation)
            elif output_activation is not None:
                layers.append(output_activation)
                
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
    def get_weights(self):
        """Get all weights of the model as a flat array"""
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])
    
    def set_weights(self, weights):
        """Set weights from a flat array"""
        start = 0
        for param in self.parameters():
            shape = param.data.shape
            size = np.prod(shape)
            param.data = torch.from_numpy(
                weights[start:start+size].reshape(shape)
            ).to(param.device)
            start += size 