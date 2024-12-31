import torch
import torch.nn as nn
from typing import Dict, Any, List


class NIF(nn.Module):
    def __init__(
        self,
        cfg_shape_net: Dict[str, Any],
        cfg_parameter_net: Dict[str, Any],
        mixed_policy: str = 'float32'
    ):
        """
        Neural Implicit Flow model in PyTorch
        
        Args:
            cfg_shape_net: Configuration for shape network
            cfg_parameter_net: Configuration for parameter network
            mixed_policy: Mixed precision policy (currently not used in PyTorch version)
        """
        super().__init__()
        
        # Store configurations
        self.cfg_shape_net = cfg_shape_net
        self.cfg_parameter_net = cfg_parameter_net
        
        # Build networks
        self.shape_net = self._build_shape_net()
        self.parameter_net = self._build_parameter_net()
    
    def _build_shape_net(self) -> nn.Module:
        """Build the shape network"""
        layers: List[nn.Module] = []
        input_dim = self.cfg_shape_net["input_dim"]
        units = self.cfg_shape_net["units"]
        nlayers = self.cfg_shape_net["nlayers"]
        activation = self._get_activation(self.cfg_shape_net["activation"])
        
        # Input layer
        layers.append(nn.Linear(input_dim, units))
        layers.append(activation)
        
        # Hidden layers
        for _ in range(nlayers - 1):
            layers.append(nn.Linear(units, units))
            layers.append(activation)
        
        # Output layer
        layers.append(nn.Linear(units, self.cfg_shape_net["output_dim"]))
        
        return nn.Sequential(*layers)
    
    def _build_parameter_net(self) -> nn.Module:
        """Build the parameter network"""
        layers: List[nn.Module] = []
        input_dim = self.cfg_parameter_net["input_dim"]
        units = self.cfg_parameter_net["units"]
        nlayers = self.cfg_parameter_net["nlayers"]
        activation = self._get_activation(self.cfg_parameter_net["activation"])
        
        # Input layer
        layers.append(nn.Linear(input_dim, units))
        layers.append(activation)
        
        # Hidden layers
        for _ in range(nlayers - 1):
            layers.append(nn.Linear(units, units))
            layers.append(activation)
        
        # Output layer
        layers.append(nn.Linear(units, self.cfg_parameter_net["latent_dim"]))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        if name.lower() == 'relu':
            return nn.ReLU()
        elif name.lower() == 'tanh':
            return nn.Tanh()
        elif name.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif name.lower() == 'swish':
            return nn.SiLU()  # PyTorch's SiLU is the same as Swish
        else:
            raise ValueError(f"Unsupported activation function: {name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the NIF model
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Split input into spatial and temporal components
        spatial = x[:, :self.cfg_shape_net["input_dim"]]
        temporal = x[:, self.cfg_shape_net["input_dim"]:]
        
        # Get latent parameters from parameter network
        latent = self.parameter_net(temporal)
        
        # Combine spatial input with latent parameters
        shape_input = torch.cat([spatial, latent], dim=-1)
        
        # Pass through shape network
        return self.shape_net(shape_input)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            "cfg_shape_net": self.cfg_shape_net,
            "cfg_parameter_net": self.cfg_parameter_net
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NIF':
        """Create model from configuration"""
        return cls(**config) 