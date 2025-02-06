import torch
import torch.nn as nn
from typing import Dict, Any, List
from torch.amp import autocast, GradScaler

from nif.torch import utils
from nif.torch.layers import StaticDense, ResNet, Shortcut

class MixedPrecisionPolicy:
    """PyTorch equivalent of tf.keras.mixed_precision.Policy"""
    def __init__(self, policy_name: str):
        self.policy_name = policy_name
        self.use_amp = policy_name in ['float16', 'bfloat16']
        
        if policy_name == 'float16':
            self.compute_dtype = torch.float16
            self.variable_dtype = torch.float32  # Variables still stored in float32
        elif policy_name == 'bfloat16':
            self.compute_dtype = torch.bfloat16
            self.variable_dtype = torch.float32
        else:  # default to float32
            self.compute_dtype = torch.float32
            self.variable_dtype = torch.float32

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
            mixed_policy: Mixed precision policy ('float32', 'float16', or 'bfloat16')
        """
        super().__init__()
        
        # Store configurations
        self.cfg_shape_net = cfg_shape_net
        self.cfg_parameter_net = cfg_parameter_net
        
        # Set up mixed precision policy and scaler for AMP
        self.policy = MixedPrecisionPolicy(mixed_policy)
        self.scaler = GradScaler(enabled=self.policy.use_amp)
        
        # Build networks
        self.shape_net = self._build_shape_net()
        self.parameter_net = self._build_parameter_net()
        
        # Move model to GPU if available and set dtype
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.to(device=self.device, dtype=self.policy.variable_dtype)
    
    def _build_parameter_net(self) -> nn.Module:
        """Build the parameter network with regularization options"""
        layers: List[nn.Module] = []

        # First Layer
        first_layer = nn.Linear(
            self.cfg_parameter_net["input_dim"],
            self.cfg_parameter_net["units"]
        )
        nn.init.trunc_normal_(first_layer.weight, std=0.1)
        nn.init.trunc_normal_(first_layer.bias, std=0.1)
        layers.append(first_layer)
        layers.append(utils.get_activation(self.cfg_parameter_net["activation"]))
        
        # Hidden Layers
        HiddenLayer = ResNet if self.cfg_parameter_net.get("use_resblock", False) else Shortcut
        for i in range(self.cfg_parameter_net["nlayers"]):
            hidden = HiddenLayer(
                units=self.cfg_parameter_net["units"],
                activation=self.cfg_parameter_net["activation"]
            )
            layers.append(hidden)
        
        # Bottleneck layer
        bottleneck = nn.Linear(
            self.cfg_parameter_net["units"],
            self.cfg_parameter_net["latent_dim"]
        )
        nn.init.trunc_normal_(bottleneck.weight, std=0.1)
        nn.init.trunc_normal_(bottleneck.bias, std=0.1)
        layers.append(bottleneck)
        layers.append(utils.get_activation(self.cfg_parameter_net["activation"]))
        
        # Output layer
        output = nn.Linear(
            self.cfg_parameter_net["latent_dim"],
            utils.get_parameter_net_output_dim(self.cfg_shape_net)
        )
        nn.init.trunc_normal_(output.weight, std=0.1)
        nn.init.trunc_normal_(output.bias, std=0.1)
        layers.append(output)
        
        return nn.Sequential(*layers)
    
    def _build_shape_net(self) -> nn.Module:
        """Build the shape network with static weights"""
        layers: List[nn.Module] = []
        
        # First layer
        layers.append(StaticDense(
            units=self.cfg_shape_net["units"],
            activation=utils.get_activation(self.cfg_shape_net["activation"]),
            weights_from=0,
            weights_to=self.cfg_shape_net["input_dim"] * self.cfg_shape_net["units"],
            bias_offset=utils.get_weights_dim(self.cfg_shape_net),
            biases_from=0,
            biases_to=self.cfg_shape_net["units"]
        ))
        
        # Hidden layers
        for i in range(self.cfg_shape_net["nlayers"]):
            weights_from = (self.cfg_shape_net["input_dim"] * self.cfg_shape_net["units"] + 
                           i * self.cfg_shape_net["units"]**2)
            weights_to = (self.cfg_shape_net["input_dim"] * self.cfg_shape_net["units"] + 
                         (i + 1) * self.cfg_shape_net["units"]**2)
            
            layers.append(StaticDense(
                units=self.cfg_shape_net["units"],
                activation=utils.get_activation(self.cfg_shape_net["activation"]),
                weights_from=weights_from,
                weights_to=weights_to,
                bias_offset=utils.get_weights_dim(self.cfg_shape_net),
                biases_from=(i+1) * self.cfg_shape_net["units"],
                biases_to=(i+2) * self.cfg_shape_net["units"]
            ))
        
        # Output layer
        weights_from = (self.cfg_shape_net["input_dim"] * self.cfg_shape_net["units"] + 
                       self.cfg_shape_net["nlayers"] * self.cfg_shape_net["units"]**2)
        weights_to = (weights_from + 
                     self.cfg_shape_net["output_dim"] * self.cfg_shape_net["units"])
        
        layers.append(StaticDense(
            units=self.cfg_shape_net["output_dim"],
            activation=None,
            weights_from=weights_from,
            weights_to=weights_to,
            bias_offset=utils.get_weights_dim(self.cfg_shape_net),
            biases_from=self.cfg_shape_net["nlayers"] * self.cfg_shape_net["units"],
            biases_to=(self.cfg_shape_net["nlayers"] * self.cfg_shape_net["units"] + 
                      self.cfg_shape_net["output_dim"])
        ))
        
        # Create a wrapper module that only passes the first parameter as input
        class ShapeNetWrapper(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = nn.ModuleList(layers)
                
            def forward(self, inputs):
                x, parameters = inputs[0], inputs[1:]
                for layer in self.layers:
                    x = layer([x, *parameters])
                return x
                
        return ShapeNetWrapper(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the NIF model"""
        # Move input to correct device and dtype
        x = x.to(device=self.device, dtype=self.policy.variable_dtype)
        
        # Use autocast only when using mixed precision
        if self.policy.use_amp:
            with autocast():
                # Split input
                parameter_input = x[:, :self.cfg_parameter_net["input_dim"]]
                shape_input = x[:, self.cfg_parameter_net["input_dim"]:]
                
                # Get parameters from parameter network
                parameters = self.parameter_net(parameter_input)
                
                # Pass through shape network
                return self.shape_net([shape_input, parameters])
        else:
            # Regular forward pass without mixed precision
            parameter_input = x[:, :self.cfg_parameter_net["input_dim"]]
            shape_input = x[:, self.cfg_parameter_net["input_dim"]:]
            parameters = self.parameter_net(parameter_input)
            return self.shape_net([shape_input, parameters])
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            "cfg_shape_net": self.cfg_shape_net,
            "cfg_parameter_net": self.cfg_parameter_net,
            "mixed_policy": self.policy.policy_name
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NIF':
        """Create model from configuration"""
        return cls(**config)