import torch
import torch.nn as nn

class StaticSIREN(nn.Module):
    def __init__(self, units, activation=None, weights_from=0, weights_to=0, 
                 bias_offset=0, biases_from=0, biases_to=0, omega_0=30.0):
        super().__init__()
        self.units = units
        self.weights_from = weights_from
        self.weights_to = weights_to
        self.bias_offset = bias_offset
        self.biases_from = biases_from
        self.biases_to = biases_to
        self.omega_0 = omega_0
        self.activation = activation

    def forward(self, inputs):
        x, parameters = inputs
        
        # Extract weights and biases from parameters
        weights = parameters[:, self.weights_from:self.weights_to]
        biases = parameters[:, self.bias_offset + self.biases_from:self.bias_offset + self.biases_to]
        
        # Reshape weights to matrix form
        batch_size = x.shape[0]
        in_features = x.shape[-1]
        weights = weights.view(batch_size, in_features, self.units)
        
        # Compute output
        output = torch.matmul(x.unsqueeze(1), weights).squeeze(1)
        if self.activation == 'sine':
            output = self.omega_0 * output + biases
            output = torch.sin(output)
            
        return output