import torch.nn as nn
from nif.torch.utils import get_activation

class Shortcut(nn.Module):
    def __init__(self, units, activation):
        super().__init__()
        self.dense = nn.Linear(units, units)
        self.activation = get_activation(activation)
        
    def forward(self, x):
        return x + self.activation(self.dense(x)) 