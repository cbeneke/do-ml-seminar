import torch.nn as nn
from nif.torch.utils import get_activation

class ResNet(nn.Module):
    def __init__(self, units, activation):
        super().__init__()
        self.dense1 = nn.Linear(units, units)
        self.dense2 = nn.Linear(units, units)
        self.activation = get_activation(activation)
        
    def forward(self, x):
        identity = x
        out = self.activation(self.dense1(x))
        out = self.dense2(out)
        return self.activation(out + identity) 