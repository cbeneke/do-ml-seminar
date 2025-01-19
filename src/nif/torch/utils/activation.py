import torch.nn as nn

def get_activation(name: str) -> nn.Module:
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