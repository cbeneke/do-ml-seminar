import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NIFDataset(Dataset):
    def __init__(self, inputs, outputs, device='cpu', dtype=torch.float32):
        """
        Args:
            inputs: Input data (numpy array or torch tensor)
            outputs: Output data (numpy array or torch tensor)
            device: Device to store the data on
            dtype: Data type for tensors
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        if isinstance(outputs, np.ndarray):
            outputs = torch.from_numpy(outputs)
            
        self.inputs = inputs.to(device=device, dtype=dtype)
        self.outputs = outputs.to(device=device, dtype=dtype)
        
        # Add shape property to match numpy array interface
        self.shape = self.inputs.shape
        # Store numpy version of inputs for array-like access
        self._numpy_inputs = self.inputs.cpu().numpy()
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        if isinstance(idx, (tuple, slice)):
            # For array-like indexing, return numpy array
            return self._numpy_inputs[idx]
        else:
            # For integer indexing (used by DataLoader), return tuple
            return self.inputs[idx], self.outputs[idx]
    
    @staticmethod
    def get_loader(dataset, batch_size=32, shuffle=True, **kwargs):
        """Create a DataLoader from the dataset"""
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    def to(self, device):
        """Move dataset to specified device"""
        self.inputs = self.inputs.to(device)
        self.outputs = self.outputs.to(device)
        return self
    
    @classmethod
    def from_numpy(cls, inputs, outputs, **kwargs):
        """Create dataset from numpy arrays"""
        return cls(inputs, outputs, **kwargs) 