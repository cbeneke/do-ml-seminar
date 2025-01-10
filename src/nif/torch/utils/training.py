import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, Any, Optional

class TrainingLogger:
    def __init__(self, 
                 display_epoch: int = 100,
                 print_figure_epoch: int = 100,
                 checkpt_epoch: int = 1000,
                 save_dir: str = './saved_weights',
                 n_epochs: int = 5000):
        """
        Training logger for PyTorch models
        
        Args:
            display_epoch: How often to display training info
            print_figure_epoch: How often to save figures
            checkpt_epoch: How often to save checkpoints
            save_dir: Directory to save checkpoints
            n_epochs: Total number of epochs for training
        """
        self.display_epoch = display_epoch
        self.print_figure_epoch = print_figure_epoch
        self.checkpt_epoch = checkpt_epoch
        self.save_dir = save_dir
        self.n_epochs = n_epochs
        
        self.train_begin_time = time.time()
        self.history_loss = []
        
        logging.basicConfig(
            filename='./log',
            level=logging.INFO,
            format='%(message)s'
        )
    
    def on_epoch_begin(self):
        """Call at the beginning of each epoch"""
        self.time_start = time.time()
    
    def on_epoch_end(self, epoch: int, loss: float, model: torch.nn.Module,
                    train_data: torch.Tensor, batch_size: int):
        """
        Call at the end of each epoch
        
        Args:
            epoch: Current epoch number
            loss: Training loss for this epoch
            model: The PyTorch model
            train_data: Training data tensor
            batch_size: Batch size used in training
        """
        if epoch % self.display_epoch == 0:
            tnow = time.time()
            time_end = tnow - self.time_start
            logging.info(
                f"Epoch {epoch:6d}: avg.loss pe = {loss:4.3e}, "
                f"{int(batch_size / time_end):d} points/sec, "
                f"time elapsed = {(tnow - self.train_begin_time) / 3600.0:4.3f} hours"
            )
            self.history_loss.append(loss)
        
        if epoch % self.print_figure_epoch == 0:
            # Plot loss history
            plt.figure()
            plt.semilogy(self.history_loss)
            plt.xlabel(f'epoch: per {self.print_figure_epoch} epochs')
            plt.ylabel('MSE loss')
            plt.savefig('./loss.png')
            plt.close()
            
            # Make predictions
            model.eval()
            with torch.no_grad():
                inputs = train_data[0].to(next(model.parameters()).device)
                u_pred = model(inputs).cpu().numpy().reshape(10, 200)
                u_true = train_data[1].cpu().numpy().reshape(10, 200)

            model.train()
            
            # Create visualization
            tt = np.linspace(0, 100, 10)
            xx = np.linspace(0, 1, 200)
            tt, xx = np.meshgrid(tt, xx, indexing='ij')
            
            _, axs = plt.subplots(1, 3, figsize=(16, 4))
            
            im1 = axs[0].contourf(tt, xx, u_true, vmin=-5, vmax=5, levels=50, cmap='seismic')
            plt.colorbar(im1, ax=axs[0])
            
            im2 = axs[1].contourf(tt, xx, u_pred, vmin=-5, vmax=5, levels=50, cmap='seismic')
            plt.colorbar(im2, ax=axs[1])
            
            im3 = axs[2].contourf(tt, xx, (u_pred - u_true), vmin=-5, vmax=5, levels=50, cmap='seismic')
            plt.colorbar(im3, ax=axs[2])
            
            axs[0].set_xlabel('t')
            axs[0].set_ylabel('x')
            axs[0].set_title('true')
            axs[1].set_title('pred')
            axs[2].set_title('error')
            plt.savefig('vis.png')
            plt.close()
        
        if epoch % self.checkpt_epoch == 0 or epoch == self.n_epochs - 1:
            print(f'save checkpoint epoch: {epoch}...')
            torch.save(
                model.state_dict(),
                f"{self.save_dir}/ckpt-{epoch}.weights.pth"
            )

def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    logger: Optional[TrainingLogger] = None
) -> Dict[str, Any]:
    """
    Train a PyTorch model
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        optimizer: PyTorch optimizer
        n_epochs: Number of epochs to train
        device: Device to train on
        logger: Optional TrainingLogger instance
    
    Returns:
        Dictionary containing training history
    """
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    
    if logger is None:
        logger = TrainingLogger()
    
    for epoch in range(n_epochs):
        logger.on_epoch_begin()
        
        # Training loop
        model.train()
        total_loss = 0
        for _, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        
        # Log progress
        logger.on_epoch_end(
            epoch,
            avg_loss,
            model,
            train_loader.dataset.tensors,
            train_loader.batch_size
        )
    
    return {"loss_history": logger.history_loss} 