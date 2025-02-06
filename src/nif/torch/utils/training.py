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
        
        self.time_start = time.time()

    def log_progress(self, model: torch.nn.Module, epoch: int, loss: float):
        """Log training progress and create visualizations.
        
        Args:
            model: The PyTorch model
            epoch: Current epoch number
            loss: Training loss for this epoch
        """
        tnow = time.time()
        time_end = tnow - self.time_start
        self.time_start = tnow  # Reset for next epoch
        
        logging.info(
            f"Epoch {epoch:6d}: avg.loss pe = {loss:4.3e}, "
            f"time elapsed = {(tnow - self.train_begin_time) / 3600.0:4.3f} hours"
        )
        self.history_loss.append(loss)
        
        # Plot loss history
        plt.figure()
        plt.semilogy(self.history_loss)
        plt.xlabel(f'epoch: per {self.print_figure_epoch} epochs')
        plt.ylabel('MSE loss')
        plt.savefig('loss.pdf')
        plt.close()
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            # Get the full dataset
            full_dataset = model.train_loader.dataset
            inputs = full_dataset.tensors[0][:2000].to(next(model.parameters()).device)  # First 2000 points
            targets = full_dataset.tensors[1][:2000]
            
            # Make predictions
            u_pred = model(inputs).cpu().numpy().reshape(10, 200)
            u_true = targets.numpy().reshape(10, 200)
        
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
        plt.savefig('vis.pdf')
        plt.close()

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
        """Save a model checkpoint.
        
        Args:
            model: The PyTorch model to save
            optimizer: The optimizer
            epoch: Current epoch number
        """
        print(f'save checkpoint epoch: {epoch}...')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': self.history_loss,
            },
            f"{self.save_dir}/ckpt-{epoch}.weights.pth"
        )

def get_scheduler(optimizer, nepoch):
    """Creates a learning rate scheduler that matches the TensorFlow implementation.
    
    The schedule is:
    - First 1000 epochs: initial learning rate (1e-4)
    - 1000-2000 epochs: 1e-3
    - 2000-4000 epochs: 5e-4
    - After 4000 epochs: 1e-4
    """
    def lr_lambda(epoch):
        if epoch < 1000:
            return 1.0  # Keep initial learning rate
        elif epoch < 2000:
            return 10.0  # 1e-3 (10 times the initial rate)
        elif epoch < 4000:
            return 5.0  # 5e-4 (5 times the initial rate)
        else:
            return 1.0  # 1e-4 (back to initial rate)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    logger: TrainingLogger,
    scaler: torch.cuda.amp.GradScaler = None,
):
    """Train a model.

    Args:
        model: The model to train
        train_loader: The data loader for training data
        optimizer: The optimizer to use
        n_epochs: Number of epochs to train for
        logger: The logger to use for tracking progress
        scaler: Optional gradient scaler for mixed precision training
    """
    # Store train_loader in model for visualization
    model.train_loader = train_loader
    
    # Create scheduler
    scheduler = get_scheduler(optimizer, n_epochs)
    
    # Training loop
    history = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_inputs, batch_targets in train_loader:
            # Move data to the same device as model
            batch_inputs = batch_inputs.to(next(model.parameters()).device)
            batch_targets = batch_targets.to(next(model.parameters()).device)
            
            optimizer.zero_grad()

            # Forward pass with automatic mixed precision if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_inputs)
                    loss = torch.nn.functional.mse_loss(outputs, batch_targets)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_inputs)
                loss = torch.nn.functional.mse_loss(outputs, batch_targets)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Calculate average loss for this epoch
        avg_loss = total_loss / num_batches
        history.append(avg_loss)

        # Update learning rate
        scheduler.step()

        # Log progress
        if (epoch + 1) % logger.display_epoch == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.8f}, LR: {scheduler.get_last_lr()[0]:.8f}")

        # Additional logging if needed
        if (epoch + 1) % logger.print_figure_epoch == 0:
            logger.log_progress(model, epoch + 1, avg_loss)

        # Save checkpoint if needed
        if (epoch + 1) % logger.checkpt_epoch == 0:
            logger.save_checkpoint(model, optimizer, epoch + 1)

    return history 