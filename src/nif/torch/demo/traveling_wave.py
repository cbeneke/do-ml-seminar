import torch
import numpy as np
from ..layers import MLP, GradientLayer
from ..data import NIFDataset
from ..optimizers import LBFGS

class TravelingWave:
    def __init__(self, hidden_dims=[32, 32], device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Create network
        self.network = MLP(
            input_dim=2,  # (x, t)
            output_dim=1,  # u(x,t)
            hidden_dims=hidden_dims,
            activation=torch.nn.Tanh()
        ).to(device)
        
        # Create gradient layer
        self.grad_layer = GradientLayer(self.network).to(device)
        
        # Initialize optimizer
        self.optimizer = None
        
        # Create dataset
        self.data = self.create_training_data()
    
    def create_training_data(self, n_points=1000):
        """Create training data for the traveling wave problem"""
        # Domain bounds
        x_min, x_max = -1.0, 1.0
        t_min, t_max = 0.0, 1.0
        
        # Random points in the domain
        x = np.random.uniform(x_min, x_max, n_points)
        t = np.random.uniform(t_min, t_max, n_points)
        points = np.stack([x, t], axis=-1)
        
        # Initial condition points (t=0)
        x_ic = np.random.uniform(x_min, x_max, n_points//4)
        t_ic = np.zeros_like(x_ic)
        ic_points = np.stack([x_ic, t_ic], axis=-1)
        
        # Combine all points
        inputs = np.concatenate([points, ic_points], axis=0)
        
        return NIFDataset(
            inputs=inputs,
            outputs=np.zeros((len(inputs), 1)),  # Dummy outputs, not used
            device=self.device
        )
    
    def compute_loss(self, batch):
        """Compute PDE residual and initial condition losses"""
        inputs = batch[0]
        
        # Split into domain and IC points
        domain_pts = inputs[:-len(inputs)//5]
        ic_pts = inputs[-len(inputs)//5:]
        
        # PDE residual: u_t + u_x = 0
        u_t = self.grad_layer(domain_pts)[..., 1]  # du/dt
        u_x = self.grad_layer(domain_pts)[..., 0]  # du/dx
        residual = u_t + u_x
        pde_loss = torch.mean(residual**2)
        
        # Initial condition: u(x,0) = sin(2πx)
        u_ic = self.network(ic_pts)
        ic_targets = torch.sin(2 * np.pi * ic_pts[:, 0:1])
        ic_loss = torch.mean((u_ic - ic_targets)**2)
        
        return pde_loss + ic_loss
    
    def train(self, dataset, n_epochs=1000):
        """Train the network"""
        self.optimizer = LBFGS.create(self.network.parameters())
        
        def closure():
            self.optimizer.zero_grad()
            loss = self.compute_loss(next(iter(dataset.get_loader(dataset))))
            loss.backward()
            return loss
        
        for epoch in range(n_epochs):
            loss = LBFGS.step(self.optimizer, closure)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    def predict(self, x, t):
        """Predict solution at given points"""
        inputs = torch.tensor(
            np.stack([x, t], axis=-1),
            dtype=torch.float32,
            device=self.device
        )
        return self.network(inputs).detach().cpu().numpy()

def plot_solution(model, t=0.5):
    """Plot the solution at a specific time"""
    import matplotlib.pyplot as plt
    
    x = np.linspace(-1, 1, 100)
    u = model.predict(x, np.full_like(x, t))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, u, label=f't = {t}')
    plt.plot(x, np.sin(2*np.pi*(x - t)), '--', label='Exact')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Traveling Wave Solution')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Create and train model
    model = TravelingWave()
    dataset = model.create_training_data()
    model.train(dataset)
    
    # Plot results
    import matplotlib.pyplot as plt
    
    x = np.linspace(-1, 1, 100)
    t = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x, t)
    
    # Predict solution
    u = model.predict(X.flatten(), T.flatten()).reshape(X.shape)
    
    # Plot solution
    plt.figure(figsize=(10, 8))
    plt.contourf(X, T, u)
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Traveling Wave Solution')
    plt.show()

if __name__ == '__main__':
    main() 