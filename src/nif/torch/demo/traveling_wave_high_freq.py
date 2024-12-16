import torch
import numpy as np
from ..layers import MLP, GradientLayer
from ..data import NIFDataset
from ..optimizers import LBFGS

class TravelingWaveHighFreq:
    def __init__(self, hidden_dims=[64, 64, 64], device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Create network with wider and deeper architecture for high-frequency solutions
        self.network = MLP(
            input_dim=2,  # (x, t)
            output_dim=1,  # u(x,t)
            hidden_dims=hidden_dims,
            activation=torch.nn.Tanh(),
            initialization='he_uniform'  # Better for deeper networks
        ).to(device)
        
        # Create gradient layer
        self.grad_layer = GradientLayer(self.network).to(device)
        
        # Initialize optimizer
        self.optimizer = None
        
        # High frequency parameter
        self.freq = 8.0  # Higher frequency for initial condition
    
    def create_training_data(self, n_points=2000):
        """Create training data with more points for high-frequency problem"""
        # Domain bounds
        x_min, x_max = -1.0, 1.0
        t_min, t_max = 0.0, 1.0
        
        # Random points in the domain (using more points for higher frequency)
        x = np.random.uniform(x_min, x_max, n_points)
        t = np.random.uniform(t_min, t_max, n_points)
        points = np.stack([x, t], axis=-1)
        
        # Initial condition points (t=0)
        x_ic = np.random.uniform(x_min, x_max, n_points//2)  # More IC points
        t_ic = np.zeros_like(x_ic)
        ic_points = np.stack([x_ic, t_ic], axis=-1)
        
        # Boundary points
        x_bc = np.array([-1.0, 1.0]).repeat(n_points//4)
        t_bc = np.random.uniform(t_min, t_max, n_points//2)
        bc_points = np.stack([x_bc, t_bc], axis=-1)
        
        # Combine all points
        inputs = np.concatenate([points, ic_points, bc_points], axis=0)
        
        return NIFDataset(
            inputs=inputs,
            outputs=np.zeros((len(inputs), 1)),  # Dummy outputs, not used
            device=self.device
        )
    
    def compute_loss(self, batch):
        """Compute PDE residual, initial condition, and boundary condition losses"""
        inputs = batch[0]
        
        # Split points
        n_domain = len(inputs) - len(inputs)//2
        domain_pts = inputs[:n_domain]
        ic_pts = inputs[n_domain:-len(inputs)//4]
        bc_pts = inputs[-len(inputs)//4:]
        
        # PDE residual: u_t + u_x = 0
        u_t = self.grad_layer(domain_pts)[..., 1]  # du/dt
        u_x = self.grad_layer(domain_pts)[..., 0]  # du/dx
        residual = u_t + u_x
        pde_loss = torch.mean(residual**2)
        
        # Initial condition: u(x,0) = sin(2π*freq*x)
        u_ic = self.network(ic_pts)
        ic_targets = torch.sin(2 * np.pi * self.freq * ic_pts[:, 0:1])
        ic_loss = torch.mean((u_ic - ic_targets)**2)
        
        # Periodic boundary conditions
        u_left = self.network(bc_pts[::2])
        u_right = self.network(bc_pts[1::2])
        bc_loss = torch.mean((u_left - u_right)**2)
        
        # Combine losses with weights
        total_loss = pde_loss + 10.0 * ic_loss + bc_loss
        return total_loss
    
    def train(self, dataset, n_epochs=2000):
        """Train the network with more epochs for high-frequency case"""
        self.optimizer = LBFGS.create(
            self.network.parameters(),
            max_iter=25,  # More iterations per step
            history_size=50
        )
        
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

def main():
    # Create and train model
    model = TravelingWaveHighFreq()
    dataset = model.create_training_data()
    model.train(dataset)
    
    # Plot results
    import matplotlib.pyplot as plt
    
    # Create high-resolution grid for better visualization
    x = np.linspace(-1, 1, 200)
    t = np.linspace(0, 1, 200)
    X, T = np.meshgrid(x, t)
    
    # Predict solution
    u = model.predict(X.flatten(), T.flatten()).reshape(X.shape)
    
    # Plot solution
    plt.figure(figsize=(12, 10))
    
    # Surface plot
    plt.subplot(211)
    plt.contourf(X, T, u, levels=50)
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('High-Frequency Traveling Wave Solution')
    
    # Plot solution at different times
    plt.subplot(212)
    t_samples = [0.0, 0.25, 0.5, 0.75, 1.0]
    for t_val in t_samples:
        t_idx = int(t_val * (len(t) - 1))
        plt.plot(x, u[t_idx], label=f't = {t_val}')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Solution at Different Times')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main() 