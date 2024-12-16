import torch
import numpy as np
from ..layers import MLP, GradientLayer
from ..data import NIFDataset
from ..optimizers import LBFGS

class CylinderFlow:
    def __init__(
        self,
        hidden_dims=[64, 64, 64, 64],
        Re=100,  # Reynolds number
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.Re = Re
        
        # Create network for stream function
        self.network = MLP(
            input_dim=3,      # (x, y, t)
            output_dim=1,     # stream function ψ
            hidden_dims=hidden_dims,
            activation=torch.nn.Tanh(),
            initialization='he_uniform'
        ).to(device)
        
        # Create gradient layer for computing derivatives
        self.grad_layer = GradientLayer(self.network).to(device)
        
        # Initialize optimizer
        self.optimizer = None
        
        # Domain parameters
        self.x_min, self.x_max = -5.0, 15.0
        self.y_min, self.y_max = -5.0, 5.0
        self.t_min, self.t_max = 0.0, 10.0
        self.cylinder_radius = 0.5
        
    def is_inside_cylinder(self, x, y):
        """Check if points are inside the cylinder"""
        return x**2 + y**2 <= self.cylinder_radius**2
    
    def create_training_data(self, n_points=5000):
        """Create training data points"""
        # Interior points
        x = np.random.uniform(self.x_min, self.x_max, n_points)
        y = np.random.uniform(self.y_min, self.y_max, n_points)
        t = np.random.uniform(self.t_min, self.t_max, n_points)
        
        # Filter out points inside cylinder
        mask = ~self.is_inside_cylinder(x, y)
        points = np.stack([x[mask], y[mask], t[mask]], axis=-1)
        
        # Cylinder boundary points
        theta = np.random.uniform(0, 2*np.pi, n_points//4)
        x_cyl = self.cylinder_radius * np.cos(theta)
        y_cyl = self.cylinder_radius * np.sin(theta)
        t_cyl = np.random.uniform(self.t_min, self.t_max, n_points//4)
        cylinder_points = np.stack([x_cyl, y_cyl, t_cyl], axis=-1)
        
        # Inlet, outlet, and wall points
        x_in = np.full(n_points//4, self.x_min)
        y_in = np.random.uniform(self.y_min, self.y_max, n_points//4)
        t_in = np.random.uniform(self.t_min, self.t_max, n_points//4)
        inlet_points = np.stack([x_in, y_in, t_in], axis=-1)
        
        # Combine all points
        inputs = np.concatenate([points, cylinder_points, inlet_points], axis=0)
        
        return NIFDataset(
            inputs=inputs,
            outputs=np.zeros((len(inputs), 1)),  # Dummy outputs
            device=self.device
        )
    
    def compute_velocity(self, points):
        """Compute velocity components from stream function"""
        psi = self.network(points)
        grad_psi = self.grad_layer(points)
        
        u = grad_psi[..., 1]  # ∂ψ/∂y
        v = -grad_psi[..., 0]  # -∂ψ/∂x
        
        return u, v
    
    def compute_vorticity(self, points):
        """Compute vorticity from velocity components"""
        u, v = self.compute_velocity(points)
        
        # Compute velocity gradients
        u_grad = self.grad_layer(points, u.unsqueeze(-1))
        v_grad = self.grad_layer(points, v.unsqueeze(-1))
        
        # Vorticity = ∂v/∂x - ∂u/∂y
        return v_grad[..., 0] - u_grad[..., 1]
    
    def compute_loss(self, batch):
        """Compute total loss including NS equations and boundary conditions"""
        points = batch[0]
        
        # Split points into different regions
        interior_pts = points[:-self.n_points//2]
        cylinder_pts = points[-self.n_points//2:-self.n_points//4]
        inlet_pts = points[-self.n_points//4:]
        
        # Compute velocities and vorticity
        u, v = self.compute_velocity(interior_pts)
        omega = self.compute_vorticity(interior_pts)
        
        # Navier-Stokes equations in vorticity form
        omega_t = self.grad_layer(interior_pts, omega.unsqueeze(-1))[..., 2]
        omega_x = self.grad_layer(interior_pts, omega.unsqueeze(-1))[..., 0]
        omega_y = self.grad_layer(interior_pts, omega.unsqueeze(-1))[..., 1]
        omega_xx = self.grad_layer(interior_pts, omega_x.unsqueeze(-1))[..., 0]
        omega_yy = self.grad_layer(interior_pts, omega_y.unsqueeze(-1))[..., 1]
        
        # Vorticity transport equation
        ns_residual = (omega_t + u*omega_x + v*omega_y - 
                      (1/self.Re)*(omega_xx + omega_yy))
        ns_loss = torch.mean(ns_residual**2)
        
        # No-slip boundary condition on cylinder
        u_cyl, v_cyl = self.compute_velocity(cylinder_pts)
        bc_loss = torch.mean(u_cyl**2 + v_cyl**2)
        
        # Inlet boundary condition (uniform flow)
        u_in, v_in = self.compute_velocity(inlet_pts)
        inlet_loss = torch.mean((u_in - 1.0)**2 + v_in**2)
        
        # Total loss with weights
        total_loss = ns_loss + 10.0*bc_loss + 10.0*inlet_loss
        return total_loss
    
    def train(self, dataset, n_epochs=5000):
        """Train the network"""
        self.optimizer = LBFGS.create(
            self.network.parameters(),
            max_iter=20,
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
    
    def predict_flow_field(self, x, y, t):
        """Predict velocity field at given points"""
        points = torch.tensor(
            np.stack([x, y, np.full_like(x, t)], axis=-1),
            dtype=torch.float32,
            device=self.device
        )
        u, v = self.compute_velocity(points)
        return u.detach().cpu().numpy(), v.detach().cpu().numpy()

def main():
    # Create and train model
    model = CylinderFlow()
    dataset = model.create_training_data()
    model.train(dataset)
    
    # Plot results
    import matplotlib.pyplot as plt
    
    # Create grid for visualization
    x = np.linspace(-5, 15, 200)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Predict flow field at t=5.0
    u, v = model.predict_flow_field(X.flatten(), Y.flatten(), 5.0)
    speed = np.sqrt(u**2 + v**2).reshape(X.shape)
    
    # Plot flow field
    plt.figure(figsize=(15, 5))
    
    # Speed contour
    plt.contourf(X, Y, speed, levels=50)
    plt.colorbar(label='Flow Speed')
    
    # Streamlines
    plt.streamplot(X, Y, u.reshape(X.shape), v.reshape(X.shape), 
                  density=2, color='white', linewidth=0.5)
    
    # Draw cylinder
    circle = plt.Circle((0, 0), 0.5, color='black')
    plt.gca().add_patch(circle)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cylinder Flow at t=5.0')
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main() 