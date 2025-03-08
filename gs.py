import numpy as np
import matplotlib.pyplot as plt
from src.plots import plot_grid, plot_comparison

# Import functions from the diffusion equation code
from diffusion_equation import plot_diffusion, animate_diffusion

def initialize_grid(n):
    u = np.full((n+2, n+2), 0.5)  # U = 0.5
    v = np.zeros((n+2, n+2))       # V = 0
    
    # Place a square of V = 0.25 in the center
    center = n // 2
    seed_size = n // 10
    v[center - seed_size:center + seed_size, center - seed_size:center + seed_size] = 0.25
    
    # Add small random noise to both U and V
    u += np.random.uniform(-0.01, 0.01, (n+2, n+2))
    v += np.random.uniform(-0.01, 0.01, (n+2, n+2))
    
    return u, v

def apply_periodic_bc(u):
    u[0, :] = u[-2, :]   # Bottom boundary from second-to-last row
    u[-1, :] = u[1, :]   # Top boundary from second row
    u[:, 0] = u[:, -2]   # Left boundary from second-to-last column
    u[:, -1] = u[:, 1]   # Right boundary from second column

def compute_laplacian(z, dx):
    return (z[:-2, 1:-1] + z[1:-1, :-2] - 4 * z[1:-1, 1:-1] + 
            z[1:-1, 2:] + z[2:, 1:-1]) / (dx**2)

def gray_scott_simulation(size=100, Du=0.16, Dv=0.08, f=0.035, k=0.060, steps=5000, dt=1.0, dx=1.0, 
                         save_interval=100, animate=False):
    stability_condition_u = 4 * Du * dt / dx**2
    stability_condition_v = 4 * Dv * dt / dx**2
    
    if stability_condition_u > 1 or stability_condition_v > 1:
        print(f"Warning: Stability condition may be violated!")
        print(f"Du condition: {stability_condition_u:.4f} (should be <= 1)")
        print(f"Dv condition: {stability_condition_v:.4f} (should be <= 1)")
        print("Consider reducing dt or increasing dx.")
    
    # Initialize U and V fields
    U, V = initialize_grid(size)
    
    # Arrays to store results for animation or analysis
    times = [0] if save_interval > 0 else []
    solutions = [U - V] if save_interval > 0 else []
    
    # Main simulation loop
    for step in range(steps):
        # Compute Laplacians with proper scaling
        Lu = compute_laplacian(U, dx)
        Lv = compute_laplacian(V, dx)
        
        # Compute reaction terms
        uvv = U[1:-1, 1:-1] * V[1:-1, 1:-1]**2  # Represents U * V^2
        
        # Update U and V using explicit finite differences
        U[1:-1, 1:-1] += dt * (Du * Lu - uvv + f * (1 - U[1:-1, 1:-1]))
        V[1:-1, 1:-1] += dt * (Dv * Lv + uvv - (f + k) * V[1:-1, 1:-1])
        
        # Apply periodic boundary conditions
        apply_periodic_bc(U)
        apply_periodic_bc(V)
        
        # Save results at specified intervals
        if save_interval > 0 and (step + 1) % save_interval == 0:
            times.append((step + 1) * dt)
            solutions.append(U.copy() - V.copy())  # Store U-V pattern
    
    # Plot final state
    plot_grid((U - V), title=f"Gray-Scott Model Simulation after {steps} steps", 
              savefig=False, filename=f"images/gs/gs_{Du}_{Dv}_{f}_{k}.pdf", 
              cmap='magma', colorbar=True)
    
    # Create animation if requested
    if animate and save_interval > 0:
        animate_diffusion(np.array(times), np.array(solutions), 
                         interval=200, time_multiplier=1, 
                         save_animation=True, 
                         filename=f"images/gs/gs_animation_{Du}_{Dv}_{f}_{k}.gif")
    
    if save_interval > 0:
        return U, V, np.array(times), np.array(solutions)
    else:
        return U, V

if __name__ == "__main__":
    # Set simulation parameters
    size = 100
    steps = 5000
    dt = 0.25  # Reduced time step for better stability
    dx = 1.0
    save_interval = 50  # Save every 50 steps
    
    # Run simulations with different parameters
    
    # Mixed dots and waves (default)
    U1, V1 = gray_scott_simulation(size=size, Du=0.16, Dv=0.08, f=0.035, k=0.060, 
                                  steps=steps, dt=dt, dx=dx, save_interval=0)
    
    # Sparse spots
    U2, V2 = gray_scott_simulation(size=size, Du=0.16, Dv=0.08, f=0.022, k=0.051, 
                                  steps=steps, dt=dt, dx=dx, save_interval=0)
    
    # Dense labyrinth-like waves
    U3, V3 = gray_scott_simulation(size=size, Du=0.16, Dv=0.08, f=0.060, k=0.062, 
                                  steps=steps, dt=dt, dx=dx, save_interval=0)
    
    # Very fine, complex structures
    U4, V4, times, solutions = gray_scott_simulation(size=size, Du=0.16, Dv=0.08, f=0.025, k=0.055, 
                                                   steps=steps, dt=dt, dx=dx, save_interval=save_interval, 
                                                   animate=True)
    
    # Create comparison plot
    results = [(U1 - V1), (U2 - V2), (U3 - V3), (U4 - V4)]
    
    plot_comparison(results, title="Gray-Scott Model Simulation",
                   sub_titles=["Mixed dots and waves", "Sparse spots", 
                              "Dense labyrinth-like waves", "Very fine, complex structures"],
                   savefig=False, filename="images/gs/gs_comparison.pdf", 
                   cmap='magma', colorbar=True)
    
    plt.show()