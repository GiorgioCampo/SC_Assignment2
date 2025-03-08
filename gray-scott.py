import numpy as np
import matplotlib.pyplot as plt
from src.plots import plot_grid, plot_comparison

def initialize_grid(n):
    """Initialize U and V fields"""
    u = np.full((n+2, n+2), 0.5)  # U = 0.5
    v = np.zeros((n+2, n+2))      # V = 0

    # Place a square of V = 0.25 in the center
    center = n // 2
    seed_size = n // 10
    v[center - seed_size:center + seed_size, center - seed_size:center + seed_size] = 0.25

    # Add small random noise to both U and V
    u += np.random.uniform(-0.01, 0.01, (n+2, n+2))
    v += np.random.uniform(-0.01, 0.01, (n+2, n+2))

    return u, v

def apply_periodic_bc(u):
    u[0, :] = u[-2, :]
    u[-1, :] = u[1, :]
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]

def compute_laplacian(z):
    return (z[:-2, 1:-1] + z[1:-1, :-2] - 4 * z[1:-1, 1:-1] +
            z[1:-1, 2:] + z[2:, 1:-1])

def gray_scott_simulation(size=100, Du=0.16, Dv=0.08, f=0.035, k=0.060, steps=5000, dt=1):
    """Simulate the Gray-Scott model."""
    
    U, V = initialize_grid(size)  # Initialize U and V fields

    for _ in range(steps):
        # Compute Laplacians
        Lu = compute_laplacian(U)
        Lv = compute_laplacian(V)

        # Compute reaction terms
        uvv = U[1:-1, 1:-1] * V[1:-1, 1:-1]**2  # Represents U * V^2
        
        # Update U and V using explicit finite differences
        U[1:-1, 1:-1] += Du * Lu - uvv + f * (1 - U[1:-1, 1:-1])
        V[1:-1, 1:-1] += Dv * Lv + uvv - (f + k) * V[1:-1, 1:-1]

        # Apply periodic boundary conditions
        apply_periodic_bc(U)
        apply_periodic_bc(V)

    plot_grid((U - V), title = f"Gray-Scott Model Simulation after {steps} steps", savefig=False, 
              filename=f"images/gs/gs_{Du}_{Dv}_{f}_{k}.pdf", cmap='magma', colorbar=True)

    return U, V

if __name__ == "__main__":
    # Run the simulation
    size = 100 
    steps = 5000  

    # Mixed dots and waves (default)
    U1, V1 = gray_scott_simulation(size=100, Du=0.16, Dv=0.08, f=0.035, k=0.060, steps=5000, dt=1)

    # Sparse spots
    U2, V2 = gray_scott_simulation(size=100, Du=0.16, Dv=0.08, f=0.022, k=0.051, steps=5000, dt=1)

    # Dense labyrinth-like waves
    U3, V3 = gray_scott_simulation(size=100, Du=0.16, Dv=0.08, f=0.060, k=0.062, steps=5000, dt=1)

    # Very fine, complex structures
    U4, V4 = gray_scott_simulation(size=100, Du=0.16, Dv=0.08, f=0.025, k=0.055, steps=5000, dt=1)

    results = [(U1 - V1), (U2 - V2), (U3 - V3), (U4 - V4)]

    plot_comparison(results, title="Gray-Scott Model Simulation", 
                    sub_titles=["Mixed dots and waves", "Sparse spots", "Dense labyrinth-like waves", "Very fine, complex structures"], 
                    savefig=False, filename="images/gs/gs_comparison.pdf", cmap='magma', colorbar=True)

    plt.show()
    
