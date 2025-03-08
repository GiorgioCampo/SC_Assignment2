import numpy as np
import matplotlib.pyplot as plt

def initialize_grid(n):
    """Initialize U and V fields"""
    u = np.full((n+2, n+2), 0.5)  # U is 0.5 everywhere
    v = np.zeros((n+2, n+2))      # V is 0 everywhere initially

    # Place a square of V = 0.25 in the center
    center = n // 2
    seed_size = n // 10
    v[center - seed_size:center + seed_size, center - seed_size:center + seed_size] = 0.25

    # Add small random noise to both U and V
    u += np.random.uniform(-0.01, 0.01, (n+2, n+2))
    v += np.random.uniform(-0.01, 0.01, (n+2, n+2))

    return u, v

def apply_periodic_bc(u):
    """Apply periodic boundary conditions to ensure seamless pattern formation."""
    u[0, :] = u[-2, :]
    u[-1, :] = u[1, :]
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]

def compute_laplacian(z):
    """Compute the discrete Laplacian using a finite difference stencil."""
    return (z[:-2, 1:-1] + z[1:-1, :-2] - 4 * z[1:-1, 1:-1] +
            z[1:-1, 2:] + z[2:, 1:-1])

def gray_scott_simulation(n=100, Du=0.16, Dv=0.08, f=0.035, k=0.060, steps=5000, dt=1):
    """Simulate the Gray-Scott model."""
    U, V = initialize_grid(n)  # Initialize U and V fields

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

    return U, V

def plot_concentration(U, V):
    """Plot the final concentration field combining U and V."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    combined = U - V  # Highlight the difference between U and V
    img = ax.imshow(combined, cmap='RdBu', origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(img, label="U - V Concentration Difference")
    
    ax.set_title(f"Gray-Scott Model Simulation after {steps} steps")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()

# Run the simulation
n = 100  # Grid size
steps = 5000  # Simulation steps

if __name__ == "__main__":

    # Mixed dots and waves (default)
    U1, V1 = gray_scott_simulation(n=100, Du=0.16, Dv=0.08, f=0.035, k=0.060, steps=5000, dt=1)
    plot_concentration(U1[1:-1, 1:-1], V1[1:-1, 1:-1])

    # Sparse spots
    U2, V2 = gray_scott_simulation(n=100, Du=0.16, Dv=0.08, f=0.022, k=0.051, steps=5000, dt=1)
    plot_concentration(U2[1:-1, 1:-1], V2[1:-1, 1:-1])

    # Dense labyrinth-like waves
    U3, V3 = gray_scott_simulation(n=100, Du=0.16, Dv=0.08, f=0.060, k=0.062, steps=5000, dt=1)
    plot_concentration(U3[1:-1, 1:-1], V3[1:-1, 1:-1])

    # Large, high-contrast patterns
    U4, V4 = gray_scott_simulation(n=100, Du=0.16, Dv=0.08, f=0.050, k=0.065, steps=5000, dt=1)
    plot_concentration(U4[1:-1, 1:-1], V4[1:-1, 1:-1])

    # Very fine, complex structures
    U5, V5 = gray_scott_simulation(n=100, Du=0.16, Dv=0.08, f=0.025, k=0.055, steps=5000, dt=1)
    plot_concentration(U5[1:-1, 1:-1], V5[1:-1, 1:-1])
