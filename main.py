import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
from matplotlib import colors

class DLA:
    def __init__(self, size=100, eta=1.0, omega=1.8, max_iterations=1000, tolerance=1e-4):
        """
        Initialize the DLA model.
        
        Parameters:
        -----------
        size : int
            Size of the square grid
        eta : float
            Parameter that determines the shape of the cluster
            - eta = 1: normal DLA cluster
            - eta < 1: more compact (Eden cluster at eta = 0)
            - eta > 1: more open (like lightning flash)
        omega : float
            Relaxation parameter for SOR iteration
        max_iterations : int
            Maximum number of iterations for SOR solver
        tolerance : float
            Convergence tolerance for SOR solver
        """
        self.size = size
        self.eta = eta
        self.omega = omega
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Initialize the grid (0: empty, 1: occupied)
        self.grid = np.zeros((size, size), dtype=int)
        
        # Place seed at the bottom center
        self.grid[size-1, size//2] = 1
        
        # Initialize concentration field with linear gradient
        self.concentration = np.zeros((size, size))
        self.initialize_concentration()
        
        # Keep track of growth candidates
        self.growth_candidates = set()
        self.update_growth_candidates()
    
    def initialize_concentration(self):
        """Initialize concentration with linear gradient from top (1.0) to bottom (0.0)"""
        for i in range(self.size):
            self.concentration[i, :] = 1.0 - i / (self.size - 1)
        
        # Set concentration to zero where the cluster is
        self.concentration[self.grid == 1] = 0.0
    
    def update_growth_candidates(self):
        """Update the set of growth candidate sites"""
        self.growth_candidates.clear()
        
        # Scan the entire grid for occupied cells
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 1:  # If cell is occupied
                    # Check neighbors (north, east, south, west)
                    neighbors = [
                        (i-1, j), (i, j+1), (i+1, j), (i, j-1)
                    ]
                    
                    for ni, nj in neighbors:
                        # Check if neighbor is within grid and empty
                        if (0 <= ni < self.size and 0 <= nj < self.size and 
                            self.grid[ni, nj] == 0):
                            self.growth_candidates.add((ni, nj))


    def solve_laplace(self):
        """Solve the Laplace equation using SOR method with current boundary conditions"""
        # Copy the current concentration for update
        new_concentration = self.concentration.copy()
        
        # Apply boundary conditions
        # Top boundary (constant concentration = 1.0)
        new_concentration[0, :] = 1.0
        
        # Bottom boundary (constant concentration = 0.0)
        new_concentration[self.size-1, :] = 0.0
        
        # Set concentration to zero at cluster sites
        new_concentration[self.grid == 1] = 0.0
        
        # SOR iteration
        error = float('inf')
        iterations = 0
        
        while error > self.tolerance and iterations < self.max_iterations:
            old_concentration = new_concentration.copy()
            
            core_sor(new_concentration, self.omega)
            
            # Calculate error
            error = np.max(np.abs(new_concentration - old_concentration))
            iterations += 1
        
        self.concentration = new_concentration
        return iterations
    
    def calculate_growth_probabilities(self):
        """Calculate growth probabilities for all growth candidates"""
        probabilities = {}
        sum_prob = 0.0
        
        for i, j in self.growth_candidates:
            prob = self.concentration[i, j] ** self.eta
            probabilities[(i, j)] = prob
            sum_prob += prob
        
        # Normalize probabilities
        if sum_prob > 0:
            for pos in probabilities:
                probabilities[pos] /= sum_prob
        
        return probabilities
    
    def grow_cluster(self):
        """Perform one growth step"""
        # Solve Laplace equation
        iterations = self.solve_laplace()
        
        # Calculate growth probabilities
        probabilities = self.calculate_growth_probabilities()
        
        # Choose a growth candidate based on probabilities
        if not probabilities:
            return False
        
        positions = list(probabilities.keys())
        probs = list(probabilities.values())
        
        # Use numpy.random.choice to select a growth site
        chosen_index = np.random.choice(len(positions), p=probs)
        chosen_position = positions[chosen_index]
        
        # Add the chosen position to the cluster
        i, j = chosen_position
        self.grid[i, j] = 1
        self.concentration[i, j] = 0.0
        
        # Update growth candidates
        self.growth_candidates.remove(chosen_position)
        
        # Check neighbors of the new cluster site for new growth candidates
        neighbors = [
            (i-1, j), (i, j+1), (i+1, j), (i, j-1)
        ]
        
        for ni, nj in neighbors:
            if (0 <= ni < self.size and 0 <= nj < self.size and 
                self.grid[ni, nj] == 0 and
                (ni, nj) not in self.growth_candidates):
                
                # Check if neighbor has a cluster neighbor
                has_cluster_neighbor = False
                neighbor_positions = [
                    (ni-1, nj), (ni, nj+1), (ni+1, nj), (ni, nj-1)
                ]
                
                for nni, nnj in neighbor_positions:
                    if (0 <= nni < self.size and 0 <= nnj < self.size and
                        self.grid[nni, nnj] == 1):
                        has_cluster_neighbor = True
                        break
                
                if has_cluster_neighbor:
                    self.growth_candidates.add((ni, nj))
        
        return True
    
    def simulate(self, steps=1000):
        """Run the DLA simulation for a given number of growth steps"""
        start_time = time.time()
        
        for step in tqdm(range(steps)):
            success = self.grow_cluster()
            if not success:
                print(f"No more growth possible after {step} steps")
                break
            
            # if (step + 1) % 100 == 0:
            #     elapsed = time.time() - start_time
            #     print(f"Step {step + 1} completed. Time elapsed: {elapsed:.2f} seconds")
        
        total_time = time.time() - start_time
        print(f"Simulation completed in {total_time:.2f} seconds")
    
    def plot_cluster(self, title=None):
        """Plot the current state of the DLA cluster"""
        plt.figure(figsize=(8, 8))
        
        # Create a colormap: white for empty, black for cluster
        cmap = colors.ListedColormap(['white', 'black'])
        bounds = [0, 0.5, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        plt.imshow(self.grid, cmap=cmap, norm=norm, origin='upper')
        
        if title:
            plt.title(title)
        else:
            plt.title(f"DLA Cluster (η={self.eta}, Size={self.size}x{self.size})")
        
        plt.colorbar(ticks=[0, 1], label="0: Empty, 1: Cluster")
        plt.tight_layout()
        plt.show()
    
    def plot_concentration(self):
        """Plot the current concentration field"""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.concentration, cmap='viridis', origin='upper')
        plt.colorbar(label="Concentration")
        plt.title(f"Concentration Field (η={self.eta}, Size={self.size}x{self.size})")
        plt.tight_layout()
        plt.show()


@jit(nopython=True)
def core_sor(c, omega):
    """Core SOR iteration calculations with mask handling"""
    # Interior points
    for i in range(1, c.shape[0]-1):
        for j in range(1, c.shape[1]-1):
                
            c[i,j] = (1 - omega) * c[i,j] + 0.25 * omega * (
                c[i+1,j] + c[i-1,j] + c[i,j+1] + c[i,j-1]
            )
    
    # Periodic boundary condition along y-axis
    for i in range(1, c.shape[0]-1):
        c[i,0] = (1 - omega) * c[i,0] + 0.25 * omega * (
            c[i+1,0] + c[i-1,0] + c[i,1] + c[i,-1]
        )
        c[i,-1] = (1 - omega) * c[i,-1] + 0.25 * omega * (
            c[i+1,-1] + c[i-1,-1] + c[i,0] + c[i,-2]
        )
    
    # Apply initial conditions as minimum values
    # for i in range(c.shape[0]):
    #     for j in range(c.shape[1]):
    #         if c_init[i,j] > c[i,j]:
    #             c[i,j] = c_init[i,j]


# Run simulations with different values of η
def run_experiment():
    # Values of eta to test
    eta_values = [0.2, 1.0, 3.0]
    
    # Set up figure for comparison
    fig, axes = plt.subplots(1, len(eta_values), figsize=(15, 5))
    
    for i, eta in enumerate(eta_values):
        
        # Create DLA model with appropriate parameters
        dla = DLA(size=100, eta=eta, omega=1.5, max_iterations=1000, tolerance=1e-4)
        
        # Run simulation
        dla.simulate(steps=2000)
        
        # Plot result
        cmap = colors.ListedColormap(['white', 'black'])
        bounds = [0, 0.5, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        axes[i].imshow(dla.grid, cmap=cmap, norm=norm, origin='upper')
        axes[i].set_title(f"η = {eta}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("dla_comparison.png")
    plt.show()

if __name__ == "__main__":
    # Run the experiment
    run_experiment()
    
    # Alternatively, you can run a single simulation with:
    # dla = DLA(size=100, eta=1.0)
    # dla.simulate(steps=1000)
    # dla.plot_cluster()
    # dla.plot_concentration()