import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm

from plots import plot_grid, plot_comparison
from utilities import *

class DLASimulation:
    def __init__(self, size=100, eta=1.0, omega=1.5, max_iterations=1000, tolerance=1e-5, domain=None, parallel=False):
        self.size = size
        self.eta = eta
        self.omega = omega
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.domain = np.zeros((size, size), dtype=np.int32)
        self.parallel = parallel

        # Metrics
        self.total_sor_iterations = 0
        self.sor_calls = 0

        # Initialize concentration field
        self.concentration = np.zeros((size, size), dtype=np.float64)

        # Set initial seed at the bottom center
        if domain is not None:
            self.domain = domain
        else:
            self.domain[0, size//2] = 1

        # Initialize with linear gradient
        self.initialize_concentration()

    def initialize_concentration(self):

        for row in range(self.size):
            self.concentration[row, :] = row / (self.size - 1)

        # Set boundary conditions directly
        # Top boundary (fixed)
        self.concentration[self.size - 1, :] = 1.0

        # Object sites (fixed)
        for row in range(self.size):
            for col in range(self.size):
                if self.domain[row, col] == 1:
                    self.concentration[row, col] = 0.0

        # Update concentration based on boundary conditions (seed)
        # self.solve_laplace()

    def solve_laplace(self):

        # Top boundary (fixed)
        self.concentration[self.size - 1, :] = 1.0

        # Object sites (fixed)
        for row in range(self.size):
            for col in range(self.size):
                if self.domain[row, col] == 1:
                    self.concentration[row, col] = 0.0

        if self.parallel:
            iter_count = sor_iteration_parallel(
                self.concentration,
                self.domain,
                self.omega,
                self.tolerance,
                self.max_iterations
            )
        else:
            iter_count = sor_iteration(
                self.concentration,
                self.domain,
                self.omega,
                self.tolerance,
                self.max_iterations
            )

        # Update metrics
        self.total_sor_iterations += iter_count
        self.sor_calls += 1

        return iter_count

    def find_growth_candidates(self):

        candidates_i, candidates_j = find_growth_candidates_numba(
            self.domain, self.size)
        return list(zip(candidates_i, candidates_j))

    def calculate_growth_probabilities(self, candidates):

        if not candidates:
            return [], []

        # Convert to separate arrays for numba
        candidates_i = np.array([i for i, j in candidates], dtype=np.int32)
        candidates_j = np.array([j for i, j in candidates], dtype=np.int32)

        # Calculate probabilities using numba function
        probs = calculate_growth_probabilities_numba(
            candidates_i,
            candidates_j,
            self.concentration,
            self.eta
        )

        # Safety check: ensure all probabilities are non-negative
        probs = np.maximum(0.0, probs)

        # Re-normalize if needed
        sum_probs = np.sum(probs)
        if sum_probs <= 0:
            print("Warning: All probabilities are zero, falling back to equal probabilities")
            probs = np.ones_like(probs) / len(probs)

        return candidates, probs

    def grow_step(self):

        # Find growth candidates
        candidates = self.find_growth_candidates()
        np.random.shuffle(candidates)


        # If no candidates, return False (growth complete)
        if not candidates:
            return False

        # Calculate growth probabilities
        candidates, probs = self.calculate_growth_probabilities(candidates)

        # Choose a growth site based on probabilities
        if len(candidates) > 0:
            try:
                idx = np.random.choice(len(candidates), p=probs)
                row, col = candidates[idx]

                # Add to the object
                self.domain[row, col] = 1
                self.concentration[row, col] = 0.0  

                # Solve Laplace equation again with new boundary
                self.solve_laplace()

                return True
            except ValueError as e:
                print(f"Warning: Error in probability calculation: {e}")
                print(f"Probabilities: min={np.min(probs)}, max={np.max(probs)}, sum={np.sum(probs)}")
                # Fall back to equal probabilities
                idx = np.random.choice(len(candidates))
                row, col = candidates[idx]

                # Add to the object
                self.domain[row, col] = 1
                self.concentration[row, col] = 0.0

                # Solve Laplace equation again
                self.solve_laplace()

                return True

        return False
    
    def run_simulation(self, steps=1000):
        for step in tqdm(range(steps), desc="Growth Progress for η=%.2f" % self.eta):
            success = self.grow_step()
            if not success:
                print(f"Growth stopped after {step} steps")
                break

# Function to calculate optimal omega for a grid
def calculate_optimal_omega(grid_size):
    return 2 / (1 + np.sin(np.pi/(grid_size + 1)))

# Function to run multiple simulations with different eta values
def run_dla_experiments(etas=[0.5, 1.0, 2.0], size=100, steps=1000, parallel=False, savefig=False):
    results = []
    optimal_omega = calculate_optimal_omega(size)

    for eta in etas:
        sim = DLASimulation(size=size, eta=eta, omega=optimal_omega, parallel=parallel)
        sim.run_simulation(steps=steps)
        results.append(sim.domain)

        plot_grid(sim.domain, title=f"DLA Structure with η = {eta} in {steps} steps", savefig=False, filename=f"images/dla/dla_eta_{eta}.pdf", 
                  cmap=colors.ListedColormap(['white', 'black']))
    
    plot_comparison(results, title="DLA Structure for Different η Values", sub_titles=[f"η = {eta}" for eta in etas], savefig=savefig, 
                    filename="images/dla/dla_comparison.pdf", cmap=colors.ListedColormap(['white', 'black']))
    return results

# Benchmark different omega values for SOR convergence
def find_optimal_omega(size=100, num_omegas=5, steps=50, etas=[0.5, 1.0, 2.0], savefig=False):  
    omegas = np.linspace(1.5, 1.99, num_omegas)
    optimal_omega = calculate_optimal_omega(size)
    
    print("\nBenchmarking different omega values using DLASimulation:")
    print("{:<10} {:<10} {:<10} {:<10}".format("Omega", "Eta", "Avg Iterations", "Std Dev"))
    print("-" * 40)
    
    results = {eta: [] for eta in etas}
    domain = np.zeros((size, size), dtype=np.int32)

    # Start from 
    center = size // 2
    radius = size // 4
    for row in range(size):
        for col in range(size):
            if (row - center)**2 + (col - center)**2 < radius**2:
                domain[row, col] = 1

    
    for eta in etas:
        for omega in omegas:
            sim = DLASimulation(size=size, eta=eta, omega=omega, domain=domain)
            
            iter_counts = []
            for _ in range(steps):
                sim.grow_step()
                iter_counts.append(sim.total_sor_iterations / max(1, sim.sor_calls))
                
            avg_iter = np.mean(iter_counts)
            std_iter = np.std(iter_counts)
            results[eta].append((omega, avg_iter))
            
            print("{:<10.4f} {:<10.2f} {:<10.2f} {:<10.2f}".format(omega, eta, avg_iter, std_iter))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for eta in etas:
        omegas, iterations = zip(*results[eta])
        best_omega = omegas[np.argmin(iterations)]
        p = plt.plot(omegas, iterations, 'o-', label=f'η={eta}')
        plt.axvline(x=best_omega, linestyle='--', label=f'Optimal for η={eta}: {best_omega:.4f}', color=p[0].get_color())
    
    plt.axvline(x=optimal_omega, color='r', linestyle='--', label=f'Theoretical optimal: {optimal_omega:.4f}')
    plt.xlabel('Omega')
    plt.ylabel('Avg Iterations to converge')
    plt.title('SOR Convergence Rate vs. Omega for Different η Values')
    plt.grid(True)
    plt.legend()
    if savefig:
        plt.savefig("images/dla/sor_benchmark_eta.pdf")
    
    return results

# Run the simulation with optimized parameters
if __name__ == "__main__":
    # Increase font size
    plt.rcParams.update({'font.size': 14})

    # Find optimal omega for SOR
    best_omega = find_optimal_omega(size=100, num_omegas=30, etas=[0, 0.5, 1.0, 1.5])

    # Test different eta values with the optimal omega
    print("\nRunning full DLA simulations with optimal omega...")
    results = run_dla_experiments(etas=[0, 0.5, 1.0, 1.5], size=100, steps=500)

    # Measure fractal dimension
    print("\nMeasuring fractal dimension...")
    for sim in results:
         pass