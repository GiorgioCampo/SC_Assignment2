import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import numba
from numba import jit, prange
from tqdm import tqdm
from src.plots import plot_grid, plot_comparison

# Optimized SOR function with Numba JIT
@numba.jit(nopython=True)
def is_neighboring_cluster(grid, row, col):
    size = grid.shape[0]
    neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
    for nrow, ncol in neighbors:
        # Peropdic borders!!
        if ((0 <= nrow < size and 0 <= ncol < size and grid[nrow, ncol] == 1) or 
            (0 <= nrow < size and ncol == -1 and grid[nrow, size-1] == 1) or 
            (0 <= nrow < size and ncol == size and grid[nrow, 0] == 1)):
            return True
    return False

@jit(nopython=True, parallel=False)
def sor_iteration(concentration, domain, omega, tolerance, max_iterations):
    size = concentration.shape[0]

    for iter_count in range(max_iterations):
        max_diff = 0.0

        # Red-black ordering for better parallelization potential
        # First update "red" cells (i+j even)
        for i in range(1, size-1):
            for j in range(1, size-1):
                if (i + j) % 2 == 0 and domain[i, j] == 0:
                    old_val = concentration[i, j]
                    new_val = 0.25 * (
                        concentration[i+1, j] +
                        concentration[i-1, j] +
                        concentration[i, j+1] +
                        concentration[i, j-1]
                    )
                    concentration[i, j] = (1 - omega) * \
                        old_val + omega * new_val
                    diff = abs(concentration[i, j] - old_val)
                    if diff > max_diff:
                        max_diff = diff

        # Then update "black" cells (i+j odd)
        for i in range(1, size-1):
            for j in range(1, size-1):
                if (i + j) % 2 == 1 and domain[i, j] == 0:
                    old_val = concentration[i, j]
                    new_val = 0.25 * (
                        concentration[i+1, j] +
                        concentration[i-1, j] +
                        concentration[i, j+1] +
                        concentration[i, j-1]
                    )
                    concentration[i, j] = (1 - omega) * \
                        old_val + omega * new_val
                    diff = abs(concentration[i, j] - old_val)
                    if diff > max_diff:
                        max_diff = diff

        # Periodic boundary condition along y-axis
        for i in range(1, concentration.shape[0]-1):
            concentration[i,-1] = (1 - omega) * concentration[i,-1] + 0.25 * omega * (
                concentration[i+1,-1] + concentration[i-1,-1] + concentration[i,0] + concentration[i,-2]
            )


        # Check for convergence
        if max_diff < tolerance:
            return iter_count + 1

    return max_iterations

@jit(nopython=True, parallel=True)
def sor_iteration_parallel(concentration, domain, omega, tolerance, max_iterations):
    size = concentration.shape[0]

    for iter_count in range(max_iterations):
        max_diff = 0.0

        # Red phase (i + j even)
        for i in prange(1, size-1):
            for j in range(1, size-1):
                if (i + j) % 2 == 0 and domain[i, j] == 0:
                    old_val = concentration[i, j]
                    new_val = 0.25 * (
                        concentration[i+1, j] +
                        concentration[i-1, j] +
                        concentration[i, j+1] +
                        concentration[i, j-1]
                    )
                    concentration[i, j] = (1 - omega) * old_val + omega * new_val
                    diff = abs(concentration[i, j] - old_val)
                    if diff > max_diff:
                        max_diff = diff

        # Black phase (i + j odd)
        for i in prange(1, size-1):
            for j in range(1, size-1):
                if (i + j) % 2 == 1 and domain[i, j] == 0:
                    old_val = concentration[i, j]
                    new_val = 0.25 * (
                        concentration[i+1, j] +
                        concentration[i-1, j] +
                        concentration[i, j+1] +
                        concentration[i, j-1]
                    )
                    concentration[i, j] = (1 - omega) * old_val + omega * new_val
                    diff = abs(concentration[i, j] - old_val)
                    if diff > max_diff:
                        max_diff = diff
        # Periodic boundary condition along y-axis
        for i in range(1, concentration.shape[0]-1):
            concentration[i,-1] = (1 - omega) * concentration[i,-1] + 0.25 * omega * (
                concentration[i+1,-1] + concentration[i-1,-1] + concentration[i,0] + concentration[i,-2]
            )

        # Convergence check
        if max_diff < tolerance:
            return iter_count + 1

    return max_iterations

@jit(nopython=True)
def find_growth_candidates_numba(domain, size):
    max_candidates = size * size
    candidates_row = np.zeros(max_candidates, dtype=np.int32)
    candidates_col = np.zeros(max_candidates, dtype=np.int32)

    count = 0

    # Check all cells
    for row in range(0, size):
        for col in range(0, size):
            if domain[row, col] == 0 and is_neighboring_cluster(domain, row, col):
                candidates_row[count] = row
                candidates_col[count] = col
                count += 1

    return candidates_row[:count], candidates_col[:count]

@jit(nopython=True)
def calculate_growth_probabilities_numba(candidates_i, candidates_j, concentration, eta):
    count = len(candidates_i)
    probs = np.zeros(count, dtype=np.float64)

    # Calculate unnormalized probabilities
    for idx in range(count):
        row, col = candidates_i[idx], candidates_j[idx]
        # Ensure concentration value is positive (avoid numerical errors)
        conc_value = max(0.0, concentration[row, col])
        probs[idx] = conc_value ** eta

    # Normalize probabilities
    total = np.sum(probs)
    if total > 0:
        probs = probs / total

    return probs

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
def run_dla_experiments(etas=[0.5, 1.0, 2.0], size=100, steps=1000, parallel=False):
    results = []
    optimal_omega = calculate_optimal_omega(size)

    for eta in etas:
        sim = DLASimulation(size=size, eta=eta, omega=optimal_omega, parallel=parallel)
        sim.run_simulation(steps=steps)
        results.append(sim)

        plot_grid(sim.domain, title=f"DLA Structure with η = {eta} in {steps} steps", savefig=True, filename=f"images/dla/dla_eta_{eta}.pdf", 
                  cmap=colors.ListedColormap(['white', 'black']))
    
    plot_comparison(results, title="DLA Structure for Different η Values", sub_titles=[f"η = {eta}" for eta in etas], savefig=False, 
                    filename="images/dla/dla_comparison.pdf", cmap=colors.ListedColormap(['white', 'black']))
    return results


# Benchmark different omega values for SOR convergence
def find_optimal_omega(size=100, num_omegas=5, steps=50, etas=[0.5, 1.0, 2.0]):
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
    plt.savefig("images/dla/sor_benchmark_eta.pdf")
    
    return results

# Run the simulation with optimized parameters
if __name__ == "__main__":

    # Compile JIT functions (first call will compile)
    print("Compiling Numba functions (first run)...")
    small_domain = np.zeros((10, 10), dtype=np.int32)
    small_concentration = np.zeros((10, 10), dtype=np.float64)
    _ = sor_iteration(small_concentration, small_domain, 1.5, 1e-5, 10)
    _ = sor_iteration_parallel(small_concentration, small_domain, 1.5, 1e-5, 10)
    _, _ = find_growth_candidates_numba(small_domain, 10)
    _ = calculate_growth_probabilities_numba(
        np.array([1]), np.array([1]), small_concentration, 1.0)
    print("Compilation completed")

    # Increase font size
    plt.rcParams.update({'font.size': 14})

    # Find optimal omega for SOR
    # best_omega = find_optimal_omega(size=100, num_omegas=30, etas=[0, 0.5, 1.0, 1.5])

    # Test different eta values with the optimal omega
    print("\nRunning full DLA simulations with optimal omega...")
    results = run_dla_experiments(etas=[0, 0.5, 1.0, 1.5], size=100, steps=500)

    # Measure fractal dimension
    print("\nMeasuring fractal dimension...")
    for sim in results:
         pass