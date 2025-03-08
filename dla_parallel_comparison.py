import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import seaborn as sns

from dla import DLASimulation, calculate_optimal_omega

def benchmark_parallel_vs_sequential(grid_sizes, steps=10, eta=1.0):
    """
    Run benchmarks comparing parallel and sequential implementations of DLA simulation
    across different grid sizes.
    
    Args:
        grid_sizes: List of grid sizes to test
        steps: Number of growth steps to simulate for each run
        eta: The eta parameter for the DLA simulation
        
    Returns:
        Dictionary containing benchmark results
    """
    results = {
        'grid_size': [],
        'parallel_time': [],
        'sequential_time': [],
        'speedup': []
    }
    
    for size in tqdm(grid_sizes, desc="Benchmarking grid sizes"):
        # Calculate optimal omega for this grid size
        optimal_omega = calculate_optimal_omega(size)
        
        # Sequential run
        print(f"\nRunning sequential simulation for grid size {size}...")
        sim_sequential = DLASimulation(size=size, eta=eta, omega=optimal_omega, parallel=False)
        start_time = time.time()
        for _ in range(steps):
            sim_sequential.grow_step()
        sequential_time = time.time() - start_time
        
        # Parallel run
        print(f"Running parallel simulation for grid size {size}...")
        sim_parallel = DLASimulation(size=size, eta=eta, omega=optimal_omega, parallel=True)
        start_time = time.time()
        for _ in range(steps):
            sim_parallel.grow_step()
        parallel_time = time.time() - start_time
        
        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        
        # Store results
        results['grid_size'].append(size)
        results['sequential_time'].append(sequential_time)
        results['parallel_time'].append(parallel_time)
        results['speedup'].append(speedup)
        
        print(f"Grid size: {size}, Sequential: {sequential_time:.2f}s, Parallel: {parallel_time:.2f}s, Speedup: {speedup:.2f}x")
    
    return results

def plot_benchmark_results(results):
    plt.figure(figsize=(10, 6))
    plt.plot(results['grid_size'], results['speedup'], 'd-', color='green', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Break-even (1.0x)')
    plt.xlabel('Grid Size')
    plt.ylabel('Speedup (Sequential / Parallel)')
    plt.title('Parallel Speedup vs Grid Size')
    plt.grid(True)
        
    # Save the figure
    plt.tight_layout()
    plt.savefig("images/dla/dla_parallel_benchmark.pdf")
    plt.show()

if __name__ == "__main__":
    grid_sizes = np.linspace(100, 1000, num=10, dtype=int)

    results = benchmark_parallel_vs_sequential(grid_sizes, steps=50, eta=1.0)
    
    plt.rcParams.update({'font.size': 14})

    # Plot the results
    plot_benchmark_results(results)
    
    # Print the final speedup summary
    print("\nBenchmark Summary (Grid Size | Speedup):")
    for i, size in enumerate(results['grid_size']):
        print(f"{size} | {results['speedup'][i]:.2f}x")
        
    # Find the grid size with the best speedup
    best_idx = np.argmax(results['speedup'])
    print(f"\nBest speedup achieved at grid size {results['grid_size'][best_idx]}: {results['speedup'][best_idx]:.2f}x")
    
    # Calculate average speedup1.0, 0.7, 0.4, 0.1
    avg_speedup = np.mean(results['speedup'])
    print(f"Average speedup across all grid sizes: {avg_speedup:.2f}x")