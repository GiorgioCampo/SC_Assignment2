from numba import jit, prange
import numpy as np


__all__ = ['find_growth_candidates_numba', 'calculate_growth_probabilities_numba', 'is_neighboring_cluster', 'sor_iteration', 'sor_iteration_parallel']

@jit(nopython=True)
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
