import numpy as np
import matplotlib.pyplot as plt
import random
import numba
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors

from plots import plot_grid, plot_comparison
from utilities import is_neighboring_cluster

def initialize_grid(size):
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    # Row, Col
    grid[0, center] = 1  # Start with a seed at the center bottom
    return grid

@numba.jit(nopython=True)
def choose_direction():
    directions = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)], dtype=np.int32)
    index = np.random.randint(0, 4)  # Random integer from {0, 1, 2, 3}
    return directions[index]

@numba.jit(nopython=True)
def random_walk(grid, ps):
    size = grid.shape[0]
    row, col = size-1, random.randint(0, size-1)  # Start at random top position
    retry = 0
    while True:
        drow, dcol = choose_direction()
        new_row = row + drow
        new_col = (col + dcol) % size  # Apply periodic BC

        if new_row < 0 or new_row >= size:  # If it exits top/bottom, stop
            return None
        
        if retry < 10:
            if grid[new_row, new_col] == 1:
                retry += 1
                continue 
        else :
            return None

        row, col = new_row, new_col  # Move walker

        if is_neighboring_cluster(grid, row, col):
            if random.random() < ps:  # Stick with probability ps
                grid[row, col] = 1
                return (row, col)

def monte_carlo_dla(size, num_walkers, ps):
    grid = initialize_grid(size)
    history = []
    history.append((grid.copy(), None))
    
    for _ in tqdm(range(num_walkers)):
        retry = 0
        while retry < 10:
            result = random_walk(grid, ps)
            if result is not None:
                history.append((grid.copy(), [result]))
                break
            retry += 1
    return grid, history

def create_animation(history, ps):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    img = ax.imshow(history[0][0], cmap='gray_r', origin='lower', animated=True)
    walker_plot = ax.scatter([], [], color='red', s=10, label="Walker Pos", animated=True)
    
    title = ax.set_title(r"Monte Carlo DLA ($p_s=%.2f$)" % ps)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Create an initial legend
    legend = ax.legend(loc='best')

    plt.close()  # Prevent display of the initial frame

    def update(frame):
        grid, walk_positions = history[frame]
        img.set_array(grid)

        if walk_positions is not None and len(walk_positions) > 0:
            x_vals, y_vals = zip(*walk_positions)
            walker_plot.set_offsets(np.column_stack([y_vals, x_vals]))
        else:
            walker_plot.set_offsets(np.empty((0, 2)))

        title.set_text(r"Monte Carlo DLA ($p_s=%.2f$)" % (ps))
        return img, walker_plot, title, legend

    ani = FuncAnimation(fig, update, frames=len(history), blit=True, interval=100)
    return ani

if __name__ == "__main__":
    # Parameters
    size = 100
    num_walkers = 10000
    ps_values = [0.1, 0.2, 0.4, 1.0]
    animation = False
    plt.rcParams.update({'font.size': 14})


    # NB: TALK ABOUT PERCOLATION THEORY FOR A 2D GRIDs
    # MEASURE THE PERCOLATION THRESHOLD AND THE FRACTAL DIMENSION
    results = []
    for ps in ps_values:
        grid, history = monte_carlo_dla(size, num_walkers, ps)
        plot_grid(grid, title=r"Monte Carlo DLA, $p_s = %.2f$" % ps, filename=f"images/mca/mca_{ps}.pdf",
                  savefig=False, cmap=colors.ListedColormap(['white', 'black']), colorbar=False)
        results.append(grid)

        if animation:
            ani = create_animation(history, ps)
            ani.save(f'images/mca/dla_animation_ps_{ps}.mp4', writer='ffmpeg', fps=10, dpi=600)
            plt.show()

    plot_comparison(results, title="Monte Carlo DLA for Different $p_s$ Values", sub_titles=[r"$p_s = %.2f$" % ps for ps in ps_values], savefig=False,
                    filename="images/mca/mca_comparison.pdf", cmap=colors.ListedColormap(['white', 'black']), colorbar=False)
