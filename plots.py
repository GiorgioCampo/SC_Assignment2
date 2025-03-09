import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
import numpy as np

def plot_comparison(results, title="", sub_titles="", savefig=False, filename=None, cmap=None, colorbar=None):
    fig, axes = plt.subplots(int(len(results)/2), int(len(results)/2), figsize=(8, 8))
    axes = axes.flatten()
    for i, grid in enumerate(results):
        img = axes[i].imshow(grid, cmap=cmap, origin='lower')
        if colorbar:
            plt.colorbar(img, label="U - V Concentration Difference")

        axes[i].set_title(sub_titles[i])
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")

    plt.suptitle(title)
    plt.tight_layout()
    if savefig:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_grid(grid, title, savefig=False, filename=None, cmap=None, colorbar=None):
    plt.figure(figsize=(8, 8))
    img = plt.imshow(grid, cmap=cmap, origin='lower')
    
    if colorbar:
        plt.colorbar(img, label="U - V Concentration Difference")

    plt.title(title)
    plt.tight_layout()
    if savefig:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def animate_diffusion(times, solutions, interval=200, time_multiplier=10,
                      save_animation=False, filename='diffusion_equilibrium_slow.gif'):
    fig, ax = plt.subplots(figsize=(6, 5))
    c_min, c_max = 0, 1
    img = ax.imshow(solutions[0], cmap='viridis', origin='lower', extent=[0, 1, 0, 1], vmin=c_min, vmax=c_max)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Concentration')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    title = ax.set_title(f'Time: {times[0] * time_multiplier:.3f} (scaled)')

    def update(frame):
        img.set_array(solutions[frame])
        title.set_text(f'Time: {times[frame] * time_multiplier:.3f} (scaled)')
        return img, title

    anim = animation.FuncAnimation(fig, update, frames=len(times), interval=interval, blit=True)

    if save_animation:
        anim.save(filename, writer='pillow', fps=10, dpi=150)  
        print(f"Animation saved as {filename}.")

    plt.tight_layout()
    plt.show()