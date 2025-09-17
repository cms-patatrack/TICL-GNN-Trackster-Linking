import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

def plot_graphs_heatmap(graphs, mode="3d", values="values", file=None, folder=None):
    """
    Plot multiple graphs with node values as heatmap colors.
    
    Parameters
    ----------
    graphs : list of dict
        Each dict must have keys:
        - 'eta': array of eta positions
        - 'phi': array of phi positions
        - 'z': array of z positions
        - 'values': array of values (neg=good, pos=bad)
        - 'label': name of the graph (for legend)
    mode : str, optional
        "2d" (eta vs phi) or "3d" (eta, phi, z). Default is "3d".
    """
    
    # Marker styles for different graphs
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    
    # Collect all values for global colormap scaling
    all_values = np.concatenate([g[values] for g in graphs])
    vmin, vmax = np.min(all_values), np.max(all_values)
    
    if mode == "3d":
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        
        for i, g in enumerate(graphs):
            sc = ax.scatter(
                g["eta"], g["phi"], g["z"],
                c=g[values]/g["full_energy"], cmap="GnBu",
                vmin=vmin, vmax=vmax,
                marker=markers[i % len(markers)],
                s=g["energy"], edgecolor="k", alpha=0.8,
                label=g.get("label", f"Graph {i+1}")
            )
        
        ax.set_xlabel("eta")
        ax.set_ylabel("phi")
        ax.set_zlabel("z")
    
    elif mode == "2d":
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for i, g in enumerate(graphs):
            sc = ax.scatter(
                g["eta"], g["phi"],
                c=g[values]/g["full_energy"], cmap="GnBu",
                vmin=vmin, vmax=vmax,
                marker=markers[i % len(markers)],
                s=60, edgecolor="k", alpha=0.8,
                label=g.get("label", f"Graph {i+1}")
            )
        
        ax.set_xlabel("eta")
        ax.set_ylabel("phi")
    
    else:
        raise ValueError("mode must be '2d' or '3d'")
    
    # Shared colorbar
    cbar = plt.colorbar(sc, ax=ax, label="Value")
    
    # Legend
    ax.legend()

    if file is None and folder is None:
        plt.show()
    else:
        path = os.path.join(folder, file)
        plt.savefig(path)


def plot_graphs_heatmap_interp(graphs, values="values", resolution=200, file=None, folder=None):
    """
    Interpolated 2D heatmap for multiple graphs (eta vs phi).
    
    Parameters
    ----------
    graphs : list of dict
        Each dict must have keys:
        - 'eta': array of eta positions
        - 'phi': array of phi positions
        - 'values': array of values
        - 'label': name of the graph
    resolution : int
        Grid resolution for interpolation.
    """
    
    fig, axes = plt.subplots(1, len(graphs), figsize=(6*len(graphs), 5))
    if len(graphs) == 1:
        axes = [axes]
    
    for ax, g in zip(axes, graphs):
        # Grid for interpolation
        grid_eta = np.linspace(min(g["eta"]), max(g["eta"]), resolution)
        grid_phi = np.linspace(min(g["phi"]), max(g["phi"]), resolution)
        grid_eta, grid_phi = np.meshgrid(grid_eta, grid_phi)
        
        # Interpolation
        grid_values = griddata(
            (g["eta"], g["phi"]), g[values] * g["energy"],
            (grid_eta, grid_phi), method="cubic", fill_value=np.nan
        )
        
        # Plot heatmap
        im = ax.imshow(
            grid_values, origin="lower", aspect="auto",
            extent=(min(g["eta"]), max(g["eta"]), min(g["phi"]), max(g["phi"])),
            cmap="GnBu"
        )
        
        ax.set_title(g.get("label", "Graph"))
        ax.set_xlabel("eta")
        ax.set_ylabel("phi")
    
    # Shared colorbar
    fig.colorbar(im, ax=axes, label=value)
    if file is None and folder is None:
        plt.show()
    else:
        path = os.path.join(folder, file)
        plt.savefig(path)
