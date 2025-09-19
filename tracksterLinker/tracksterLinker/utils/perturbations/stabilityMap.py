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
    eta_all = []
    phi_all = []
    val_all = []
    labels = []

    for g in graphs:
        eta_all.extend(g["eta"])
        phi_all.extend(g["phi"])
        val_all.extend(g[values])
        labels.extend([g.get("label", "Graph")] * len(g["eta"]))

    eta_all = np.array(eta_all)
    phi_all = np.array(phi_all)
    val_all = np.array(val_all)
    labels = np.array(labels)

    # --- average duplicates ---
    buckets = defaultdict(list)
    for e, p, v in zip(eta_all, phi_all, val_all):
        buckets[(round(e,5), round(p,5))].append(v)  # round for numerical stability

    eta_unique = np.array([k[0] for k in buckets.keys()])
    phi_unique = np.array([k[1] for k in buckets.keys()])
    val_unique = np.array([np.mean(vs) for vs in buckets.values()])

    # --- grid for interpolation ---
    grid_eta = np.linspace(min(eta_unique), max(eta_unique), resolution)
    grid_phi = np.linspace(min(phi_unique), max(phi_unique), resolution)
    grid_eta, grid_phi = np.meshgrid(grid_eta, grid_phi)

    grid_values = griddata(
        (eta_unique, phi_unique), val_unique,
        (grid_eta, grid_phi), method="cubic", fill_value=np.nan
    )
    
    fig, ax = plt.subplots(figsize=(8,6))
    # Plot heatmap
    im = ax.imshow(
        grid_values, origin="lower", aspect="auto",
        extent=(min(eta_unique), max(eta_unique), min(phi_unique), max(phi_unique)),
        cmap="GnBu"
    )
    
    ax.set_title(g.get("label", "Graph"))
    ax.set_xlabel("eta")
    ax.set_ylabel("phi")
    
    # Shared colorbar
    fig.colorbar(im, ax=axes, label=values)
    if file is None and folder is None:
        plt.show()
    else:
        path = os.path.join(folder, file)
        plt.savefig(path)
