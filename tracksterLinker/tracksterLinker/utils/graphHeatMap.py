import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from collections import defaultdict

from scipy.spatial import cKDTree
import numpy.ma as ma

class GraphHeatmap:
    def __init__(self, resolution=200):
        """
        Class for combining multiple graphs into one interpolated heatmap,
        keeping only running means (no raw storage).
        
        Parameters
        ----------
        resolution : int
            Grid resolution for interpolation.
        """
        self.resolution = resolution
        self.data = {}  # {(eta, phi): (mean_value, count)}

    def add_graph(self, eta, phi, values):
        """
        Add a graph's nodes to the dataset, updating running means.
        
        Parameters
        ----------
        eta, phi, values : array-like
            Node positions and values.
        """
        for e, p, v in zip(eta, phi, values):
            key = (round(float(e), 1), round(float(p), 1))  # rounding for stability
            if key not in self.data:
                self.data[key] = (v, 1)  # first entry
            else:
                old_mean, n = self.data[key]
                new_mean = (old_mean * n + v) / (n + 1)
                self.data[key] = (new_mean, n + 1)

    def _get_arrays(self):
        """Convert dict storage into numpy arrays."""
        eta = np.array([k[0] for k in self.data.keys()])
        phi = np.array([k[1] for k in self.data.keys()])
        values = np.array([v[0] for v in self.data.values()])
        return eta, phi, values

    def plot(self, show_nodes=False, max_distance=0.1, file=None, folder=None):
        """Plot the combined interpolated heatmap."""
        eta, phi, values = self._get_arrays()

        # Grid for interpolation
        grid_eta = np.linspace(1.5, 3.1, self.resolution)
        grid_phi = np.linspace(-3.14, 3.14, self.resolution)
        grid_eta, grid_phi = np.meshgrid(grid_eta, grid_phi)

        print(len(eta))
        grid_values = griddata(
            (eta, phi), values,
            (grid_eta, grid_phi),
            method="nearest", fill_value=np.nan
        )
        print("after")

        tree = cKDTree(np.c_[eta, phi])
        dist, _ = tree.query(np.c_[grid_eta.ravel(), grid_phi.ravel()])
        dist = dist.reshape(grid_eta.shape)

        # Mask out far-away cells
        grid_values = ma.masked_where(dist > max_distance, grid_values)


        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            grid_values, origin="lower", aspect="auto",
            extent=(1.5, 3.1, -3.14, 3.14),
            cmap="GnBu", vmin=0
        )
        fig.colorbar(im, ax=ax, label="Value")

        if show_nodes:
            ax.scatter(eta, phi, s=15, c="k", alpha=0.7, label="nodes")

        ax.set_xlabel("eta")
        ax.set_ylabel("phi")
        ax.set_title("Incremental Combined Heatmap")
        if show_nodes:
            ax.legend()
        if file is None and folder is None:
            plt.show()
        else:
            path = os.path.join(folder, file)
            plt.savefig(path)

