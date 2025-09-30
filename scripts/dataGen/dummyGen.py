import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp

def generate_dummy_simtrackster_data(stats, n_events=1, random_state=None):
    rng = np.random.default_rng(random_state)
    dummy_events = []

    for _ in range(n_events):
        event = []
        
        for i in range(stats["poses_mean"].shape[0]):
            # centroid of cluster
            print(stats["poses_mean"])
            mean = stats["poses_mean"][i]
            std = stats["poses_std"][i]
            print("mean", mean)
            centroid = rng.normal(mean, std)

            # covariance (make symmetric)
            cov_mean = stats["cov_mean"][i, :]
            cov_std = stats["cov_std"][i, :]
            cov = cov_mean + rng.normal(0, cov_std)

            # number of simTracksters in this cluster
            mean_count = stats["per_cluster_count_mean"][i]
            std_count = stats["per_cluster_count_std"][i]
            n_points = max(1, int(rng.normal(mean_count, std_count)))

            # generate simTracksters in this cluster
            cluster_points = rng.multivariate_normal(centroid, cov, size=n_points)

            event.append({
                "centroid": centroid,
                "covariance": cov,
                "points": cluster_points
            })
        
        dummy_events.append(event)

    return dummy_events


def plot_event_3d(events):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    for event in events:
        for cluster in event:
            print(cluster)
            pts = cluster["points"]
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=10, alpha=0.6)
            ax.scatter(cluster["centroid"][0],
                       cluster["centroid"][1],
                       cluster["centroid"][2],
                       c="red", marker="x", s=80)

        ax.set_title("Dummy simTrackster clusters (3D example)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.savefig("/home/czeh/dummy_data/simTrackster.png")


if __name__ == '__main__':
    base_folder = "/home/czeh"
    output_folder = osp.join(base_folder, "dummy_data/data_stats")
    data = np.load(osp.join(output_folder, "simTrackster.npz"), allow_pickle=True)

    print(data)
    # num_signals = res["signal_simTrackster"]["count_mean"].shape[0]
    dummy_event = generate_dummy_simtrackster_data(data["signal_simTrackster"].item())
    plot_event_3d(dummy_event)
