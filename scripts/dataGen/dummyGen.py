import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp

def generate_multi_event(stats, n_events=1, random_state=None):
    dummy_events = []

    print(f"signal: {stats['signal_simTrackster'].item()['poses_mean'].shape[0]}, pu: {stats['pu_simTrackster'].item()['poses_mean'].shape[0]}")

    for _ in range(n_events):
        event = []
        event.extend(generate_dummy_data(stats["signal_simTrackster"].item(), random_state=random_state))
        event.extend(generate_dummy_data(stats["pu_simTrackster"].item(), random_state=random_state))
        dummy_events.append(event)

    return dummy_events


def generate_dummy_data(stats, n_events=1, random_state=None):
    rng = np.random.default_rng(random_state)

    event = []
    for i in range(stats["poses_mean"].shape[0]):
        all_tracksters = []

        # centroid of cluster
        mean = stats["poses_mean"][i]
        std = stats["poses_std"][i]
        centroid = rng.normal(mean, std)

        # covariance (make symmetric)
        cov_mean = stats["cov_mean"][i, :]
        cov_std = stats["cov_std"][i, :]
        cov = cov_mean + rng.normal(0, cov_std)

        # number of simTracksters in this cluster
        mean_count = stats["per_cluster_count_mean"]
        std_count = stats["per_cluster_count_std"]
        n_points = max(1, int(rng.normal(mean_count, std_count)))

        # generate simTracksters in this cluster
        cluster_points = rng.multivariate_normal(centroid, cov, size=n_points)

        for cluster in cluster_points:
            new_pos = cluster + stats["trackster_rel_poses_mean"]
            trackster_count = max(1, int(rng.normal(stats["trackster_count_mean"], stats["trackster_count_std"])))
            tracksters = rng.normal(new_pos, stats["trackster_rel_poses_std"], size=(trackster_count, new_pos.shape[0]))
            all_tracksters.append(tracksters)

        event.append({
            "sim_trackster": cluster_points,
            "trackster": np.concatenate(all_tracksters)
        })
        
    return event


def plot_event_3d(events):
    for i, event in enumerate(events):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        for cluster in event:
            pts = cluster["trackster"]
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=10, alpha=0.6)
            ax.scatter(cluster["sim_trackster"][:, 0],
                       cluster["sim_trackster"][:, 1],
                       cluster["sim_trackster"][:, 2],
                       c="red", marker="x", s=8)

        ax.set_title("Dummy simTrackster clusters (3D example)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.savefig(f"/home/czeh/dummy_data/images/data_{i}.png")


if __name__ == '__main__':
    base_folder = "/home/czeh"
    output_folder = osp.join(base_folder, "dummy_data/data_stats")
    data = np.load(osp.join(output_folder, "simTrackster.npz"), allow_pickle=True)

    dummy_event = generate_multi_event(data, n_events=10)
    plot_event_3d(dummy_event)
