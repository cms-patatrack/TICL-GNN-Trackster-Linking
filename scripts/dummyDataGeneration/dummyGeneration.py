import os.path as osp
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp

import awkward as ak

# interest_features = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "num_LCs", "raw_energy", "z_min", "z_max", "LC_density"]

interest_features = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "EV1", "EV2", "EV3",
                         "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "num_LCs", "num_hits", "raw_energy", "raw_em_energy", "photon_prob", "electron_prob", "muon_prob",
                         "neutral_pion_prob", "charged_hadron_prob", "neutral_hadron_prob", "z_min", "z_max", "LC_density", "trackster_density", "time"]

def generate_multi_event(stats, n_events=1, random_state=None):
    dummy_events = []

    print(f"signal: {stats['signal_simTrackster'].item()['poses_mean'].shape[0]}, pu: {stats['pu_simTrackster'].item()['poses_mean'].shape[0]}")

    for _ in range(n_events):
        signal_event, signal_y, signal_n = generate_dummy_data(stats["signal_simTrackster"].item(), random_state=random_state)
        signal_isPU = np.zeros(signal_n)
        pu_event, pu_y, pu_n = generate_dummy_data(stats["pu_simTrackster"].item(), random_state=random_state, first_id=signal_y[-1][0])

        signal_event.extend(pu_event)
        signal_y.extend(pu_y)
        event = {
            "y": np.concatenate(signal_y),
            "isPU": np.concatenate([signal_isPU, np.ones(pu_n)])
        }

        event = {
            "y": np.concatenate(signal_y),
            "isPU": signal_isPU
        }
        signal_event = np.concatenate(signal_event)
        for idx, name in enumerate(interest_features):
            event[name] = signal_event[:, idx]

        event["barycenter_z"] = np.abs(event["barycenter_z"])
        event["raw_energy"] = np.abs(event["raw_energy"])
        event = ak.Record(event)
        print(f"Event with signal {signal_n} and pu {pu_n}")
        dummy_events.append(event)

    return ak.Array(dummy_events)


def generate_dummy_data(stats, random_state=None, first_id=0):
    rng = np.random.default_rng(random_state)

    all_tracksters = []
    all_y = []
    nTrackster = 0
    for clu in range(stats["poses_mean"].shape[0]):
        # centroid of cluster
        mean = stats["poses_mean"][clu]
        std = stats["poses_std"][clu]
        centroid = rng.normal(mean, std)

        # covariance (make symmetric)
        cov_mean = stats["cov_mean"][clu, :]
        cov_std = stats["cov_std"][clu, :]
        cov = cov_mean + rng.normal(0, cov_std)

        # number of simTracksters in this cluster
        mean_count = stats["per_cluster_count_mean"]
        std_count = stats["per_cluster_count_std"]
        n_points = max(1, int(rng.normal(mean_count, std_count)))

        # generate simTracksters in this cluster
        cluster_points = rng.multivariate_normal(centroid, cov, size=n_points)

        for sim_idx, cluster in enumerate(cluster_points):
            new_pos = cluster + stats["trackster_rel_poses_mean"][clu, :]
            trackster_count = max(1, int(rng.normal(stats["trackster_count_mean"], stats["trackster_count_std"])))
            tracksters = rng.normal(new_pos, stats["trackster_rel_poses_std"][clu, :], size=(trackster_count, new_pos.shape[0]))

            x = tracksters[:, 0]
            y = tracksters[:, 1]
            z = tracksters[:, 2]
            tracksters[:, 4] = np.arctan2(y, x)
            theta = np.arctan2(np.sqrt(x**2 + y**2), z)
            tracksters[:, 3] = -np.log(np.tan(theta / 2))

            nTrackster += tracksters.shape[0]
            all_tracksters.append(tracksters)
            all_y.append(np.full(tracksters.shape[0], (sim_idx+first_id)))

    return all_tracksters, all_y, nTrackster


def plot_event_3d(events):
    fig = plt.figure(figsize=(8,8))
    for i, event in enumerate(events):
        fig.clear()
        ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=10, alpha=0.6)
        # c = (event["isPU"] == 1) ? "r" : "b"
        c = ["r" if x == 1 else "b" for x in event["isPU"]]
        ax.scatter(event["barycenter_x"],
                   event["barycenter_y"],
                   event["barycenter_z"],
                   s=event["raw_energy"], c=c, alpha=0.6)

        ax.set_title("Dummy simTrackster clusters (3D example)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.savefig(f"/home/czeh/dummy_data/images/data_{i}.png")


if __name__ == '__main__':
    base_folder = "/home/czeh"
    in_folder = osp.join(base_folder, "dummy_data/data_stats")
    data_folder = osp.join(base_folder, "dummy_data/data")
    data = np.load(osp.join(in_folder, "simTrackster.npz"), allow_pickle=True)
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(osp.join(data_folder, "train"), exist_ok=True)
    os.makedirs(osp.join(data_folder, "test"), exist_ok=True)

    n_events = 10
    n_files_train = 1000
    n_files_test = 100

    for i in range(500, n_files_train):
        dummy_event = generate_multi_event(data, n_events=n_events)
        #plot_event_3d(dummy_event)

        ak.to_parquet(dummy_event, osp.join(data_folder, "train", f"dummy_data_{i}.parquet"))

    for i in range(50, n_files_test):
        dummy_event = generate_multi_event(data, n_events=n_events)
        #plot_event_3d(dummy_event)
        
        ak.to_parquet(dummy_event, osp.join(data_folder, "val", f"dummy_data_{i}.parquet"))
