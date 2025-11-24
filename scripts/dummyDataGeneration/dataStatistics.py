import numpy as np
import cupy as cp

import sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import tracksterLinker
from tracksterLinker.utils.graphUtils import *
from tracksterLinker.datasets.NeoGNNDataset import *


def statistics_of_gaussians(centers, covariances, poses, rel_poses, counts, cluster=10):
    kmeans = KMeans(n_clusters=cluster, random_state=0).fit(centers)
    labels = kmeans.labels_
    print(poses)
    trackster_labels = kmeans.predict(poses)

    cluster_center_means = []
    cluster_center_stds = []
    cluster_cov_means = []
    cluster_cov_stds = []

    trackster_rel_poses_mean = []
    trackster_rel_poses_std = []
    trackster_counts_mean = []
    trackster_counts_std = []


    for cluster_id in range(kmeans.n_clusters):
        cluster_centers = centers[labels == cluster_id]
        trackster_rel_poses = rel_poses[trackster_labels == cluster_id]
        # trackster_counts = counts[labels == cluster_id]

        cluster_covs = covariances[labels == cluster_id]
        cov_mean = cluster_covs.mean(axis=0)
        cov_std = cluster_covs.std(axis=0)

        rel_poses_mean = trackster_rel_poses.mean(axis=0)
        rel_poses_std = trackster_rel_poses.std(axis=0)

        # counts_mean = trackster_counts.mean(axis=0)
        # counts_std = trackster_counts.std(axis=0)

        cluster_cov_means.append(cov_mean)
        cluster_cov_stds.append(cov_std)

        trackster_rel_poses_mean.append(rel_poses_mean)
        trackster_rel_poses_std.append(rel_poses_std)
        
        # trackster_counts_mean.append(counts_mean)
        # trackster_counts_std.append(counts_std)

        # mean & std of positions
        center_mean = cluster_centers.mean(axis=0)
        center_std = cluster_centers.std(axis=0)
        
        cluster_center_means.append(center_mean)
        cluster_center_stds.append(center_std)

    cluster_center_means = np.array(cluster_center_means)
    cluster_center_stds = np.array(cluster_center_stds)

    cluster_cov_means = np.array(cluster_cov_means)
    cluster_cov_stds = np.array(cluster_cov_stds)

    trackster_rel_poses_mean = np.array(trackster_rel_poses_mean)
    trackster_rel_poses_std = np.array(trackster_rel_poses_std)

    trackster_counts_mean = counts.mean()
    trackster_counts_std = counts.std()

    return cluster_center_means, cluster_center_stds, cluster_cov_means, cluster_cov_stds, trackster_rel_poses_mean, trackster_rel_poses_std, trackster_counts_mean, trackster_counts_std


base_folder = "/home/czeh"
hist_folder = osp.join(base_folder, "new_graph_histo")
output_folder = osp.join(base_folder, "dummy_data/data_stats")
os.makedirs(output_folder, exist_ok=True)

files = glob(f"{hist_folder}/test/*.root")

# interest_features = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "num_LCs", "raw_energy", "z_min", "z_max", "LC_density"]

interest_features = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "EV1", "EV2", "EV3",
                         "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "num_LCs", "num_hits", "raw_energy", "raw_em_energy", "photon_prob", "electron_prob", "muon_prob",
                         "neutral_pion_prob", "charged_hadron_prob", "neutral_hadron_prob", "z_min", "z_max", "LC_density", "trackster_density", "time"]

count_signal_simTrackster = []
count_pu_simTrackster = []
cluster_poses_simTrackster = []
pu_cluster_poses_simTrackster = []
cluster_counts_simTrackster = []
pu_cluster_counts_simTrackster = []
cluster_cov_simTrackster = []
pu_cluster_cov_simTrackster = []

trackster_count = []
pu_trackster_count = []
trackster_rel_poses = []
pu_trackster_rel_poses = []
pu_trackster_poses = []
trackster_poses = []

for file in files:
    file = uproot.open(file)

    allGNNtrain = load_branch_with_highest_cycle(file, 'ticlDumperGNN/GNNTraining')
    allGNNtrain_array = allGNNtrain.arrays()
    print(allGNNtrain_array.fields)

    cnt = 0    
    for event in allGNNtrain_array:
        nTracksters = 0

        features = np.stack([ak.to_numpy(event[f"node_{field}"]) for field in interest_features], axis=1)
        signal_poses = np.zeros((len(event["simTrackster_isPU"]), features.shape[1]))
        pu_poses = np.zeros((np.sum(event["simTrackster_isPU"]), features.shape[1]))
        sim_i = 0
        pu_i = 0
        for simTr in range(len(event["simTrackster_isPU"])):
            trackster = ak.to_numpy(features[event["node_match_idx"] == simTr]) 
            if (trackster.shape[0] == 0):
                continue
            nTracksters += trackster.shape[0]

            energy = trackster[:, 16]
            sum_energy = np.tile(energy, (trackster.shape[1], 1))
            sim_features = np.sum(trackster * sum_energy.T, axis=0) / np.sum(energy)

            trackster_rel = trackster - sim_features

            if(not event["simTrackster_isPU"][simTr]):
                signal_poses[sim_i] = sim_features 
                sim_i +=1
                trackster_rel_poses.append(trackster_rel)
                trackster_poses.append(trackster)
                trackster_count.append(trackster.shape[0])

            else:
                pu_poses[pu_i] = sim_features 
                pu_i += 1
                pu_trackster_rel_poses.append(trackster_rel)
                pu_trackster_poses.append(trackster)
                pu_trackster_count.append(trackster.shape[0])

        print(f"Event with {nTracksters}, signal: {sim_i}, pu: {pu_i}")
        signal_poses = signal_poses[:sim_i]
        pu_poses = pu_poses[:pu_i]
        
        signal_kmeans = GaussianMixture(n_components=min(21, sim_i)).fit(signal_poses)
        pu_kmeans = GaussianMixture(n_components=min(200, pu_i)).fit(pu_poses)

        cluster_poses_simTrackster.append(signal_kmeans.means_)
        cluster_cov_simTrackster.append(signal_kmeans.covariances_)
        pu_cluster_poses_simTrackster.append(pu_kmeans.means_)
        pu_cluster_cov_simTrackster.append(pu_kmeans.covariances_)
        cluster_counts_simTrackster.append(np.bincount(signal_kmeans.predict(signal_poses)))
        pu_cluster_counts_simTrackster.append(np.bincount(pu_kmeans.predict(pu_poses)))
        
        count_signal_simTrackster.append(sim_i)
        count_pu_simTrackster.append(pu_i)

        if (cnt == 10):
            break
        cnt += 1
    break

count_signal_simTrackster = np.array(count_signal_simTrackster)
count_pu_simTrackster = np.array(count_pu_simTrackster)
cluster_poses_simTrackster = np.concatenate(cluster_poses_simTrackster)
cluster_cov_simTrackster = np.concatenate(cluster_cov_simTrackster)
pu_cluster_poses_simTrackster = np.concatenate(pu_cluster_poses_simTrackster)
pu_cluster_cov_simTrackster = np.concatenate(pu_cluster_cov_simTrackster)
cluster_counts_simTrackster = np.concatenate(cluster_counts_simTrackster)
pu_cluster_counts_simTrackster = np.concatenate(pu_cluster_counts_simTrackster)

trackster_rel_poses = np.concatenate(trackster_rel_poses)
trackster_poses = np.concatenate(trackster_poses, dtype=float)
pu_trackster_rel_poses = np.concatenate(pu_trackster_rel_poses)
pu_trackster_poses = np.concatenate(pu_trackster_poses, dtype=float)
trackster_count = np.array(trackster_count)
pu_trackster_count = np.array(pu_trackster_count)

cluster_poses_simTrackster_mean, cluster_poses_simTrackster_std, cluster_cov_simTrackster_mean, cluster_cov_simTrackster_std, trackster_rel_poses_mean, trackster_rel_poses_std, trackster_count_mean, trackster_count_std = statistics_of_gaussians(cluster_poses_simTrackster, cluster_cov_simTrackster, trackster_poses, trackster_rel_poses, trackster_count, cluster=21)
pu_cluster_poses_simTrackster_mean, pu_cluster_poses_simTrackster_std, pu_cluster_cov_simTrackster_mean, pu_cluster_cov_simTrackster_std, pu_trackster_rel_poses_mean, pu_trackster_rel_poses_std, pu_trackster_count_mean, pu_trackster_count_std = statistics_of_gaussians(pu_cluster_poses_simTrackster, pu_cluster_cov_simTrackster, pu_trackster_poses, pu_trackster_rel_poses, pu_trackster_count, cluster=200)

res = {}
res["signal_simTrackster"] = {"count_mean": np.mean(count_signal_simTrackster, axis=0), 
                              "count_std": np.std(count_signal_simTrackster, axis=0), 
                              "poses_mean": cluster_poses_simTrackster_mean, 
                              "poses_std": cluster_poses_simTrackster_std, 
                              "cov_mean": cluster_cov_simTrackster_mean, 
                              "cov_std": cluster_cov_simTrackster_std, 
                              "per_cluster_count_mean": np.mean(cluster_counts_simTrackster, axis=0), 
                              "per_cluster_count_std": np.std(cluster_counts_simTrackster, axis=0),
                              "trackster_rel_poses_mean": trackster_rel_poses_mean, 
                              "trackster_rel_poses_std": trackster_rel_poses_std,
                              "trackster_count_mean": trackster_count_mean, 
                              "trackster_count_std": trackster_count_std
                              }

res["pu_simTrackster"] = {"count_mean": np.mean(count_pu_simTrackster, axis=0), 
                              "count_std": np.std(count_pu_simTrackster, axis=0), 
                              "poses_mean": pu_cluster_poses_simTrackster_mean, 
                              "poses_std": pu_cluster_poses_simTrackster_std, 
                              "cov_mean": pu_cluster_cov_simTrackster_mean, 
                              "cov_std": pu_cluster_cov_simTrackster_std, 
                              "per_cluster_count_mean": np.mean(pu_cluster_counts_simTrackster, axis=0), 
                              "per_cluster_count_std": np.std(pu_cluster_counts_simTrackster, axis=0), 
                              "trackster_rel_poses_mean": pu_trackster_rel_poses_mean, 
                              "trackster_rel_poses_std": pu_trackster_rel_poses_std, 
                              "trackster_count_mean": pu_trackster_count_mean, 
                              "trackster_count_std": pu_trackster_count_std
                              }
np.savez(osp.join(output_folder, "simTrackster.npz"), **res)
