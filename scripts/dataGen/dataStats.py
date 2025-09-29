import numpy as np
import cupy as cp

import sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import tracksterLinker
from tracksterLinker.utils.graphUtils import *
from tracksterLinker.datasets.NeoGNNDataset import *


def statistics_of_gaussians(centers, covariances=None, cluster=10):
    kmeans = KMeans(n_clusters=5, random_state=0).fit(centers)
    labels = kmeans.labels_

    cluster_center_means = []
    cluster_center_stds = []
    cluster_cov_means = []
    cluster_cov_stds = []

    for cluster_id in range(kmeans.n_clusters):
        cluster_centers = centers[labels == cluster_id]

        if covariances is not None:
            cluster_covs = covariances[labels == cluster_id]
            cov_mean = cluster_covs.mean(axis=0)
            cov_std = cluster_covs.std(axis=0)

            cluster_cov_means.append(cov_mean)
            cluster_cov_stds.append(cov_std)
        
        # mean & std of positions
        center_mean = cluster_centers.mean(axis=0)
        center_std = cluster_centers.std(axis=0)
        
        cluster_center_means.append(center_mean)
        cluster_center_stds.append(center_std)

    cluster_center_means = np.array(cluster_center_means)
    cluster_center_stds = np.array(cluster_center_stds)

    if covariances is not None:
        cluster_cov_means = np.array(cluster_cov_means)
        cluster_cov_stds = np.array(cluster_cov_stds)
        return cluster_center_means, cluster_center_stds, cluster_cov_means, cluster_cov_stds
    return cluster_cemter_means, cluster_center_stds


base_folder = "/home/czeh"
hist_folder = osp.join(base_folder, "new_graph_histo")
output_folder = osp.join(base_folder, "dummy_data/data_stats")
os.makedirs(output_folder, exist_ok=True)

files = glob(f"{hist_folder}/test/*.root")

interest_features = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "num_LCs", "raw_energy", "z_min", "z_max", "LC_density"]

count_signal_simTrackster = []
count_pu_simTrackster = []
cluster_poses_simTrackster = []
pu_cluster_poses_simTrackster = []
cluster_counts_simTrackster = []
pu_cluster_counts_simTrackster = []
cluster_cov_simTrackster = []
pu_cluster_cov_simTrackster = []

for file in files:
    file = uproot.open(file)

    allGNNtrain = load_branch_with_highest_cycle(file, 'ticlDumperGNN/GNNTraining')
    allGNNtrain_array = allGNNtrain.arrays()
    print(allGNNtrain_array.fields)

    cnt = 0    
    for event in allGNNtrain_array:
        nTracksters = len(event["node_barycenter_x"])
        features = cp.stack([ak.to_cupy(event[f"node_{field}"]) for field in interest_features], axis=1)
        signal_poses = np.zeros((len(event["simTrackster_isPU"]), 3))
        pu_poses = np.zeros((np.sum(event["simTrackster_isPU"]), 3))
        sim_i = 0
        pu_i = 0
        for simTr in range(len(event["simTrackster_isPU"])):
            trackster = ak.to_numpy(features[event["node_match_idx"] == simTr]) 

            if (trackster.shape[0] == 0):
                continue

            energy = trackster[:, 9]
            sum_energy = np.tile(energy, (trackster.shape[1], 1))
            sim_features = np.sum(trackster * sum_energy.T, axis=0) / np.sum(energy)

            if(event["simTrackster_isPU"][simTr]):
                signal_poses[sim_i] = sim_features[:3] 
                sim_i +=1
            else:
                pu_poses[pu_i] = sim_features[:3] 
                pu_i += 1

        signal_poses = signal_poses[:sim_i]
        pu_poses = pu_poses[:pu_i]
        
        signal_kmeans = GaussianMixture(n_components=min(20, sim_i)).fit(signal_poses)
        pu_kmeans = GaussianMixture(n_components=min(100, pu_i)).fit(pu_poses)

        cluster_poses_simTrackster.append(signal_kmeans.means_)
        cluster_cov_simTrackster.append(signal_kmeans.covariances_)
        pu_cluster_poses_simTrackster.append(pu_kmeans.means_)
        pu_cluster_cov_simTrackster.append(pu_kmeans.covariances_)
        cluster_counts_simTrackster.append(np.bincount(signal_kmeans.predict(signal_poses)))
        pu_cluster_counts_simTrackster.append(np.bincount(pu_kmeans.predict(pu_poses)))
        
        if (cnt == 100):
            break
        cnt += 1
    break

count_signal_simTrackster = np.array(count_signal_simTrackster)
count_pu_simTrackster = np.array(count_signal_simTrackster)
cluster_poses_simTrackster = np.array(cluster_poses_simTrackster)
cluster_cov_simTrackster = np.array(cluster_cov_simTrackster)
pu_cluster_poses_simTrackster = np.array(cluster_poses_simTrackster)
pu_cluster_cov_simTrackster = np.array(cluster_cov_simTrackster)
cluster_counts_simTrackster = np.array(cluster_counts_simTrackster)
pu_cluster_counts_simTrackster = np.array(cluster_counts_simTrackster)

cluster_poses_simTrackster = cluster_poses_simTrackster.reshape(cluster_poses_simTrackster.shape[0] * cluster_poses_simTrackster.shape[1], 3)
cluster_cov_simTrackster = cluster_cov_simTrackster.reshape(cluster_cov_simTrackster.shape[0] * cluster_cov_simTrackster.shape[1], 3, 3)
pu_cluster_poses_simTrackster = pu_cluster_poses_simTrackster.reshape(pu_cluster_poses_simTrackster.shape[0] * pu_cluster_poses_simTrackster.shape[1], 3)
pu_cluster_cov_simTrackster = pu_cluster_cov_simTrackster.reshape(pu_cluster_cov_simTrackster.shape[0] * pu_cluster_cov_simTrackster.shape[1], 3, 3)

cluster_poses_simTrackster_mean, cluster_poses_simTrackster_std, cluster_cov_simTrackster_mean, cluster_cov_simTrackster_std = statistics_of_gaussians(cluster_poses_simTrackster, cluster_cov_simTrackster, cluster=20)
pu_cluster_poses_simTrackster_mean, pu_cluster_poses_simTrackster_std, pu_cluster_cov_simTrackster_mean, pu_cluster_cov_simTrackster_std = statistics_of_gaussians(pu_cluster_poses_simTrackster, pu_cluster_cov_simTrackster, cluster=100)

res = {}
res["signal_simTrackster"] = {"count_mean": np.mean(count_signal_simTrackster, axis=0), 
                              "count_std": np.std(count_signal_simTrackster, axis=0), 
                              "poses_mean": cluster_poses_simTrackster_mean, 
                              "poses_std": cluster_poses_simTrackster_std, 
                              "cov_mean": cluster_cov_simTrackster_mean, 
                              "cov_std": cluster_cov_simTrackster_std, 
                              "per_cluster_count_mean": np.mean(cluster_counts_simTrackster, axis=0), 
                              "per_cluster_count_std": np.std(cluster_counts_simTrackster, axis=0)
                              }

res["pu_simTrackster"] = {"count_mean": np.mean(count_pu_simTrackster, axis=0), 
                              "count_std": np.std(count_pu_simTrackster, axis=0), 
                              "poses_mean": pu_cluster_poses_simTrackster_mean, 
                              "poses_std": pu_cluster_poses_simTrackster_std, 
                              "cov_mean": pu_cluster_cov_simTrackster_mean, 
                              "cov_std": pu_cluster_cov_simTrackster_std, 
                              "per_cluster_count_mean": np.mean(pu_cluster_counts_simTrackster, axis=0), 
                              "per_cluster_count_std": np.std(pu_cluster_counts_simTrackster, axis=0)
                              }
np.savez(osp.join(output_folder, "simTrackster.npz"), **res)
print(res)
