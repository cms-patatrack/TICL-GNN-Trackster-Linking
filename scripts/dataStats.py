import numpy as np
import cupy as cp

import sklearn
from sklearn.cluster import KMeans

import tracksterLinker
from tracksterLinker.utils.graphUtils import *
from tracksterLinker.datasets.NeoGNNDataset import *

base_folder = "/home/czeh"
hist_folder = osp.join(base_folder, "new_graph_histo")

files = glob(f"{hist_folder}/train/*.root")

interest_features = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "num_LCs", "raw_energy", "num_hits", "z_min", "z_max", "LC_density"]


count_signal_simTrackster = []
count_pu_simTrackster = []
cluster_poses_simTrackster = []
pu_cluster_poses_simTrackster = []
cluster_counts_simTrackster = []
pu_cluster_counts_simTrackster = []

for file in files:
    file = uproot.open(file)

    if (len(file.keys()) == 0):
        print("SKIP")
        continue

    allGNNtrain = load_branch_with_highest_cycle(file, 'ticlDumperGNN/GNNTraining')
    allGNNtrain_array = allGNNtrain.arrays()
    # print(allGNNtrain_array.fields)

    cnt = 0
    for event in allGNNtrain_array:
        nTracksters = len(event["node_barycenter_x"])
        print(f"Num Trackster: {nTracksters}, num sim trackster: {len(event['simTrackster_isPU'])}")
        features = cp.stack([ak.to_cupy(event[f"node_{field}"]) for field in interest_features], axis=1)
        print(cp.sum(features[:, 10]))
        signal_poses = np.zeros((len(event["simTrackster_isPU"]), 3))
        pu_poses = np.zeros((np.sum(event["simTrackster_isPU"]), 3))
        sim_i = 0
        pu_i = 0
        for simTr in range(len(event["simTrackster_isPU"])):
            trackster = ak.to_numpy(features[event["node_match_idx"] == simTr])

            if(event["simTrackster_isPU"][simTr]):
                count_pu_simTrackster.append(trackster.shape[0])

            else:
                count_signal_simTrackster.append(trackster.shape[0])

            if(trackster.shape[0] == 0):
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
        
        signal_kmeans = KMeans(n_clusters=min(20, sim_i), random_state=0, n_init="auto").fit(signal_poses)
        pu_kmeans = KMeans(n_clusters=min(100, pu_i), random_state=0, n_init="auto").fit(pu_poses)

        cluster_poses_simTrackster.append(signal_kmeans.cluster_centers_)
        pu_cluster_poses_simTrackster.append(pu_kmeans.cluster_centers_)
        cluster_counts_simTrackster.append(np.bincount(signal_kmeans.labels_))
        pu_cluster_counts_simTrackster.append(np.bincount(pu_kmeans.labels_))
        
        if cnt == 10:
            break
        cnt += 1
    break

count_signal_simTrackster = np.array(count_signal_simTrackster)
count_pu_simTrackster = np.array(count_signal_simTrackster)
cluster_poses_simTrackster = np.array(cluster_poses_simTrackster)
pu_cluster_poses_simTrackster = np.array(cluster_poses_simTrackster)
cluster_counts_simTrackster = np.array(cluster_counts_simTrackster)
pu_cluster_counts_simTrackster = np.array(cluster_counts_simTrackster)

pu_cluster_poses_simTrackster = pu_cluster_poses_simTrackster.reshape(pu_cluster_poses_simTrackster.shape[0] * pu_cluster_poses_simTrackster.shape[1], 3)
pu_cluster_counts_simTrackster = pu_cluster_counts_simTrackster.reshape(pu_cluster_counts_simTrackster.shape[0] * pu_cluster_counts_simTrackster.shape[1])


res = {}
res["signal_simTrackster"] = {"count_mean": np.mean(count_signal_simTrackster, axis=0), 
                              "count_std": np.std(count_signal_simTrackster, axis=0), 
                              "poses_mean": np.mean(cluster_poses_simTrackster, axis=0), 
                              "poses_std": np.std(cluster_poses_simTrackster, axis=0), 
                              "per_cluster_count_mean": np.mean(cluster_counts_simTrackster, axis=0), 
                              "per_cluster_count_std": np.std(cluster_counts_simTrackster, axis=0)
                              }

res["pu_simTrackster"] = {"count_mean": np.mean(count_pu_simTrackster, axis=0), 
                              "count_std": np.std(count_pu_simTrackster, axis=0), 
                              "poses_mean": np.mean(pu_cluster_poses_simTrackster, axis=0), 
                              "poses_std": np.std(pu_cluster_poses_simTrackster, axis=0), 
                              "per_cluster_count_mean": np.mean(pu_cluster_counts_simTrackster, axis=0), 
                              "per_cluster_count_std": np.std(pu_cluster_counts_simTrackster, axis=0)
                              }
print(res)
