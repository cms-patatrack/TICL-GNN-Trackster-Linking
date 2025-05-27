import os.path as osp
from glob import glob

import tqdm as tqdm
from itertools import chain

import uproot as uproot
import awkward as ak
import numpy as np
from sklearn.neighbors import KDTree

import torch
from torch_geometric.data import Dataset, Data

from graph_utils import build_ticl_graph


class ClusterDataset(Dataset):
    node_feature_keys = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "EV1", "EV2", "EV3",
                         "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "num_LCs", "num_hits", "raw_energy", "raw_em_energy", "photon_prob", "electron_prob", "muon_prob",
                         "neutral_pion_prob", "charged_hadron_prob", "neutral_hadron_prob", "z_min", "z_max", "LC_density", "trackster_density", "time", "idx"]
    node_feature_dict = {k: v for v, k in enumerate(node_feature_keys)}
    model_feature_keys = np.array(["idx", "barycenter_eta", "barycenter_phi", "raw_energy"])
    # model_feature_keys = np.array([0,  2,  3,  4,  6,  7, 10, 14, 15, 16, 17, 18, 22, 24, 25, 26, 28])

    def __init__(self, root, histo_path, transform=None, test=False, pre_transform=None, pre_filter=None):
        self.test = test
        self.histo_path = histo_path
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return glob(f"{self.raw_dir}/*")

    @property
    def processed_file_names(self):
        return glob(f"{self.processed_dir}/data_*.pt")

    # use this to load the tree if some of file.keys() are duplicates ending with different numbers
    def load_branch_with_highest_cycle(self, file, branch_name):

        # Get all keys in the file
        all_keys = file.keys()

        # Filter keys that match the specified branch name
        matching_keys = [
            key for key in all_keys if key.startswith(branch_name)]

        if not matching_keys:
            raise ValueError(
                f"No branch with name '{branch_name}' found in the file.")

        # Find the key with the highest cycle
        highest_cycle_key = max(
            matching_keys, key=lambda key: int(key.split(";")[1]))

        # Load the branch with the highest cycle
        branch = file[highest_cycle_key]

        return branch

    def download(self):
        if (self.test):
            files = glob(f"{self.histo_path}/test/*.root")
        else:
            files = glob(f"{self.histo_path}/train/*.root")

        for id in range(len(files)):
            file = uproot.open(files[id])

            alltracksters = self.load_branch_with_highest_cycle(file, 'ticlDumper/ticlTrackstersCLUE3DHigh')
            allclusters = self.load_branch_with_highest_cycle(file, 'ticlDumper/clusters')
            allassociations = self.load_branch_with_highest_cycle(file, 'ticlDumper/associations')

            alltracksters_array = alltracksters.arrays()
            NTracksters = alltracksters.arrays().NTracksters

            try:
                allgraph = self.load_branch_with_highest_cycle(file, 'ticlDumper/TICLGraph')
                allgraph_array = allgraph.arrays()
            except:
                allgraph = []
                for i in range(len(NTracksters)):
                    allgraph.append(build_ticl_graph(NTracksters[i], alltracksters_array[i]))
                allgraph_array = ak.Array(allgraph)

            node_feature_keys_before = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x",
                                        "eVector0_y", "eVector0_z",  "EV1", "EV2", "EV3", "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "raw_energy", "raw_em_energy", "time"]
            data = alltracksters.arrays(node_feature_keys_before)

            cluster_number_of_hits = allclusters.arrays().cluster_number_of_hits
            cluster_layer_id = allclusters.arrays().cluster_layer_id
            vertices_indexes = alltracksters.arrays().vertices_indexes
            allassociations_array = allassociations.arrays()

            num_LCs = ak.count(alltracksters_array.vertices_indexes, axis=2)
            data["num_LCs"] = num_LCs
            data["z_min"] = ak.min(alltracksters_array.vertices_z, axis=2)
            data["z_max"] = ak.max(alltracksters_array.vertices_z, axis=2)

            data["vertices"] = ak.concatenate([alltracksters_array["vertices_x"][:, :, :, np.newaxis], alltracksters_array["vertices_y"]
                                              [:, :, :, np.newaxis], alltracksters_array["vertices_z"][:, :, :, np.newaxis]], axis=-1)

            hits = ak.to_list(np.zeros_like(data.num_LCs))
            length = ak.to_list(np.zeros_like(data.num_LCs))

            cluster_hits = cluster_number_of_hits[ak.flatten(vertices_indexes, axis=-1)]
            cluster_layer_ids = cluster_layer_id[ak.flatten(vertices_indexes, axis=-1)]
            vertices_count = ak.count(vertices_indexes, axis=-1)

            for i in range(len(data.num_LCs)):
                hits[i] = ak.sum(ak.unflatten(
                    cluster_hits[i], vertices_count[i]), axis=-1)
                length[i] = (ak.max(ak.unflatten(cluster_layer_ids[i], vertices_count[i]), axis=-1) -
                             ak.min(ak.unflatten(cluster_layer_ids[i], vertices_count[i]), axis=-1)) / 47

            data["num_hits"] = hits
            data["length"] = length

            volume = 2*(3 - 1.5) * (2 * 47)
            data["LC_density"] = data.num_LCs / volume
            data["trackster_density"] = ak.Array(
                np.zeros_like(data.num_LCs)) + NTracksters / volume

            probabilities = alltracksters_array.id_probabilities
            data["photon_prob"] = probabilities[:, :, 0]
            data["electron_prob"] = probabilities[:, :, 1]
            data["muon_prob"] = probabilities[:, :, 2]
            data["neutral_pion_prob"] = probabilities[:, :, 3]
            data["charged_hadron_prob"] = probabilities[:, :, 4]
            data["neutral_hadron_prob"] = probabilities[:, :, 5]

            # TODO: Check if correct calc
            idx = allassociations_array.ticlTrackstersCLUE3DHigh_recoToSim_CP_score < 0.2
            simTracksters = allassociations_array.ticlTrackstersCLUE3DHigh_recoToSim_CP[allassociations_array.ticlTrackstersCLUE3DHigh_recoToSim_CP_score < 0.2]
            emptys = np.full_like(ak.count(allassociations_array.ticlTrackstersCLUE3DHigh_recoToSim_CP, axis=-1), -1)

            data["y"] = ak.flatten(ak.where(
                ak.count(simTracksters, axis=-1) == 1, allassociations_array.ticlTrackstersCLUE3DHigh_recoToSim_CP[idx],
                ak.unflatten(emptys, 1, axis=-1)),
                axis=-1)
            data["shared_e"] = ak.flatten(ak.where(
                ak.count(simTracksters, axis=-1) == 1, allassociations_array.ticlTrackstersCLUE3DHigh_recoToSim_CP_sharedE[idx],
                ak.unflatten(emptys, 1, axis=-1)),
                axis=-1)
            data["score"] = ak.flatten(ak.where(
                ak.count(simTracksters, axis=-1) == 1, allassociations_array.ticlTrackstersCLUE3DHigh_recoToSim_CP_score[idx],
                ak.unflatten(emptys, 1, axis=-1)),
                axis=-1)

            data["inner"] = allgraph_array.inner
            data["outer"] = allgraph_array.outer

            roots = ak.num(allgraph_array.inner, axis=-1)
            data["roots"] = ak.local_index(roots)[roots == 0]
            data["idx"] = ak.local_index(data.barycenter_x)

            torch.save(data, osp.join(self.raw_dir, f'data_id_{id}.pt'))

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            print(raw_path)
            run = torch.load(raw_path, weights_only=False)
            nEvents = len(run)

            for event in range(nEvents):
                nTracksters = len(run[event].barycenter_x)
                if (nTracksters <= 1):
                    continue

                features = np.zeros((nTracksters, len(self.node_feature_keys)))
                for i, key in enumerate(self.node_feature_keys):
                    features[:, i] = ak.to_numpy(run[event][key])

                # Create base graph from geometrical graph
                edges = [[], []]
                for i in range(nTracksters):
                    edges[0].extend([i] * len(run[event].outer[i]))
                    edges[1].extend(run[event].outer[i])

                edges = np.array(edges)
                if (edges.shape[1] < 2):
                    continue

                edge_features = np.zeros((len(edges[0, :]), 7))
                edge_features[:, 0] = np.abs(features[edges[1, :], 16] - features[edges[0, :], 16])
                edge_features[:, 1] = np.abs(features[edges[1, :], 2] - features[edges[0, :], 2])
                edge_features[:, 4] = np.linalg.norm(features[edges[1, :], :2] - features[edges[0, :], :2], axis=1)
                edge_features[:, 5] = np.arccos(np.clip(np.sum(np.multiply(features[edges[1, :], 5:8], features[edges[0, :], 5:8]), axis=1), a_min=-1, a_max=1))
                edge_features[:, 6] = np.abs(features[edges[1, :], 28] - features[edges[0], 28])

                transp = edges.T
                edge_indices = np.zeros((nTracksters, nTracksters, ), dtype=np.int64)

                for i in range(len(edges[0, :])):
                    edge_indices[transp[i, 0], transp[i, 1]] = i

                # for root in range(nTracksters):
                #     tree = KDTree(run.vertices[event, root], leaf_size=2)
                #     num = len(run.vertices[event, root])
                #     for target in range(root, nTracksters):
                #         if (root != target):
                #             dist, _ = tree.query(
                #                 run.vertices[event, target], k=num)
                #             edge_features[edge_indices[root, target], 2] = np.min(dist)
                #             edge_features[edge_indices[root, target], 3] = np.max(dist)

                #             edge_features[edge_indices[target, root], 2] = np.min(
                #                 dist)
                #             edge_features[edge_indices[target, root], 3] = np.max(
                #                 dist)
                #         else:
                #             edge_features[edge_indices[root, target], 2] = 0
                #             edge_features[edge_indices[root, target], 3] = 0

                y = np.zeros(edges.shape[1])
                for i, e in enumerate(edges.T):
                    if (run[event].y[e[0]] != -1 and (run[event].y[e[0]] == run[event].y[e[1]])):
                        y[i] = np.round((1-run[event].score[e[0]]) * run[event].shared_e[e[0]]/run[event].raw_energy[e[0]] +
                                        (1-run[event].score[e[1]]) * run[event].shared_e[e[1]]/run[event].raw_energy[e[1]], 3)/2

                # Read data from `raw_path`.
                data = Data(
                    x=torch.from_numpy(features),
                    num_nodes=nTracksters, edge_index=torch.from_numpy(edges),
                    edges_features=torch.from_numpy(edge_features),
                    y=torch.from_numpy(y),
                    cluster=torch.from_numpy(run[event].y.to_numpy()),
                    roots=torch.from_numpy(run[event].roots.to_numpy()))

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(
                    self.processed_dir, f'data_{idx}.pt'))
                idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data
