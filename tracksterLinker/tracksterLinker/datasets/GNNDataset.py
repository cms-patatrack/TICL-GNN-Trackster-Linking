import os.path as osp
from glob import glob

import uproot as uproot
import awkward as ak
import cupy as cp
import numpy as np

from sklearn.preprocessing import MaxAbsScaler
import joblib
from tqdm import tqdm

import torch
from torch_geometric.data import Dataset, Data

from tracksterLinker.utils.graphUtils import build_ticl_graph
from tracksterLinker.utils.dataUtils import *


class GNNDataset(Dataset):
    node_feature_keys = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "EV1", "EV2", "EV3",
                         "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "num_LCs", "num_hits", "raw_energy", "raw_em_energy", "photon_prob", "electron_prob", "muon_prob",
                         "neutral_pion_prob", "charged_hadron_prob", "neutral_hadron_prob", "z_min", "z_max", "LC_density", "trackster_density", "time", "idx"]
    node_feature_dict = {k: v for v, k in enumerate(node_feature_keys)}
    model_feature_keys = ["idx", "barycenter_eta", "barycenter_phi", "raw_energy"]

    # Skeleton Features computional intensive -> Turn off if not needed
    def __init__(self, root, histo_path, transform=None, test=False, skeleton_features=False, pre_transform=None, pre_filter=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.test = test
        self.skeleton_features = skeleton_features
        self.device = device

        self.histo_path = histo_path
        self.root_dir = root
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
        highest_cycle_key = max(matching_keys, key=lambda key: int(key.split(";")[1]))

        # Load the branch with the highest cycle
        branch = file[highest_cycle_key]

        return branch

    def download(self):
        if (self.test):
            files = glob(f"{self.histo_path}/test/*.root")
        else:
            files = glob(f"{self.histo_path}/train/*.root")

        for id in tqdm(range(len(files))):
            file = uproot.open(files[id])

            alltracksters = self.load_branch_with_highest_cycle(file, 'ticlDumper/ticlTrackstersCLUE3DHigh')
            allclusters = self.load_branch_with_highest_cycle(file, 'ticlDumper/clusters')
            allassociations = self.load_branch_with_highest_cycle(file, 'ticlDumper/associations')

            alltracksters_array = alltracksters.arrays()
            allclusters_array = allclusters.arrays()
            allassociations_array = allassociations.arrays()
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

            # conatenate all axes of vertices
            data["vertices"] = ak.concatenate([alltracksters_array["vertices_x"][:, :, :, cp.newaxis], alltracksters_array["vertices_y"]
                                              [:, :, :, cp.newaxis], alltracksters_array["vertices_z"][:, :, :, cp.newaxis]], axis=-1)

            data["num_LCs"], data["num_hits"], data["length"] = calc_trackster_size(alltracksters_array, allclusters_array)
            data["z_min"] = ak.min(alltracksters_array["vertices_z"], axis=2)
            data["z_max"] = ak.max(alltracksters_array["vertices_z"], axis=2)

            data["LC_density"] = calc_LC_density(data["num_LCs"])
            # trackster density per event -> every trackster has same value
            data["trackster_density"] = ak.Array(np.zeros_like(data["num_LCs"])) + calc_trackster_density(NTracksters)

            probabilities = alltracksters_array["id_probabilities"]
            data["photon_prob"] = probabilities[:, :, 0]
            data["electron_prob"] = probabilities[:, :, 1]
            data["muon_prob"] = probabilities[:, :, 2]
            data["neutral_pion_prob"] = probabilities[:, :, 3]
            data["charged_hadron_prob"] = probabilities[:, :, 4]
            data["neutral_hadron_prob"] = probabilities[:, :, 5]

            data["y"], data["shared_e"], data["score"] = calc_reco_2_sim_trackster_fit(
                allassociations_array["ticlTrackstersCLUE3DHigh_recoToSim_CP"],
                allassociations_array["ticlTrackstersCLUE3DHigh_recoToSim_CP_score"],
                allassociations_array["ticlTrackstersCLUE3DHigh_recoToSim_CP_sharedE"])
            data["inner"] = allgraph_array["inner"]
            data["outer"] = allgraph_array["outer"]

            roots = ak.num(allgraph_array["inner"], axis=-1)
            data["roots"] = ak.local_index(roots)[roots == 0]
            data["idx"] = ak.local_index(data["barycenter_x"])

            torch.save(data, osp.join(self.raw_dir, f'data_id_{id}.pt'))

    def process(self):
        idx = 0
        self.scaler = MaxAbsScaler()
        for raw_path in tqdm(self.raw_paths):
            print(raw_path)
            run = ak.to_backend(torch.load(raw_path, weights_only=False), "cuda")
            nEvents = len(run)

            for event in range(nEvents):
                nTracksters = len(run[event]["barycenter_x"])

                # Skip if not multiple tracksters
                if (nTracksters <= 1):
                    continue

                # build feature list
                features = cp.zeros((nTracksters, len(self.node_feature_keys)), dtype='f')
                for i, key in enumerate(self.node_feature_keys):
                    features[:, i] = ak.to_cupy(run[event][key])

                # Fit a normalization scaler on training data
                self.scaler.partial_fit(features.get())

                # Create base graph from geometrical graph
                edges = [[], []]
                for i in range(nTracksters):
                    edges[0].extend([i] * len(run[event].outer[i]))
                    edges[1].extend(ak.to_list(run[event].outer[i]))

                edges = cp.array(edges)
                if (edges.shape[1] < 2):
                    continue

                if self.skeleton_features:
                    edge_features = cp.zeros((len(edges[0, :]), 7), dtype='f')

                    edge_features[:, 5], edge_features[:, 6] = calc_min_max_skeleton_dist(nTracksters, edges, run.vertices[event])
                else:
                    edge_features = cp.zeros((len(edges[0, :]), 5), dtype='f')

                edge_features[:, 0] = calc_edge_difference(edges, features, self.node_feature_dict, key="raw_energy")
                edge_features[:, 1] = calc_edge_difference(edges, features, self.node_feature_dict, key="barycenter_z")
                edge_features[:, 2] = calc_transverse_plane_separation(edges, features, self.node_feature_dict)
                edge_features[:, 3] = calc_spatial_compatibility(edges, features, self.node_feature_dict)
                edge_features[:, 4] = calc_edge_difference(edges, features, self.node_feature_dict, key="time")

                y = cp.zeros(edges.shape[1], dtype='f')
                for i, e in enumerate(edges.T):
                    calc_group_score(run[event].y[e], run[event].score[e], run[event].shared_e[e], run[event].raw_energy[e])

                # Read data from `raw_path`.
                data = Data(
                    x=torch.utils.dlpack.from_dlpack(features.toDlpack()),
                    num_nodes=nTracksters, edge_index=torch.utils.dlpack.from_dlpack(edges.toDlpack()),
                    edges_features=torch.utils.dlpack.from_dlpack(edge_features.toDlpack()),
                    y=torch.utils.dlpack.from_dlpack(y.toDlpack()),
                    cluster=ak.to_torch(run[event].y),
                    roots=ak.to_torch(run[event].roots))

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1

        if (not self.test):
            joblib.dump(self.scaler, osp.join(self.root_dir, "scaler.joblib"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data
