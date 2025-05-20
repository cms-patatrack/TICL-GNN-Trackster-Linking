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

from lang import Lang


class ClusterDataset(Dataset):
    node_feature_keys = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "EV1", "EV2", "EV3",
                         "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "num_LCs", "num_hits", "raw_energy", "raw_em_energy", "photon_prob", "electron_prob", "muon_prob",
                         "neutral_pion_prob", "charged_hadron_prob", "neutral_hadron_prob", "z_min", "z_max", "LC_density", "trackster_density", "time", "idx"]

    model_feature_keys = np.array([0,  2,  3,  4,  6,  7, 10, 14, 15, 16, 17, 18, 22, 24, 25, 26, 28])

    def __init__(self, root, data_path, max_nodes, input_length, neighborhood=4, transform=None, pre_transform=None, pre_filter=None):
        self.path = data_path
        self.max_nodes = max_nodes
        self.input_length = input_length
        self.neighborhood = neighborhood
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
        files = glob(f"{self.path}/*.pt")
        id = 0

        for path in files:
            file = torch.load(path, weights_only=False)
            torch.save(file, osp.join(self.raw_dir, f'data_id_{id}.pt'))
            id += 1

    def build_subgraph(self, graph, root, neighborhood=1):
        neighbors = graph[1][graph[0] == root]

        if (neighborhood == 0):
            return neighbors
        subgraph = np.copy(neighbors)

        for n in neighbors:
            subgraph = np.append(subgraph, self.build_subgraph(graph, n, neighborhood-1))

        return np.unique(subgraph)

    def process(self):
        converter = Lang(self.max_nodes)
        idx = 0
        for raw_path in self.raw_paths:
            sample = torch.load(raw_path, weights_only=False)

            if (sample == None):
                continue

            if isinstance(sample, str):
                continue

            for root in sample.roots:
                root_subgraph = np.append(self.build_subgraph(sample.edge_index, root, self.neighborhood), root)
                root_subgraph = np.array(root_subgraph, dtype=int)

                sample_seq = converter.y2seq(np.array(sample.y_trans[root_subgraph]))
                length = sample_seq.shape[0]-1

                if (length < self.input_length):
                    length += 1

                for i in range(length-1):
                    new_sample = sample.clone()
                    seq = converter.subseq(sample_seq, seq_length=self.input_length+1, index=i-self.input_length+2)
                    new_sample.input = torch.tensor(seq[:-1])
                    new_sample.y_trans = torch.tensor(seq[1:])

                    if (new_sample.y_trans[-2] != 2 and new_sample.y_trans[-2] != 3):
                        subgraph = np.append(self.build_subgraph(new_sample.edge_index, new_sample.y_trans[-2], self.neighborhood), root)
                        subgraph = np.array(subgraph, dtype=int)
                    else:
                        subgraph = root_subgraph

                    new_sample.x = new_sample.x[subgraph]

                    if self.pre_filter is not None and not self.pre_filter(new_sample):
                        continue

                    if self.pre_transform is not None:
                        new_sample = self.pre_transform(new_sample)

                    torch.save(new_sample, osp.join(
                        self.processed_dir, f'data_{idx}.pt'))
                    idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
