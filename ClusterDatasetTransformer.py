import sys
import os.path as osp
from glob import glob
from operator import itemgetter

import tqdm as tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.io import fs
import torch.nn.functional as F

from lang import Lang

from collections.abc import Sequence
from typing import Callable


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([fs.exists(f) for f in files])


def to_list(value):
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class ClusterDataset(Dataset):
    node_feature_keys = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "EV1", "EV2", "EV3",
                         "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "num_LCs", "num_hits", "raw_energy", "raw_em_energy", "photon_prob", "electron_prob", "muon_prob",
                         "neutral_pion_prob", "charged_hadron_prob", "neutral_hadron_prob", "z_min", "z_max", "LC_density", "trackster_density", "time", "idx"]
    node_feature_dict = {k: v for v, k in enumerate(node_feature_keys)}

    # model_feature_keys = np.array([0,  2,  3,  4,  6,  7, 10, 14, 15, 16, 17, 18, 22, 24, 25, 26, 28, 29])
    model_feature_keys = np.array(["idx", "barycenter_eta", "barycenter_phi", "raw_energy"])

    def __init__(self, converter, root, data_path, max_nodes, input_length, neighborhood=1, filter=True):
        self.path = data_path
        self.max_nodes = max_nodes
        self.input_length = input_length
        self.neighborhood = neighborhood
        self.converter = converter
        self.filter = filter

        self.root = root
        self.processed_dir = f"{root}/processed"
        self.raw_dir = f"{root}/raw"

        self._download()
        self._process()

        super().__init__()

    @property
    def raw_file_names(self):
        return glob(f"{self.raw_dir}/*")

    @property
    def raw_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        downloading.
        """
        files = self.raw_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [f for f in to_list(files)]

    @property
    def processed_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        processing.
        """
        files = self.processed_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [f for f in to_list(files)]

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

    def _download(self):
        if files_exist(self.raw_paths):  # pragma: no cover
            return

        fs.makedirs(self.raw_dir, exist_ok=True)
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

    def _process(self):
        if files_exist(self.processed_paths):  # pragma: no cover
            return

        print('Processing...', file=sys.stderr)

        fs.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        print('Done!', file=sys.stderr)

    def process(self):
        converter = Lang(self.max_nodes)
        idx = 0
        for raw_path in self.raw_paths:
            print(raw_path)
            orig_sample = torch.load(raw_path, weights_only=False)

            if (orig_sample == None):
                continue

            if isinstance(orig_sample, str):
                continue

            visited = []

            for root in orig_sample.roots[np.argsort(-orig_sample.x[orig_sample.roots, self.node_feature_dict["raw_energy"]])]:
                root = root.item()
                if (root in visited):
                    continue
                sample = orig_sample.clone()
                sorted_graph = np.array(list(range(sample.x.shape[0])), dtype=int)

                if (sorted_graph.shape[0] > 1):
                    root_group = sorted_graph[sample.cluster[sorted_graph] == sample.cluster[root].item()]
                else:
                    root_group = sorted_graph

                sample_seq = converter.y2seq(root, sorted_graph, np.array(sample.cluster))
                # length = sample_seq.shape[0]-1

                # if (length < self.input_length):
                #     length += 1

                for i in range(sample_seq.shape[0]-2):
                    if files_exist([osp.join(self.processed_dir, f'data_{idx}.pt')]):
                        print(f"{osp.join(self.processed_dir, f'data_{idx}.pt')} exists")
                        idx += 1
                        break

                    new_sample = sample.clone()
                    seq = self.converter.subseq(sample_seq, seq_length=self.input_length, index=i-self.input_length+2)
                    new_sample.input = torch.from_numpy(seq)

                    if (seq[-1] > self.converter.word2index[";"]):
                        new_root = int(self.converter.index2word[seq[-1]])
                        subgraph = np.append(self.build_subgraph(new_sample.edge_index, new_root, self.neighborhood), new_root)
                        subgraph = np.array(subgraph, dtype=int)

                        subgraph = np.setdiff1d(subgraph, visited)
                        visited.append(seq[-1])

                        if (subgraph.shape[0] > 1):
                            new_sample.group = subgraph[new_sample.cluster[subgraph] == new_sample.cluster[new_root].item()]
                            new_sample.group = np.setdiff1d(new_sample.group, visited)

                            if (new_sample.group.shape[0] == 0):
                                new_sample.group = np.array([sample_seq[i+2]])
                        elif (subgraph.shape[0] == 1):
                            new_sample.group = subgraph
                        else:
                            new_sample.group = np.array([sample_seq[i+2]])
                    else:
                        subgraph = sorted_graph
                        new_sample.group = root_group

                    new_sample.group = torch.from_numpy(new_sample.group)
                    ys = torch.cat([torch.unsqueeze(torch.roll(new_sample.input, -1), dim=0)] * new_sample.group.shape[0], dim=0).long()
                    ys[:, -1] = new_sample.group
                    new_sample.y_trans = ys

                    new_sample.x = new_sample.x[subgraph]
                    torch.save(new_sample, osp.join(self.processed_dir, f'data_{idx}.pt'))
                    idx += 1

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        X = data.x
        X = F.pad(X, pad=(0, 0, self.max_nodes - data.x.shape[0], 0), value=self.converter.word2index["<PAD>"])

        if (self.filter):
            X = X[:, list(map(self.node_feature_dict.get, self.model_feature_keys))]
            # X[:, 0] = (X[:, 0] - self.max_nodes/2)/self.max_nodes

        Y = data.input

        ys = data.y_trans
        ys = F.pad(ys, pad=(0, 0, self.max_nodes - ys.shape[0], 0), value=self.converter.word2index["<PAD>"])
        return X.float(), Y.long(), ys.long()
