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
from graph_utils import find_connected_components

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
            components = find_connected_components(orig_sample.edge_index, orig_sample.x.shape[0])
            cluster = torch.unique(orig_sample.cluster)
            cluster = cluster[cluster >= 0]
            for component in components:
                component = np.array(component, dtype=int)

                # Root node with max "raw_energy"
                root = torch.argmax(orig_sample.x[component, self.node_feature_dict["raw_energy"]]).item()
                root_cluster = orig_sample.cluster[root].item()

                if (component.shape[0] > 1):
                    root_group = component[orig_sample.cluster[component] == root_cluster]
                else:
                    root_group = component

                sample_seq = converter.y2seq(root, component, np.array(orig_sample.cluster))

                for i in range(sample_seq.shape[0]-3):
                    sample = orig_sample.clone()
                    seq = self.converter.subseq(sample_seq, seq_length=self.input_length+1, index=i-self.input_length+2)
                    sample.input = torch.from_numpy(seq[:-1])
                    sample.y = torch.from_numpy(seq[1:])

                    if (seq[-2] > self.converter.word2index[";"]):
                        # new_root = int(self.converter.index2word[seq[-2]])
                        # subgraph = np.append(self.build_subgraph(sample.edge_index, new_root, self.neighborhood), new_root)
                        # subgraph = np.array(subgraph, dtype=int)

                        # subgraph = np.setdiff1d(subgraph, visited)
                        visited.append(self.converter.index2word[seq[-2]])

                        # if (subgraph.shape[0] > 1):
                        # sample.group = subgraph[sample.cluster[subgraph] == sample.cluster[new_root].item()]
                        # sample.group = np.setdiff1d(sample.group, visited)
                        group = np.setdiff1d(root_group, visited)
                        group = np.array(list(map(self.converter.word2index.get, group)))

                        if (group.shape[0] == 0):
                            print("Group is null")
                            group = np.array([seq[-1]])
                        # elif (subgraph.shape[0] == 1):
                        #     sample.group = subgraph
                        # else:
                        #     sample.group = np.array([sample_seq[i+2]])
                    else:
                        # subgraph = component
                        print("in else")
                        cluster = cluster[cluster != root_cluster]
                        if (seq[-1] == self.converter.word2index["<EOS>"]):
                            print("Stop")
                            group = np.array([self.converter.word2index["<EOS>"]])
                        else:
                            root_cluster = cluster[0].item()
                            print(root_cluster)
                            root_group = component[sample.cluster[component] == root_cluster]
                            print(root_group)
                            group = np.array(list(map(self.converter.word2index.get, root_group)))

                    sample.group = torch.from_numpy(group)
                    print(sample.y)
                    print(sample.group)
                    ys = torch.cat([torch.unsqueeze(sample.y, dim=0)] * sample.group.shape[0], dim=0).long()
                    ys[:, -1] = sample.group
                    sample.ys = ys

                    sample.x = sample.x[component]
                    torch.save(sample, osp.join(self.processed_dir, f'data_{idx}.pt'))
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
        y = data.y

        ys = data.ys
        ys = F.pad(ys, pad=(0, 0, self.max_nodes - ys.shape[0], 0), value=self.converter.word2index["<PAD>"])
        return X.float(), Y.long(), y.long(), ys.long()
