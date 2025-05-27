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

        self.processed_dir = root

        self._process()

        data_dict, comp_dict = self._load()
        self.data_dict = data_dict
        self.comp_dict = comp_dict
        self.count = len(self.data_dict)

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
        return glob(f"{self.processed_dir}/comp_*.pt")

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
        event = 0

        files = glob(f"{self.path}/*.pt")
        for path in files:
            sample = torch.load(path, weights_only=False)

            if (sample == None):
                continue

            if isinstance(sample, str):
                continue

            visited = []
            components = find_connected_components(sample.edge_index, sample.x.shape[0])
            cluster = torch.unique(sample.cluster)
            cluster = cluster[cluster >= 0]
            for comp_cnt, component in enumerate(components):
                component = np.array(component, dtype=int)

                # Root node with max "raw_energy"
                root = component[torch.argmax(sample.x[component, self.node_feature_dict["raw_energy"]]).item()].item()
                root_cluster = sample.cluster[root].item()
                sample_seq = converter.y2seq(root, component, np.array(sample.cluster))

                data = {}
                data["x"] = sample.x[component]
                data["seq"] = sample_seq
                data["root"] = root
                data["name"] = f"{event}_{comp_cnt}"
                data["inputs"] = {}

                if (component.shape[0] > 1):
                    root_group = component[sample.cluster[component] == root_cluster]
                else:
                    root_group = component

                for i in range(sample_seq.shape[0]-3):
                    vals = {}
                    seq = self.converter.subseq(sample_seq, seq_length=self.input_length+1, index=i-self.input_length+2)
                    vals["input"] = torch.from_numpy(seq[:-1])
                    vals["y"] = torch.from_numpy(seq[1:])

                    if (seq[-2] > self.converter.word2index[";"]):
                        visited.append(self.converter.index2word[seq[-2]])
                        group = np.setdiff1d(root_group, visited)
                        group = np.array(list(map(self.converter.word2index.get, group)))

                        if (group.shape[0] == 0):
                            group = np.array([seq[-1]])
                    else:
                        cluster = cluster[cluster != root_cluster]
                        if (cluster.shape[0] == 0 or seq[-1] == self.converter.word2index["<EOS>"]):
                            group = np.array([self.converter.word2index["<EOS>"]])
                        else:
                            root_cluster = cluster[0].item()
                            root_group = component[sample.cluster[component] == root_cluster]
                            group = np.array(list(map(self.converter.word2index.get, root_group)))

                    group = torch.from_numpy(group)
                    ys = torch.cat([torch.unsqueeze(vals["y"], dim=0)] * group.shape[0], dim=0).long()
                    ys[:, -1] = group
                    vals["options"] = ys
                    vals["component"] = f"{event}_{comp_cnt}"

                    data["inputs"][idx] = vals
                    idx += 1
                torch.save(data, osp.join(self.processed_dir, f'comp_{event}_{comp_cnt}.pt'))

            event += 1

    def _load(self):
        data_dict = {}
        comp_dict = {}

        for paths in self.processed_paths:
            data = torch.load(paths, weights_only=False)

            for key, value in data["inputs"].items():
                data_dict[key] = value

            data.pop("inputs", None)
            comp_dict[data["name"]] = data

        return data_dict, comp_dict

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        data = self.data_dict[idx]
        X = self.comp_dict[data["component"]]["x"]
        X = F.pad(X, pad=(0, 0, self.max_nodes - X.shape[0], 0), value=self.converter.word2index["<PAD>"])

        if (self.filter):
            X = X[:, list(map(self.node_feature_dict.get, self.model_feature_keys))]
            X[:, 0] = (X[:, 0] - self.max_nodes/2)/self.max_nodes
            X[:, 1] = X[:, 1]/(2*np.pi)
            X[:, 2] = X[:, 2]/(2*np.pi)

        Y = data["input"]
        y = data["y"]

        ys = data["options"]
        ys = F.pad(ys, pad=(0, 0, self.max_nodes - ys.shape[0], 0), value=self.converter.word2index["<PAD>"])
        return X.float(), Y.long(), y.long(), ys.long()
