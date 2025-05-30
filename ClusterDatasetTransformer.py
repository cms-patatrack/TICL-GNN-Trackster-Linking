import sys
import os.path as osp
from glob import glob
from operator import itemgetter
from random import randrange

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
    model_feature_keys = np.array(["barycenter_eta", "barycenter_phi", "raw_energy"])

    def __init__(self, root, data_path, input_length, filter=True, output_group=False, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.path = data_path
        self.input_length = input_length
        self.filter = filter
        self.output_group = output_group
        self.dummy_converter = Lang(0)

        self.processed_dir = root
        self.component_dir = osp.join(self.processed_dir, "component")
        self.component_dict_dir = osp.join(self.processed_dir, "component_dict")
        self.sequence_dir = osp.join(self.processed_dir, "sequence")
        self.output_group_dir = osp.join(self.processed_dir, "output_group")

        self._process(device)
        print("Done")

        super().__init__()

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

    def _process(self, device):
        if fs.exists(osp.join(self.processed_dir, "metadata.pt")):  # pragma: no cover
            metadata = torch.load(f"{self.processed_dir}/metadata.pt", weights_only=False)
            self.max_nodes = metadata["max_nodes"]
            self.count = metadata["count"]
            self.data_access = metadata["data_access"]
            self.output_group = self.output_group and metadata["output_group"]
            return

        print('Processing...', file=sys.stderr)

        fs.makedirs(self.processed_dir, exist_ok=True)
        fs.makedirs(self.component_dir, exist_ok=True)
        fs.makedirs(self.component_dict_dir, exist_ok=True)
        fs.makedirs(self.sequence_dir, exist_ok=True)

        if (self.output_group):
            fs.makedirs(self.output_group_dir, exist_ok=True)
        self.process(device)

        print('Done!', file=sys.stderr)

    def process(self, device):
        event = 0
        max_nodes = 0
        idx = 0

        data_access = {}
        files = glob(f"{self.path}/*.pt")
        for path in files:
            print(event)
            sample = torch.load(path, weights_only=False)

            if (sample == None):
                continue

            if isinstance(sample, str):
                continue

            components = find_connected_components(sample.edge_index, sample.x.shape[0])

            for comp_cnt, component in enumerate(components):
                visited = []
                component = np.array(component, dtype=int)

                if (component.shape[0] <= 1):
                    continue

                if (component.shape[0] > max_nodes):
                    max_nodes = component.shape[0]

                cluster = torch.unique(sample.cluster[component])
                cluster = cluster[cluster >= 0]

                converter = Lang(trackster_list=component)

                # Root node with max "raw_energy"
                root = component[torch.argmax(sample.x[component, self.node_feature_dict["raw_energy"]]).item()].item()
                root_cluster = sample.cluster[root].item()
                sample_seq = converter.y2seq(root, component, np.array(sample.cluster))

                data = {}
                data["x"] = sample.x[component].float().to(device)
                data["seq"] = sample_seq
                data["root"] = root
                data["name"] = f"{event}_{comp_cnt}"
                data["lang"] = converter.getTracksterList()
                data["nTrackster"] = component.shape[0]
                data["inputs"] = {}

                if (component.shape[0] > 1):
                    root_group = component[sample.cluster[component] == root_cluster]
                else:
                    root_group = component

                for i in range(sample_seq.shape[0]-3):
                    vals = {}
                    seq = torch.from_numpy(converter.subseq(sample_seq, seq_length=self.input_length+1, index=i-self.input_length+2)).long().to(device)
                    vals["input"] = seq[:-1]
                    vals["y"] = seq[1:]

                    torch.save({"input": seq[:-1], "output": seq[1:]}, osp.join(self.sequence_dir, f'comp_{event}_{comp_cnt}_{i}.pt'))
                    if (self.output_group):
                        last_word = seq[-2].item()
                        if (last_word > converter.word2index[";"]):
                            visited.append(converter.index2word[last_word])
                            group = np.setdiff1d(root_group, visited)
                            group = torch.tensor(list(map(converter.word2index.get, group)))

                            if (group.shape[0] == 0):
                                group = torch.unsqueeze(seq[-1], dim=0)
                        else:
                            cluster = cluster[cluster != root_cluster]
                            if (cluster.shape[0] == 0 or seq[-1].item() == converter.word2index["<EOS>"]):
                                group = torch.tensor([converter.word2index["<EOS>"]])
                            else:
                                root_cluster = cluster[0].item()
                                root_group = component[sample.cluster[component] == root_cluster]
                                group = torch.tensor(list(map(converter.word2index.get, root_group)))

                        ys = torch.cat([torch.unsqueeze(seq[1:], dim=0)] * group.shape[0], dim=0).long()
                        ys[:, -1] = group
                        vals["options"] = ys.long().to(device)

                        torch.save(vals["options"], osp.join(self.output_group_dir, f'comp_{event}_{comp_cnt}_{i}.pt'))

                    data["inputs"][i] = vals
                    data_access[idx] = {"event": event, "component": comp_cnt, "step": i}
                    idx += 1

                data["nInputs"] = i
                torch.save(data, osp.join(self.component_dict_dir, f'comp_{event}_{comp_cnt}.pt'))

                torch.save(sample.x[component].float().to(device), osp.join(self.component_dir, f'comp_{event}_{comp_cnt}.pt'))

            event += 1

        self.max_nodes = max_nodes
        self.count = event
        self.data_access = data_access
        metadata = {"max_nodes": max_nodes, "count": event, "output_group": self.output_group, "data_access": data_access}
        torch.save(metadata, osp.join(self.processed_dir, f'metadata.pt'))

    def __len__(self):
        return self.count

    def get(self, event):
        files = glob(f"{self.component_dict_dir}/comp_{event}_*.pt")
        if len(files) == 0:
            return

        components = []
        for file in files:
            comp = torch.load(file, weights_only=False)
            components.append(comp)
        return components

    def __getitem__(self, idx):
        vals = self.data_access[idx]
        X = torch.load(osp.join(self.component_dir, f'comp_{vals["event"]}_{vals["component"]}.pt'), weights_only=True)
        X = F.pad(X, pad=(0, 0, self.max_nodes - X.shape[0], 0), value=self.dummy_converter.word2index["<PAD>"])

        if (self.filter):
            X = X[:, list(map(self.node_feature_dict.get, self.model_feature_keys))]

        seq_data = torch.load(osp.join(self.sequence_dir, f'comp_{vals["event"]}_{vals["component"]}_{vals["step"]}.pt'), weights_only=True)
        Y = seq_data["input"]
        y = seq_data["output"]

        if (self.output_group):
            ys = torch.load(osp.join(sself.output_group_dir, f'comp_{vals["event"]}_{vals["component"]}_{vals["step"]}.pt'), weights_only=True)
            ys = F.pad(ys, pad=(0, 0, self.max_nodes - ys.shape[0], 0), value=self.dummy_converter.word2index["<PAD>"])
            return X, Y, y, ys

        return X, Y, y
