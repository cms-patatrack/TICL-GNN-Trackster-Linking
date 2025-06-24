import sys
import os.path as osp
from glob import glob

import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from lang import Lang

from collections.abc import Sequence
from typing import Callable


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.isfile(f) for f in files])


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
    model_feature_keys = np.array(["barycenter_eta", "barycenter_phi", "raw_energy"])

    def __init__(self, root, input_length, filter=True, scale=None, output_group=False, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.input_length = input_length
        self.filter = filter
        self.scale = scale

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

    def _process(self, device):
        if osp.isfile(osp.join(self.processed_dir, "metadata.pt")):  # pragma: no cover
            metadata = torch.load(f"{self.processed_dir}/metadata.pt", weights_only=False)
            self.max_nodes = metadata["max_nodes"]
            self.count = metadata["count"]
            self.data_access = metadata["data_access"]
            self.output_group = self.output_group and metadata["output_group"]
            return
        else:
            print('Dataset Folder not complete! Run Builder first.', file=sys.stderr)

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
        component = torch.load(osp.join(self.component_dict_dir, f'comp_{vals["event"]}_{vals["component"]}.pt'), weights_only=False)
        X = component["x"]

        if (self.scale is not None):
            X /= self.scale

        X = F.pad(X, pad=(0, 0, self.max_nodes - X.shape[0], 0), value=self.dummy_converter.word2index["<PAD>"])

        if (self.filter):
            X = X[:, list(map(self.node_feature_dict.get, self.model_feature_keys))]

        seq_data = component["inputs"][vals["step"]]
        Y = seq_data["input"]

        # Remove masking after next data building
        mask = Y > 0
        seq_length = Y[mask].shape[0]
        Y = F.pad(Y[mask], pad=(0, self.input_length - seq_length), value=self.dummy_converter.word2index["<PAD>"])

        y = seq_data["y"]
        y = F.pad(y[mask], pad=(0, self.input_length - seq_length), value=self.dummy_converter.word2index["<PAD>"])

        if (self.output_group):
            ys = torch.load(osp.join(self.output_group_dir, f'comp_{vals["event"]}_{vals["component"]}_{vals["step"]}.pt'), weights_only=True)
            ys = F.pad(ys, pad=(0, 0, self.max_nodes - ys.shape[0], 0), value=self.dummy_converter.word2index["<PAD>"])
            return X, Y, y, ys

        return X, Y, y
