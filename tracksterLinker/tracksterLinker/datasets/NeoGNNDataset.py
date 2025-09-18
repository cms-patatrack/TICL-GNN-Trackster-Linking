import os
import sys
import os.path as osp
from glob import glob

import numpy as np
import cupy as cp

import awkward as ak
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.data import Data

from tracksterLinker.utils.dataUtils import calc_weights, cross_PU, mask_PU 

from collections.abc import Sequence
from typing import Callable

import uproot as uproot

def load_branch_with_highest_cycle(file, branch_name):
    # use this to load the tree if some of file.keys() are duplicates ending with different numbers
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

def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.isfile(f) for f in files])


def to_list(value):
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class NeoGNNDataset(Dataset):
    node_feature_keys = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "EV1", "EV2", "EV3",
                         "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "num_LCs", "num_hits", "raw_energy", "raw_em_energy", "photon_prob", "electron_prob", "muon_prob",
                         "neutral_pion_prob", "charged_hadron_prob", "neutral_hadron_prob", "z_min", "z_max", "LC_density", "trackster_density", "time"]
    node_feature_dict = {k: v for v, k in enumerate(node_feature_keys)}
    model_feature_keys = node_feature_keys 
    edge_feature_keys = ["raw_energy", "barycenter_z", "barycenter_xy", "eigenvector0", "time"]

    def __init__(self, root, histo_path, test=False, edge_scaler=None, node_scaler=None, only_signal=False, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.test = test
        self.root_dir = root
        self.histo_path = histo_path
        self.processed_dir = osp.join(self.root_dir, "processed")
        self.device = device
        self.only_signal = only_signal

        if (node_scaler is None and osp.isfile(osp.join(self.root_dir, "node_scaler.pt"))):
            self.node_scaler = torch.load(osp.join(self.root_dir, "node_scaler.pt"))
        else:
            self.node_scaler = node_scaler

        if (edge_scaler is None and osp.isfile(osp.join(self.root_dir, "edge_scaler.pt"))):
            self.edge_scaler = torch.load(osp.join(self.root_dir, "edge_scaler.pt"))
        else:
            self.edge_scaler = edge_scaler

        self._process()
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
        return glob(f"{self.processed_dir}/data_*.pt")

    def _process(self):
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        if osp.isfile(osp.join(self.root_dir, "DONE")):  # pragma: no cover
            return
        else:
            idx = 0
            if (self.test):
                files = glob(f"{self.histo_path}/test/*.root")
            else:
                files = glob(f"{self.histo_path}/train/*.root")
                self.node_scaler = torch.zeros(len(self.model_feature_keys), device=self.device) 
                self.edge_scaler = torch.zeros(len(self.edge_feature_keys), device=self.device)
            
            for file in files:
                print(file)
                file = uproot.open(file)

                if (len(file.keys()) == 0):
                    print("SKIP")
                    continue

                allGNNtrain = load_branch_with_highest_cycle(file, 'ticlDumperGNN/GNNTraining')
                allGNNtrain_array = allGNNtrain.arrays()

                for event in allGNNtrain_array:
                    nTracksters = len(event["node_barycenter_x"])
                    features = cp.stack([ak.to_cupy(event[f"node_{field}"]) for field in self.model_feature_keys], axis=1)
                    edges = cp.stack([ak.to_cupy(ak.flatten(event[f"edgeIndex_{field}"])) for field in ["out", "in"]], axis=1)
                    edge_features = cp.stack([ak.to_cupy(ak.flatten(event[f"edge_{field}"])) for field in self.edge_feature_keys], axis=1)
                    y = ak.to_cupy(ak.flatten(event["edge_weight"]))
                    isPU = ak.to_cupy(event["simTrackster_isPU"][event["node_match_idx"]])

                    PU_info = cp.stack([cross_PU(isPU, edges), mask_PU(isPU, edges, PU=False), mask_PU(isPU, edges, PU=True)], axis=1)
                    y[(y > 0) & PU_info[:, 0]] = 0
                    
                    if (self.only_signal):
                        y[PU_info[:, 2]] = 0

                    data = Data(
                        x=torch.as_tensor(features, device=self.device).float(),
                        num_nodes=nTracksters, 
                        edge_index=torch.as_tensor(edges, device=self.device).long(),
                        edge_features=torch.as_tensor(edge_features, device=self.device).float(),
                        y=torch.as_tensor(y, device=self.device).float(),
                        isPU=torch.as_tensor(isPU, device=self.device).int(),
                        PU_info=torch.as_tensor(PU_info, device=self.device).bool())
                    torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))

                    idx += 1
                    if (not self.test):
                        self.node_scaler = torch.maximum(self.node_scaler, torch.max(torch.abs(data["x"]), axis=0).values)
                        self.edge_scaler = torch.maximum(self.edge_scaler, torch.max(data["edge_features"], axis=0).values)
                   
            
            if (not self.test):
                torch.save(self.node_scaler, osp.join(self.root_dir, "node_scaler.pt"))
                torch.save(self.edge_scaler, osp.join(self.root_dir, "edge_scaler.pt"))
            torch.save(self.edge_scaler, osp.join(self.root_dir, "DONE"))

    def __len__(self):
        return len(self.processed_file_names) 

    def __iter__(self): 
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        return torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
