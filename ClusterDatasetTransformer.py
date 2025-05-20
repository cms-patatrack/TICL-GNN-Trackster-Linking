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
                         "neutral_pion_prob", "charged_hadron_prob", "neutral_hadron_prob", "z_min", "z_max", "LC_density", "trackster_density", "time"]

    model_feature_keys = np.array([0,  2,  3,  4,  6,  7, 10, 14, 15, 16, 17, 18, 22, 24, 25, 26, 28])

    def __init__(self, root, data_path, max_nodes, input_length, transform=None, pre_transform=None, pre_filter=None):
        self.path = data_path
        self.max_nodes = max_nodes
        self.input_length = input_length
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

    def process(self):
        converter = Lang(self.max_nodes)
        idx = 0
        for raw_path in self.raw_paths:
            sample = torch.load(raw_path, weights_only=False)

            if (sample == None):
                continue
            sample_seq = converter.y2seq(np.array(sample.y_trans))
            length = sample_seq.shape[0]-1

            if (length < self.input_length):
                length += 1

            permuted_seqs = converter.permute_groups(sample_seq)
            for k in range(permuted_seqs.shape[0]):
                seq = permuted_seqs[k]
                for i in range(length):
                    sample.y_trans = converter.subseq(seq, seq_length=self.input_length+1, index=i-self.input_length+1)

                if self.pre_filter is not None and not self.pre_filter(sample):
                    continue

                if self.pre_transform is not None:
                    sample = self.pre_transform(sample)

                torch.save(sample, osp.join(
                    self.processed_dir, f'data_{idx}.pt'))
                idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
