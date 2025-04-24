import os.path as osp
from glob import glob

import tqdm as tqdm

import uproot as uproot
import awkward as ak
import numpy as np

import torch
from torch_geometric.data import Dataset, Data


class ClusterDataset(Dataset):
    node_feature_keys = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z",  "EV1", "EV2", "EV3", "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "num_LCs",
                         "num_hits", "raw_energy", "raw_em_energy", "photon_prob", "electron_prob", "muon_prob", "neutral_pion_prob", "charged_hadron_prob", "neutral_hadron_prob", "z_min", "z_max", "LC_density", "trackster_density", "time"]

    def __init__(self, root, histo_path, transform=None, pre_transform=None, pre_filter=None):
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
        files = glob(f"{self.histo_path}/*.root")

        for id in range(len(files)):
            file = uproot.open(files[id])

            alltracksters = self.load_branch_with_highest_cycle(
                file, 'ticlDumper/trackstersCLUE3DHigh')
            allclusters = self.load_branch_with_highest_cycle(
                file, 'ticlDumper/clusters')
            allsuperclustering = self.load_branch_with_highest_cycle(
                file, 'ticlDumper/superclustering')

            node_feature_keys_before = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x",
                                        "eVector0_y", "eVector0_z",  "EV1", "EV2", "EV3", "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "raw_energy", "raw_em_energy", "time"]
            data = alltracksters.arrays(node_feature_keys_before)

            num_LCs = ak.count(alltracksters.arrays().vertices_indexes, axis=2)
            data["num_LCs"] = num_LCs
            data["z_min"] = ak.min(alltracksters.arrays().vertices_z, axis=2)
            data["z_max"] = ak.max(alltracksters.arrays().vertices_z, axis=2)

            hits = ak.to_list(np.zeros_like(data.num_LCs))
            length = ak.to_list(np.zeros_like(data.num_LCs))
            density = ak.to_list(np.zeros_like(data.num_LCs))

            volume = 2*(3 - 1.5) * (2 * 47)
            data["LC_density"] = data.num_LCs / volume

            cluster_number_of_hits = allclusters.arrays().cluster_number_of_hits
            cluster_layer_id = allclusters.arrays().cluster_layer_id
            vertices_indexes = alltracksters.arrays().vertices_indexes
            NTracksters = alltracksters.arrays().NTracksters

            for i in range(len(alltracksters.arrays())):
                for j in range(alltracksters.arrays().NTracksters[i]):
                    hits[i][j] = ak.sum(
                        cluster_number_of_hits[i][vertices_indexes[i][j]])
                    length[i][j] = (ak.max(cluster_layer_id[i][vertices_indexes[i][j]]) -
                                    ak.min(cluster_layer_id[i][vertices_indexes[i][j]])) / 47
                    density[i][j] = NTracksters[i] / volume

            data["num_hits"] = hits
            data["length"] = length
            data["trackster_density"] = density

            data["photon_prob"] = alltracksters.arrays()[
                "id_probabilities"][:, :, 0]
            data["electron_prob"] = alltracksters.arrays()[
                "id_probabilities"][:, :, 1]
            data["muon_prob"] = alltracksters.arrays()[
                "id_probabilities"][:, :, 2]
            data["neutral_pion_prob"] = alltracksters.arrays()[
                "id_probabilities"][:, :, 3]
            data["charged_hadron_prob"] = alltracksters.arrays()[
                "id_probabilities"][:, :, 4]
            data["neutral_hadron_prob"] = alltracksters.arrays()[
                "id_probabilities"][:, :, 5]

            edges = allsuperclustering.arrays().linkedResultTracksters
            data["y"] = [edge[ak.num(edge) > 1] for edge in edges]

            torch.save(data, osp.join(self.raw_dir, f'data_id_{id}.pt'))

            # for event in range(len(NTracksters)):
            #     features = np.zeros(
            #         (NTracksters[event], len(node_feature_keys)))
            #     for i, key in enumerate(node_feature_keys):
            #         features[:, i] = ak.to_numpy(data[event][key])

            #     edges = allsuperclustering.arrays(
            #     )[event].linkedResultTracksters

            #     y = torch.from_numpy(ak.to_numpy(edges[ak.num(edges) > 1]))
            #     torch.save(torch.from_numpy(features),
            #                osp.join(self.raw_dir, f'data_id_{id}_event_{event}.pt'))

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            run = torch.load(raw_path)
            nEvents = len(run)

            for event in range(nEvents):
                nTracksters = len(run[event].barycenter_x)

                features = np.zeros((nTracksters, len(self.node_feature_keys)))
                for i, key in enumerate(self.node_feature_keys):
                    features[:, i] = ak.to_numpy(run[event][key])

                # Create fully connected graph, as sparse graph building not stored anymore
                edges = [[], []]
                for i in range(nTracksters):
                    edges[0].extend([i] * (nTracksters))
                    edges[1].extend(list(range(nTracksters)))
                edges = np.array(edges)

                # Read data from `raw_path`.
                data = Data(x=features, num_nodes=nTracksters, edge_index=torch.from_numpy(
                    edges), y=torch.from_numpy(ak.to_numpy(run[event].y)).t().contiguous())

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
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
