import os.path as osp
from glob import glob

import awkward as ak
import numpy as np
import cupy as cp

import joblib
from tqdm import tqdm

import torch
from torch_geometric.data import Dataset, Data

from tracksterLinker.utils.graphUtils import build_ticl_graph
from tracksterLinker.utils.dataUtils import *

from concurrent.futures import ProcessPoolExecutor, as_completed


def download_event(id, file, raw_dir):
    data = ak.from_parquet(file)

    allgraph = []
    for event in range(len(data)):
        allgraph.append(build_ticl_graph(len(data[event]["barycenter_x"]), data[event]))
    allgraph_array = ak.Array(allgraph)

    data["inner"] = allgraph_array["inner"]
    data["outer"] = allgraph_array["outer"]

    roots = ak.num(allgraph_array["inner"], axis=-1)
    data["roots"] = ak.local_index(roots)[roots == 0]
    data["idx"] = ak.local_index(data["barycenter_x"])

    torch.save(data, osp.join(raw_dir, f'data_id_{id}.pt'))


def process_event(idx, event, model_feature_keys, node_feature_dict, processed_dir, skeleton_features, device):
    import cupy as cp
    nTracksters = len(event["barycenter_x"])

    # Skip if not multiple tracksters
    if (nTracksters <= 1):
        return None, None

    # build feature list
    features = cp.stack([ak.to_cupy(event[field]) for field in model_feature_keys], axis=1)

    # Create base graph from geometrical graph = [[], []]
    targets = ak.ravel(event.outer)
    sources = ak.local_index(event.outer, axis=0)
    sources = ak.broadcast_arrays(sources, event.outer)[0]
    sources = ak.ravel(sources)

    edges = cp.transpose(cp.stack([ak.to_cupy(targets), ak.to_cupy(sources)]))
    if (edges.shape[0] < 2):
        return None, None

    edge_features = cp.zeros((len(edges[:, 0]), 5), dtype='f')
    edge_features[:, 0] = calc_edge_difference(edges, features, node_feature_dict, key="raw_energy")
    edge_features[:, 1] = calc_edge_difference(edges, features, node_feature_dict, key="barycenter_z")
    edge_features[:, 2] = calc_transverse_plane_separation(edges, features, node_feature_dict)
    edge_features[:, 3] = calc_spatial_compatibility(edges, features, node_feature_dict)
    edge_features[:, 4] = calc_edge_difference(edges, features, node_feature_dict, key="time")

    e_y = ak.to_cupy(event.y)
    y = cp.zeros(edges.shape[0], dtype='i')
    y[e_y[edges[:, 0]] == e_y[edges[:, 1]]] = 1
    y[e_y[edges[:, 0]] != e_y[edges[:, 1]]] = 0
    y[e_y[edges[:, 0]] == -1] = 0
    y[e_y[edges[:, 1]] == -1] = 0

    isPU = ak.to_cupy(event["isPU"])
    PU_info = cp.stack([cross_PU(isPU, edges), mask_PU(isPU, edges, PU=False), mask_PU(isPU, edges, PU=True)], axis=1)

    # Read data from `raw_path`.
    data = Data(
        x=torch.as_tensor(features, device=device).float(),
        num_nodes=nTracksters, 
        edge_index=torch.as_tensor(edges, device=device).long(),
        edge_features=torch.as_tensor(edge_features, device=device).float(),
        y=torch.as_tensor(y, device=device).float(),
        isPU=torch.as_tensor(isPU, device=device).int(),
        PU_info=torch.as_tensor(PU_info, device=device).bool())

    torch.save(data, osp.join(processed_dir, f'data_{idx}.pt'))
    return torch.max(torch.abs(data.x), axis=0).values, torch.max(data.edge_features, axis=0).values


class DummyDataset(Dataset):
    # node_feature_keys = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "num_LCs", "raw_energy", "z_min", "z_max", "LC_density"]
    node_feature_keys = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "EV1", "EV2", "EV3",
                         "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "num_LCs", "num_hits", "raw_energy", "raw_em_energy", "photon_prob", "electron_prob", "muon_prob",
                         "neutral_pion_prob", "charged_hadron_prob", "neutral_hadron_prob", "z_min", "z_max", "LC_density", "trackster_density", "time"]
    node_feature_dict = {k: v for v, k in enumerate(node_feature_keys)}
    model_feature_keys = node_feature_keys

    # Skeleton Features computional intensive -> Turn off if not needed
    def __init__(self, root, histo_path, transform=None, test=False, skeleton_features=False, pre_transform=None, pre_filter=None, edge_scaler=None, node_scaler=None,
                 num_workers=24, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.test = test
        self.skeleton_features = skeleton_features
        self.device = device
        self.num_workers = num_workers

        self.histo_path = histo_path
        self.root_dir = root

        if (node_scaler is None and osp.isfile(osp.join(self.root_dir, "node_scaler.pt"))):
            self.node_scaler = torch.load(osp.join(self.root_dir, "node_scaler.pt"))
        else:
            self.node_scaler = node_scaler

        if (edge_scaler is None and osp.isfile(osp.join(self.root_dir, "edge_scaler.pt"))):
            self.edge_scaler = torch.load(osp.join(self.root_dir, "edge_scaler.pt"))
        else:
            self.edge_scaler = edge_scaler
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return glob(f"{self.raw_dir}/*")

    @property
    def processed_file_names(self):
        return glob(f"{self.processed_dir}/data_*.pt")

    def download(self):
        if (self.test):
            files = glob(f"{self.histo_path}/test/*.parquet")
        else:
            files = glob(f"{self.histo_path}/train/*.parquet")
        print(files)

        with tqdm(total=len(files)) as pbar:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(download_event, id, files[id], self.raw_dir) for id in range(len(files))]

                for future in as_completed(futures):
                    future.result()
                    pbar.update()

    def process(self):
        idx = 0

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for raw_path in tqdm(self.raw_paths):
                run = torch.load(raw_path, weights_only=False)
                nEvents = len(run)
                process_event(0, run[0], self.model_feature_keys, self.node_feature_dict, self.processed_dir, self.skeleton_features, self.device)
                futures = [executor.submit(process_event, idx+event, run[event], self.model_feature_keys, self.node_feature_dict,
                                           self.processed_dir, self.skeleton_features, self.device) for event in range(nEvents)]
                idx += nEvents
                for future in as_completed(futures):
                    max_features, max_edge_features = future.result()
                    if (not self.test and max_features is not None):
                        if self.node_scaler is not None:
                            self.node_scaler = torch.maximum(self.node_scaler, max_features)
                            self.edge_scaler = torch.maximum(self.edge_scaler, max_edge_features)
                        else:
                            self.node_scaler = max_features
                            self.edge_scaler = max_edge_features

        if (not self.test):
            torch.save(self.node_scaler, osp.join(self.root_dir, "node_scaler.pt"))
            torch.save(self.edge_scaler, osp.join(self.root_dir, "edge_scaler.pt"))

        idx = 0
        for i, file in tqdm(enumerate(self.processed_file_names), desc="Fixing holes"):
            sample = torch.load(file, weights_only=False)

            if (i != idx):
                os.remove(file)
            torch.save(sample, osp.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data
