import os.path as osp
import os
from glob import glob

import uproot as uproot
import awkward as ak
import numpy as np

from tqdm import tqdm

import torch
from torch_geometric.data import Dataset, Data

from tracksterLinker.utils.graphUtils import build_ticl_graph
from tracksterLinker.utils.dataUtils import *

from concurrent.futures import ProcessPoolExecutor, as_completed


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


def download_event(id, file, raw_dir):
    file = uproot.open(file)

    alltracksters = load_branch_with_highest_cycle(file, 'ticlDumper/ticlTrackstersCLUE3DHigh')
    allclusters = load_branch_with_highest_cycle(file, 'ticlDumper/clusters')
    allassociations = load_branch_with_highest_cycle(file, 'ticlDumper/associations')

    alltracksters_array = alltracksters.arrays()
    allclusters_array = allclusters.arrays()
    allassociations_array = allassociations.arrays()
    NTracksters = alltracksters_array.NTracksters

    try:
        allgraph = load_branch_with_highest_cycle(file, 'ticlDumper/TICLGraph')
        allgraph_array = allgraph.arrays()
    except:
        allgraph = []
        for i in range(len(NTracksters)):
            allgraph.append(build_ticl_graph(NTracksters[i], alltracksters_array[i]))
        allgraph_array = ak.Array(allgraph)

    node_feature_keys_before = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x",
                                "eVector0_y", "eVector0_z",  "EV1", "EV2", "EV3", "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "raw_energy", "raw_em_energy", "time"]
    data = alltracksters.arrays(node_feature_keys_before)
    if "isPU" in alltracksters_array.fields:
        data["isPU"] = alltracksters_array["isPU"]

    data["vertices"] = ak.concatenate([alltracksters_array["vertices_x"][:, :, :, np.newaxis], alltracksters_array["vertices_y"]
                                       [:, :, :, np.newaxis], alltracksters_array["vertices_z"][:, :, :, np.newaxis]], axis=-1)

    data["num_LCs"], data["num_hits"], data["length"] = calc_trackster_size(alltracksters_array, allclusters_array)
    data["z_min"] = ak.min(alltracksters_array["vertices_z"], axis=2)
    data["z_max"] = ak.max(alltracksters_array["vertices_z"], axis=2)

    data["LC_density"] = calc_LC_density(data["num_LCs"])
    # trackster density per event -> every trackster has same value
    data["trackster_density"] = ak.Array(np.zeros_like(data["num_LCs"])) + calc_trackster_density(NTracksters)

    probabilities = alltracksters_array["id_probabilities"]
    data["photon_prob"] = probabilities[:, :, 0]
    data["electron_prob"] = probabilities[:, :, 1]
    data["muon_prob"] = probabilities[:, :, 2]
    data["neutral_pion_prob"] = probabilities[:, :, 3]
    data["charged_hadron_prob"] = probabilities[:, :, 4]
    data["neutral_hadron_prob"] = probabilities[:, :, 5]

    data["y"], data["score"], data["shared_e"] = calc_reco_2_sim_trackster_fit(
        allassociations_array["ticlTrackstersCLUE3DHigh_recoToSim_CP"],
        allassociations_array["ticlTrackstersCLUE3DHigh_recoToSim_CP_score"],
        allassociations_array["ticlTrackstersCLUE3DHigh_recoToSim_CP_sharedE"])
    data["inner"] = allgraph_array["inner"]
    data["outer"] = allgraph_array["outer"]

    roots = ak.num(allgraph_array["inner"], axis=-1)
    data["roots"] = ak.local_index(roots)[roots == 0]
    data["idx"] = ak.local_index(data["barycenter_x"])

    torch.save(data, osp.join(raw_dir, f'data_id_{id}.pt'))


def process_event(idx, event, model_feature_keys, node_feature_dict, processed_dir, skeleton_features, device=torch.device("cpu")):
    device = processing_device(device)
    xp = get_array_module(prefer_cupy=device.type == "cuda")
    nTracksters = len(event["barycenter_x"])

    # Skip if not multiple tracksters
    if (nTracksters <= 1):
        return None, None

    # build feature list
    features = xp.stack([awkward_to_array(event[field], xp=xp) for field in model_feature_keys], axis=1)

    # Create base graph from geometrical graph = [[], []]
    targets = ak.ravel(event.outer)
    sources = ak.local_index(event.outer, axis=0)
    sources = ak.broadcast_arrays(sources, event.outer)[0]
    sources = ak.ravel(sources)

    edges = xp.transpose(xp.stack([awkward_to_array(targets, dtype=xp.int64, xp=xp), awkward_to_array(sources, dtype=xp.int64, xp=xp)]))
    if (edges.shape[0] < 2):
        return None, None

    if skeleton_features:
        edge_features = xp.zeros((len(edges[:, 0]), 7), dtype='f')

        edge_features[:, 5], edge_features[:, 6] = calc_min_max_skeleton_dist(nTracksters, edges, event["vertices"])
    else:
        edge_features = xp.zeros((len(edges[:, 0]), 5), dtype='f')
    edge_features[:, 0] = calc_edge_difference(edges, features, node_feature_dict, key="raw_energy")
    edge_features[:, 1] = calc_edge_difference(edges, features, node_feature_dict, key="barycenter_z")
    edge_features[:, 2] = calc_transverse_plane_separation(edges, features, node_feature_dict)
    edge_features[:, 3] = calc_spatial_compatibility(edges, features, node_feature_dict)
    edge_features[:, 4] = calc_edge_difference(edges, features, node_feature_dict, key="time")

    y = calc_group_score(edges, event.y, event.score, event.shared_e, event.raw_energy)
    if "isPU" in event.fields:
        isPU = awkward_to_array(event["isPU"], dtype=xp.int64, xp=xp)
    else:
        isPU = xp.zeros(nTracksters, dtype=xp.int64)
    cross_pu_edges = cross_PU(isPU, edges)
    signal_edges = mask_PU(isPU, edges, PU=False)
    pu_edges = mask_PU(isPU, edges, PU=True)
    y[cross_pu_edges | pu_edges] = 0
    PU_info = xp.stack([cross_pu_edges, signal_edges, pu_edges], axis=1)

    # Read data from `raw_path`.
    data = Data(
        x=array_to_tensor(features, device).float(),
        num_nodes=nTracksters, 
        edge_index=array_to_tensor(edges, device).long(),
        edge_features=array_to_tensor(edge_features, device).float(),
        y=array_to_tensor(y, device).float(),
        cluster=ak.to_torch(event.y).to(device),
        roots=ak.to_torch(event.roots).to(device),
        isPU=array_to_tensor(isPU, device).int(),
        PU_info=array_to_tensor(PU_info, device).bool())

    torch.save(data, osp.join(processed_dir, f'data_{idx}.pt'))
    return torch.max(torch.abs(data.x), axis=0).values, torch.max(data.edge_features, axis=0).values


class GNNDataset(Dataset):
    node_feature_keys = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "EV1", "EV2", "EV3",
                         "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "num_LCs", "num_hits", "raw_energy", "raw_em_energy", "photon_prob", "electron_prob", "muon_prob",
                         "neutral_pion_prob", "charged_hadron_prob", "neutral_hadron_prob", "z_min", "z_max", "LC_density", "trackster_density", "time"]
    node_feature_dict = {k: v for v, k in enumerate(node_feature_keys)}
    model_feature_keys = node_feature_keys

    # Skeleton Features computional intensive -> Turn off if not needed
    def __init__(self, root, histo_path, transform=None, test=False, skeleton_features=False, pre_transform=None, pre_filter=None, edge_scaler=None, node_scaler=None,
                 num_workers=24, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), split=None):
        self.split = split or ("test" if test else "train")
        self.test = test if split is None else self.split != "train"
        self.skeleton_features = skeleton_features
        self.device = processing_device(device)
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
        return ["DONE"]

    @property
    def processed_file_names(self):
        return ["DONE"]

    @property
    def raw_data_paths(self):
        return [osp.join(self.raw_dir, name) for name in sorted_basenames(f"{self.raw_dir}/data_id_*.pt")]

    @property
    def processed_data_paths(self):
        return [osp.join(self.processed_dir, name) for name in sorted_basenames(f"{self.processed_dir}/data_*.pt")]

    def download(self):
        files = sorted(glob(f"{self.histo_path}/{self.split}/*.root"))

        with tqdm(total=len(files)) as pbar:
            if self.num_workers <= 1:
                for id, file in enumerate(files):
                    download_event(id, file, self.raw_dir)
                    pbar.update()
            else:
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [executor.submit(download_event, id, files[id], self.raw_dir) for id in range(len(files))]

                    for future in as_completed(futures):
                        future.result()
                        pbar.update()
        torch.save({"split": self.split, "test": self.test, "files": len(files)}, osp.join(self.raw_dir, "DONE"))

    def process(self):
        idx = 0

        for raw_path in tqdm(self.raw_data_paths):
            run = torch.load(raw_path, weights_only=False)
            nEvents = len(run)

            if self.num_workers <= 1:
                results = [
                    process_event(idx+event, run[event], self.model_feature_keys, self.node_feature_dict,
                                  self.processed_dir, self.skeleton_features, self.device)
                    for event in range(nEvents)
                ]
            else:
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [executor.submit(process_event, idx+event, run[event], self.model_feature_keys, self.node_feature_dict,
                                               self.processed_dir, self.skeleton_features, self.device) for event in range(nEvents)]
                    results = [future.result() for future in as_completed(futures)]

            idx += nEvents
            for max_features, max_edge_features in results:
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

        for idx, file in tqdm(enumerate(self.processed_data_paths), desc="Fixing holes"):
            sample = torch.load(file, weights_only=False)
            fixed_path = osp.join(self.processed_dir, f"data_{idx}.pt")

            if (file != fixed_path):
                os.remove(file)
            torch.save(sample, fixed_path)
        torch.save({"split": self.split, "test": self.test, "events": len(self.processed_data_paths)}, osp.join(self.processed_dir, "DONE"))


    def len(self):
        return len(self.processed_data_paths)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data
