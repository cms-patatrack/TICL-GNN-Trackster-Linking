import numpy as np

import tracksterLinker
from tracksterLinker.utils.graphUtils import *
from tracksterLinker.datasets.NeoGNNDataset import *

base_folder = "/home/czeh"
hist_folder = osp.join(base_folder, "new_graph_histo")

files = glob(f"{hist_folder}/test/*.root")

interest_features = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "num_LCs", "raw_energy", "z_min", "z_max", "LC_density"]


z_min = []
z_max = []
barycenter_x = []
barycenter_y = []
eVectori_x = []
eVectori_y = []
eVectori_z = []
LC_density = []
nTrackster = []



for file in files:
    file = uproot.open(file)

    allGNNtrain = load_branch_with_highest_cycle(file, 'ticlDumperGNN/GNNTraining')
    allGNNtrain_array = allGNNtrain.arrays()
    print(allGNNtrain_array.fields)

    
    for event in allGNNtrain_array:
        nTracksters = len(event["node_barycenter_x"])
        features = cp.stack([ak.to_cupy(event[f"node_{field}"]) for field in interest_features], axis=1)
        edges = cp.stack([ak.to_cupy(ak.flatten(event[f"edgeIndex_{field}"])) for field in ["out", "in"]], axis=1)
        edge_features = cp.stack([ak.to_cupy(ak.flatten(event[f"edge_{field}"])) for field in NeoGNNDataset.edge_feature_keys], axis=1)
        y = ak.to_cupy(ak.flatten(event["edge_weight"]))
        isPU = ak.to_cupy(event["simTrackster_isPU"][event["node_match_idx"]])

        print(event["simTrackster_isPU"])
        for simTr in range(len(event["simTrackster_isPU"])):
            if(event["simTrackster_isPU"][simTr]):
                continue

            trackster = ak.to_numpy(features[event["node_match_idx"] == simTr]) 
            print(np.mean(trackster, axis=0))
        break

    break
