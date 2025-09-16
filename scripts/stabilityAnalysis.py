import os.path as osp
import os
from datetime import datetime

import awkward as ak

import torch
from torch import jit
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader.dataloader import DataLoader
import matplotlib.pyplot as plt

from tracksterLinker.datasets.NeoGNNDataset import NeoGNNDataset
from tracksterLinker.utils.dataStatistics import *
from tracksterLinker.utils.graphUtils import *

from tracksterLinker.utils.perturbations.allNodes import perturbate
from tracksterLinker.utils.perturbations.stabilityMap import *

model_name = "model-08-21"

base_folder = "/home/czeh"
model_folder = osp.join(base_folder, "GNN/model")
hist_folder = osp.join(base_folder, "GNN/full_PU")
data_folder = osp.join(base_folder, "GNN/datasetPU")
os.makedirs(model_folder, exist_ok=True)

# Prepare Dataset
batch_size = 1
dataset = NeoGNNDataset(data_folder, hist_folder, test=True)
data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

# CUDA Setup
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

# Prepare Model
model = jit.load(osp.join(model_folder, f"{model_name}.pt"))
model = model.to(device)
model.eval()

for sample in data_loader:
    print(sample.isPU)
    tracksters = []
    nn_pred = model.forward(sample.x, sample.edge_features, sample.edge_index, device=device)
     
    y_pred = (nn_pred > model.threshold).squeeze()
    y_true = (sample.y > 0).squeeze()

    graph_true = sample.edge_index[y_true]
    graph_pred = sample.edge_index[y_pred]
    true_components = find_connected_components(graph_true, sample.x.shape[0], device=device)
    true_component_features = get_component_features(true_components, sample.x)
    full_energy = torch.sum(true_component_features[:, NeoGNNDataset.node_feature_dict["raw_energy"]])
    print(f"Event Energy: {full_energy}")

    pred_components, pred_component_features = calc_overlapping_components(graph_pred, sample.x, true_components, device=device)
    #baseline_diff, true_energy = calc_missing_energy(graph_true, graph_pred, sample.x)
    baseline_energy_diff = true_component_features[:, NeoGNNDataset.node_feature_dict["raw_energy"]] - pred_component_features[:, NeoGNNDataset.node_feature_dict["raw_energy"]]
    print(f"Baseline: {pred_component_features[:, NeoGNNDataset.node_feature_dict['raw_energy']]}")
    print(baseline_energy_diff.shape[0])
    tracksters.append({"eta": pred_component_features[:, NeoGNNDataset.node_feature_dict["barycenter_eta"]].cpu(),
                       "phi": pred_component_features[:, NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(),
                       "z": pred_component_features[:, NeoGNNDataset.node_feature_dict["barycenter_z"]].cpu(),
                       "values": (baseline_energy_diff - baseline_energy_diff).cpu(),
                       "label": "baseline"})

    random_values, perturbated_data = perturbate(sample.x, "barycenter_z", max_val=10, num_data=20)
    for i, data in enumerate(perturbated_data):
        nn_pred = model.forward(data, sample.edge_features, sample.edge_index, device=device)
         
        y_pred = (nn_pred > model.threshold).squeeze()
        graph_pred = sample.edge_index[y_pred]

        pred_components, pred_component_features = calc_overlapping_components(graph_pred, sample.x, true_components, device=device)
        energy_diff = true_component_features[:, NeoGNNDataset.node_feature_dict["raw_energy"]] - pred_component_features[:, NeoGNNDataset.node_feature_dict["raw_energy"]]
        print(f"{random_values[i]:.05f}: {baseline_energy_diff}")
        tracksters.append({"eta": pred_component_features[:, NeoGNNDataset.node_feature_dict["barycenter_eta"]].cpu(),
                           "phi":pred_component_features[:, NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(),
                           "z":pred_component_features[:, NeoGNNDataset.node_feature_dict["barycenter_z"]].cpu(),
                           "values": (energy_diff - baseline_energy_diff).cpu(),
                           "label": f"{random_values[i]:.05f}"})

    plot_graphs_heatmap(tracksters, mode="3d", file="heatmap", folder='/eos/user/c/czeh/stabilityCheck/')
    plot_graphs_heatmap_interp(tracksters, file="cont_map", folder='/eos/user/c/czeh/stabilityCheck/')

    break
