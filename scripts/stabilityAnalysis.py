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
from tracksterLinker.utils.graphMetric import *

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

i = 0
for sample in data_loader:
    trackstersPU = []
    trackstersSignal = []
    nn_pred = model.forward(sample.x, sample.edge_features, sample.edge_index, device=device)
     
    y_pred = (nn_pred > model.threshold).squeeze()
    y_true = (sample.y > 0).squeeze()

    graph_true = sample.edge_index[y_true]
    graph_pred = sample.edge_index[y_pred]

    metrics = graph_dist(graph_true, graph_pred, sample.x, sample.isPU, device=device, verbose=True)

    # TODO: How to work with negative postions in map???
    trackstersSignal.append({"eta": torch.abs(metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_eta"]].cpu()),
                       "phi": metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(),
                       "z": torch.abs(metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_z"]].cpu()),
                       "energy": metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["raw_energy"]].cpu(),

                       "dU": metrics["comp_dU_Signal"],
                       "dO": metrics["comp_dO_Signal"],
                       "full_energy": metrics["energy_Signal"],
                       "label": f"Graph {i}"})

    trackstersPU.append({"eta": torch.abs(metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_eta"]].cpu()),
                       "phi": metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(),
                       "z": torch.abs(metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_z"]].cpu()),
                       "energy": metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["raw_energy"]].cpu(),

                       "dU": metrics["comp_dU_PU"],
                       "dO": metrics["comp_dO_PU"],
                       "full_energy": metrics["energy_PU"],
                       "label": f"Graph {i}"})

    if i == 4:
        break
    i += 1
plot_graphs_heatmap(trackstersSignal, mode="3d", values="dO", file="dO_Signal", folder='/eos/user/c/czeh/stabilityCheck/')
plot_graphs_heatmap(trackstersSignal, mode="3d", values="dU", file="dU_Signal", folder='/eos/user/c/czeh/stabilityCheck/')
plot_graphs_heatmap(trackstersPU, mode="3d", values="dU", file="dU_PU", folder='/eos/user/c/czeh/stabilityCheck/')
plot_graphs_heatmap(trackstersPU, mode="3d", values="dO", file="dO_PU", folder='/eos/user/c/czeh/stabilityCheck/')
plot_graphs_heatmap_interp(trackstersSignal, values="dO", file="cont_map_dO_Signal", folder='/eos/user/c/czeh/stabilityCheck/')
plot_graphs_heatmap_interp(trackstersSignal, values="dU", file="cont_map_dU_Signal", folder='/eos/user/c/czeh/stabilityCheck/')
plot_graphs_heatmap_interp(trackstersPU, values="dU", file="cont_map_dU_PU", folder='/eos/user/c/czeh/stabilityCheck/')
plot_graphs_heatmap_interp(trackstersPU, values="dO", file="cont_map_dO_PU", folder='/eos/user/c/czeh/stabilityCheck/')

