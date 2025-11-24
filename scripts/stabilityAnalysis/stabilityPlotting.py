import os.path as osp
import os
from datetime import datetime
import json
from glob import glob

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
from tracksterLinker.utils.graphHeatMap import GraphHeatmap

model_name = "model-08-21"

base_folder = "/home/czeh"
model_folder = osp.join(base_folder, "GNN/model")
output_folder = "/eos/user/c/czeh/stabilityCheck/energy_perturbations"
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

hm_dU_signal = GraphHeatmap(resolution=250)
hm_dO_signal = GraphHeatmap(resolution=250)
hm_dU_PU = GraphHeatmap(resolution=250)
hm_dO_PU = GraphHeatmap(resolution=250)

files = glob(osp.join(output_folder, "*", f"*.pt"), recursive=True)
graph_folders = glob(osp.join(output_folder, "*"))

for graph_path in graph_folders:
    if (not os.path.isdir(graph_path) or not os.path.isfile(osp.join(graph_path, f"baseline.pt"))):
        continue

    baseline_metrics = torch.load(osp.join(graph_path, f"baseline.pt"), weights_only=False)

    files = glob(osp.join(graph_path, f"graph_*.pt"), recursive=True)
    for file in files:
        print(file)
        metrics = torch.load(file, weights_only=False)
        #print(torch.max(metrics["features"][:, NeoGNNDataset.node_feature_dict["barycenter_eta"]]))
        
        hm_dU_signal.add_graph(torch.abs(metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_eta"]]).cpu(), metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(), baseline_metrics["comp_dU_Signal"] - metrics["comp_dU_Signal"])
        hm_dO_signal.add_graph(torch.abs(metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_eta"]]).cpu(), metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(), baseline_metrics["comp_dO_Signal"] - metrics["comp_dO_Signal"])
        hm_dU_PU.add_graph(torch.abs(metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_eta"]]).cpu(), metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(), baseline_metrics["comp_dU_PU"] - metrics["comp_dU_PU"])
        hm_dO_PU.add_graph(torch.abs(metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_eta"]]).cpu(), metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(), baseline_metrics["comp_dO_PU"] - metrics["comp_dO_PU"])

hm_dU_signal.plot(show_nodes=False, file="multi_heat_dU_signal", folder=output_folder)
hm_dO_signal.plot(show_nodes=False, file="multi_heat_dO_signal", folder=output_folder)
hm_dU_PU.plot(show_nodes=False, file="multi_heat_dU_PU", folder=output_folder)
hm_dO_PU.plot(show_nodes=False, file="multi_heat_dO_PU", folder=output_folder)
