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
output_folder = "/eos/user/c/czeh/stabilityCheck"
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

files = glob(osp.join(output_folder, "metrics", f"graph_*.pt"))
hm_dU_signal = GraphHeatmap(resolution=250)
hm_dO_signal = GraphHeatmap(resolution=250)
hm_dU_PU = GraphHeatmap(resolution=250)
hm_dO_PU = GraphHeatmap(resolution=250)

i = 0

for file in files:
    print(file)
    metrics = torch.load(file, weights_only=False)
    
    hm_dU_signal.add_graph(metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_eta"]].cpu(), metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(), metrics["comp_dU_Signal"])
    hm_dO_signal.add_graph(metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_eta"]].cpu(), metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(), metrics["comp_dO_Signal"])
    hm_dU_PU.add_graph(metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_eta"]].cpu(), metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(), metrics["comp_dU_PU"])
    hm_dO_PU.add_graph(metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_eta"]].cpu(), metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(), metrics["comp_dO_PU"])

    i += 1

hm_dU_signal.plot(show_nodes=False, file="multi_heat_dU_signal", folder=output_folder)
hm_dO_signal.plot(show_nodes=False, file="multi_heat_dO_signal", folder=output_folder)
hm_dU_PU.plot(show_nodes=False, file="multi_heat_dU_PU", folder=output_folder)
hm_dO_PU.plot(show_nodes=False, file="multi_heat_dO_PU", folder=output_folder)
