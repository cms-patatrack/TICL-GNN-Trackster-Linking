import os.path as osp
import os
from datetime import datetime
import json

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

i = 0
trackstersPU = []
trackstersSignal = []
for sample in data_loader:
    print(f"Graph {i}")
    nn_pred = model.forward(sample.x, sample.edge_features, sample.edge_index, device=device)
     
    y_pred = (nn_pred > model.threshold).squeeze()
    y_true = (sample.y > 0).squeeze()

    graph_true = sample.edge_index[y_true]
    graph_pred = sample.edge_index[y_pred]

    metrics = graph_dist(graph_true, graph_pred, sample.x, sample.isPU, device=device, verbose=True)
    torch.save(metrics, osp.join(output_folder, "metrics", f"graph_{i}.pt"))
    i += 1
