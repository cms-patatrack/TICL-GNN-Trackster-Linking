import os.path as osp
import os
from datetime import datetime

import torch
from torch import jit
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader.dataloader import DataLoader
import matplotlib.pyplot as plt

from tracksterLinker.datasets.NeoGNNDataset import NeoGNNDataset
from tracksterLinker.GNN.TrackLinkingNet import GNN_TrackLinkingNet, FocalLoss, EarlyStopping, weight_init
from tracksterLinker.GNN.train import *
from tracksterLinker.utils.dataStatistics import *
from tracksterLinker.utils.graphUtils import print_graph_statistics, negative_edge_imbalance
from tracksterLinker.utils.plotResults import *


model_name = "model-08-21"

base_folder = "/home/czeh"
model_folder = osp.join(base_folder, "GNN/model")
hist_folder = osp.join(base_folder, "GNN/full_PU")
data_folder = osp.join(base_folder, "GNN/datasetPU")
os.makedirs(model_folder, exist_ok=True)

# Prepare Dataset
batch_size = 1
dataset = NeoGNNDataset(data_folder, hist_folder, test=True, only_signal=True)
data_loader = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)

# CUDA Setup
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

# Prepare Model
model = jit.load(osp.join(model_folder, f"{model_name}.pt"))
model = model.to(device)



