import os.path as osp
import os
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader.dataloader import DataLoader

from tracksterLinker.datasets.GNNDataset import GNNDataset
from tracksterLinker.GNN.TrackLinkingNet import GNN_TrackLinkingNet, FocalLoss, EarlyStopping, weight_init
from tracksterLinker.GNN.train import *
from tracksterLinker.utils.dataStatistics import *
from tracksterLinker.utils.graphUtils import print_graph_statistics
from tracksterLinker.utils.plotResults import *


load_weights = False
model_name = ""

base_folder = "/home/czeh"
model_folder = osp.join(base_folder, "GNN/model")
hist_folder = osp.join(base_folder, "histo_CloseByMP_0PU")
data_folder_training = osp.join(base_folder, "GNN/dataset")
data_folder_test = osp.join(base_folder, "GNN/dataset_test")
os.makedirs(model_folder, exist_ok=True)

# Prepare Dataset
dataset_test = GNNDataset(data_folder_test, hist_folder, test=True)
dataset_training = GNNDataset(data_folder_training, hist_folder)
train_dl = DataLoader(dataset_training, shuffle=True)
test_dl = DataLoader(dataset_test, shuffle=True)

# CUDA Setup
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

# Prepare Model
epochs = 200

model = GNN_TrackLinkingNet(input_dim=dataset_training.model_feature_keys.shape[0],
                            edge_feature_dim=dataset_training.get(0).edges_features.shape[1],
                            edge_hidden_dim=16, hidden_dim=16, weighted_aggr=True,
                            dropout=0.3)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
loss_obj = FocalLoss(alpha=0.45, gamma=2)
early_stopping = EarlyStopping(patience=20, delta=-2)

model.apply(weight_init)

# Load weights if needed
start_epoch = 0

if load_weights:
    weights = torch.load(osp.join(model_folder, model_name), weights_only=True)
    model.load_state_dict(weights["model_state_dict"])
    optimizer.load_state_dict(weights["optimizer_state_dict"])
    start_epoch = weights["epoch"]


train_loss_hist = []
val_loss_hist = []
date = f"{datetime.now():%Y-%m-%d}"

for epoch in range(start_epoch, epochs):
    print(f'Epoch: {epoch+1}')

    loss = train(model, optimizer, train_dl, epoch+1, device=device, loss_obj=loss_obj)
    train_loss_hist.append(loss)

    val_loss, pred, y = test(model, test_dl, epoch+1, loss_obj=loss_obj, device=device)
    val_loss_hist.append(val_loss)

    plot_loss(train_loss_hist, val_loss_hist, save=True, filename=f"model_date_{date}")
    early_stopping(model, val_loss)

    if early_stopping.early_stop:
        print(f"Early stopping after {epoch+1} epochs")
        early_stopping.load_best_model(model)

        plot_validation_results(pred, y, save=True, output_folder=model_folder, file_suffix=f"epoch_{epoch+1}_date_{date}")
        save_model(model, epoch, optimizer, loss, val_loss, output_folder=model_folder, filename=f"model_date_{date}_final_loss_{-early_stopping.best_score:.4f}.pt")
        break

    elif ((epoch+1) % 20 == 0):
        print(f'Epoch: {epoch+1}')
        plot_validation_results(pred, y, save=((epoch+1) % 60 == 0 or epoch+1 == epochs))

        if ((epoch+1) % 60 == 0 or epoch+1 == epochs):
            save_model(model, epoch, optimizer, loss, val_loss, output_folder=model_folder, filename=f"model_epoch_{epoch+1}_date_{date}_loss_{val_loss:.4f}.pt")

    scheduler.step()
