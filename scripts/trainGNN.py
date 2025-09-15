import os.path as osp
import os
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader.dataloader import DataLoader
import matplotlib.pyplot as plt

from tracksterLinker.datasets.NeoGNNDataset import NeoGNNDataset
from tracksterLinker.GNN.TrackLinkingNet import GNN_TrackLinkingNet, FocalLoss, EarlyStopping, weight_init
from tracksterLinker.GNN.train import *
from tracksterLinker.utils.dataStatistics import *
from tracksterLinker.utils.graphUtils import print_graph_statistics, negative_edge_imbalance
from tracksterLinker.utils.plotResults import *


load_weights = False
model_name = "model_2025-08-14_final_loss_0.6184_epoch_20_dict"

base_folder = "/home/czeh"
model_folder = osp.join(base_folder, "GNN/modelPU")
hist_folder = osp.join(base_folder, "histo_fullPU")
data_folder_training = osp.join(base_folder, "GNN/datasetPU_hardronics")
data_folder_test = osp.join(base_folder, "GNN/datasetPU_hardronics_test")
os.makedirs(model_folder, exist_ok=True)

# Prepare Dataset
batch_size = 1
dataset_training = NeoGNNDataset(data_folder_training, hist_folder, only_signal=True)
dataset_test = NeoGNNDataset(data_folder_test, hist_folder, test=True, only_signal=True)
train_dl = DataLoader(dataset_training, shuffle=True, batch_size=batch_size)
test_dl = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)
print(f"Training Dataset: {len(train_dl)}, Test Dataset: {len(test_dl)}")

# CUDA Setup
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

# Prepare Model
start_epoch = 0
epochs = 200

model = GNN_TrackLinkingNet(input_dim=len(dataset_training.model_feature_keys),
                            edge_feature_dim=dataset_training[0].edge_features.shape[1],
                            edge_hidden_dim=16, hidden_dim=16, weighted_aggr=True, dropout=0.3,
                            node_scaler=dataset_training.node_scaler, edge_scaler=dataset_training.edge_scaler)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#increase weight on positive edges just a bit more
alpha = 0.5 + negative_edge_imbalance(dataset_test)/2
print(f"Focal loss with alpha={alpha}")
loss_obj = FocalLoss(alpha=alpha, gamma=2)
early_stopping = EarlyStopping(patience=20, delta=0)
model.apply(weight_init)

# Load weights if needed
date = f"{datetime.now():%Y-%m-%d}"

if load_weights:
    weights = torch.load(osp.join(model_folder, f"{model_name}.pt"), weights_only=True)
    model.load_state_dict(weights["model_state_dict"])
    optimizer.load_state_dict(weights["optimizer_state_dict"])
    start_epoch = weights["epoch"]

    save_model(model, 0, optimizer, [], [], output_folder=model_folder, filename=model_name, dummy_input=dataset_training[0])

# Scheduler after weight loading, to take new epoch size into account
scheduler = CosineAnnealingLR(optimizer, start_epoch+epochs, eta_min=1e-6)

train_loss_hist = []
val_loss_hist = []

for epoch in range(start_epoch, start_epoch+epochs):
    print(f'Epoch: {epoch+1}')
    loss = train(model, optimizer, train_dl, epoch+1, device=device, loss_obj=loss_obj)
    train_loss_hist.append(loss)

    val_loss, cross_edges, signal_edges, pu_edges = test(model, test_dl, epoch+1, loss_obj=loss_obj, device=device, weighted="raw_energy")
    val_loss_hist.append(val_loss)
    print(f'Training loss: {loss}, Validation loss: {val_loss}')

    plot_loss(train_loss_hist, val_loss_hist, save=True, output_folder=model_folder, filename=f"model_date_{date}_loss_epochs")

    print("Fast statistic on model threshold:")
    print("Only cross selected:")
    print_acc_scores_from_precalc(*cross_edges)
    print("Only signal trackster:") 
    print_acc_scores_from_precalc(*signal_edges)
    print("Only PU trackster:") 
    print_acc_scores_from_precalc(*pu_edges)
    
    if ((epoch+1) % 10 == 0):
        print("Store Diagrams")

        val_loss, pred, y, weight, PU_info = validate(model, test_dl, epoch+1, loss_obj=loss_obj, device=device, weighted="raw_energy")
        threshold = get_best_threshold(pred, y, weight)
        model.threshold = threshold

        print("weighted by raw energy:")
        print_binned_acc_scores(pred, y, weight, thres=threshold)
        plot_binned_validation_results(pred, y, weight, thres=threshold, output_folder=model_folder, file_suffix=f"epoch_{epoch+1}_date_{date}")
        plot_validation_results(pred, y, save=True, output_folder=model_folder, file_suffix=f"epoch_{epoch+1}_date_{date}", weight=weight)

    if ((epoch+1) % 5 == 0):
        print("Store Model")
        save_model(model, epoch, optimizer, train_loss_hist, val_loss_hist, output_folder=model_folder, filename=f"model_{date}", dummy_input=dataset_training[0])

    early_stopping(model, val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping after {epoch+1} epochs")
        early_stopping.load_best_model(model)

        save_model(model, epoch, optimizer, train_loss_hist, val_loss_hist, output_folder=model_folder, filename=f"model_{date}_final_loss_{-early_stopping.best_score:.4f}", dummy_input=dataset_training[0])
        break

    scheduler.step()
    plt.close()
