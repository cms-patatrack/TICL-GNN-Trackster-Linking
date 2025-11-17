import os.path as osp
import os
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader.dataloader import DataLoader
import matplotlib.pyplot as plt

from tracksterLinker.datasets.NeoGNNDataset import NeoGNNDataset
from tracksterLinker.GNN.TrackLinkingNet import GNN_TrackLinkingNet, EarlyStopping, weight_init
from tracksterLinker.multiGNN.PUNet import PUNet
from tracksterLinker.GNN.LossFunctions import *
from tracksterLinker.GNN.train import *
from tracksterLinker.utils.dataStatistics import *
from tracksterLinker.utils.graphUtils import print_graph_statistics, negative_edge_imbalance
from tracksterLinker.utils.plotResults import *


load_weights = True 
model_name = "start_model"

base_folder = "/home/czeh"
model_folder = osp.join(base_folder, "GNN/0005_signal_pu_edges")
hist_folder = osp.join(base_folder, "GNN/histo")
data_folder_training = osp.join(base_folder, "GNN/dataset_hardronics")
data_folder_test = osp.join(base_folder, "GNN/dataset_hardronics_test")
os.makedirs(model_folder, exist_ok=True)

# Prepare Dataset
batch_size = 1
dataset_training = NeoGNNDataset(data_folder_training, hist_folder, only_signal=False)
dataset_test = NeoGNNDataset(data_folder_test, hist_folder, test=True, only_signal=False)
train_dl = DataLoader(dataset_training, shuffle=True, batch_size=batch_size)
test_dl = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)
print(f"Training Dataset: {len(train_dl)}, Test Dataset: {len(test_dl)}")

# CUDA Setup
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

# Prepare Model
start_epoch = 0
epochs = 60

# model = GNN_TrackLinkingNet(input_dim=len(dataset_training.model_feature_keys),
#                             edge_feature_dim=dataset_training[0].edge_features.shape[1], niters=2,
#                             edge_hidden_dim=16, hidden_dim=16, weighted_aggr=True, dropout=0.3,
#                             node_scaler=dataset_training.node_scaler, edge_scaler=dataset_training.edge_scaler)
model = PUNet(input_dim=len(dataset_training.model_feature_keys),
                            edge_feature_dim=dataset_training[0].edge_features.shape[1], niters=4,
                            edge_hidden_dim=64, hidden_dim=128, num_heads=8, weighted_aggr=True, dropout=0.3,
                            node_scaler=dataset_training.node_scaler, edge_scaler=dataset_training.edge_scaler)
model = model.to(device)
# LR is upper bound for Adam
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95),
                              eps=1e-8, weight_decay=0.01, amsgrad=True)

#increase weight on positive edges just a bit more
alpha = 0.5 + negative_edge_imbalance(dataset_test)/2
print(f"Focal loss with alpha={alpha}")
loss_obj = CombinedLoss(alpha=alpha, gamma=2, margin=2.0, weightFocal=100, weightContrastive=0.0001)


early_stopping = EarlyStopping(patience=20, delta=0)
model.apply(weight_init)

# Load weights if needed
date = f"{datetime.now():%Y-%m-%d}"

if load_weights:
    weights = torch.load(osp.join(model_folder, f"{model_name}.pt"), weights_only=True)
    model.load_state_dict(weights["model_state_dict"], strict=False)
    # optimizer.load_state_dict(weights["optimizer_state_dict"])
    # start_epoch = weights["epoch"]

    save_model(model, 0, optimizer, [], [], output_folder=model_folder, filename=model_name, dummy_input=dataset_training[0])

# Scheduler after weight loading, to take new epoch size into account
scheduler = CosineAnnealingLR(optimizer, T_max=start_epoch+epochs)

train_loss_hist = []
val_loss_hist = []

print(scheduler.get_last_lr())
last_epoch = start_epoch + epochs
for epoch in range(start_epoch, last_epoch):
    print(f'Epoch: {epoch}')
    loss = train(model, optimizer, train_dl, epoch, loss_obj=loss_obj, scores=True)
    train_loss_hist.append(loss)

    val_loss, cross_edges, signal_edges, pu_edges = test(model, test_dl, epoch, loss_obj=loss_obj.focal, device=device, weighted="raw_energy")
    val_loss_hist.append(val_loss)
    print(f'Training loss: {loss}, Validation loss: {val_loss}, Learning Rate: {scheduler.get_last_lr()}')

    plot_loss(train_loss_hist, val_loss_hist, save=True, output_folder=model_folder, filename=f"model_date_{date}_loss_epochs")

    print("Fast statistic on model threshold:")
    print("Only cross selected:")
    print_acc_scores_from_precalc(*cross_edges)
    print("Only signal trackster:") 
    print_acc_scores_from_precalc(*signal_edges)
    print("Only PU trackster:") 
    print_acc_scores_from_precalc(*pu_edges)
    
    if ((epoch % 5 == 0) or (epoch + 1 == last_epoch)):
        print("Store Model")
        save_model(model, epoch, optimizer, train_loss_hist, val_loss_hist, output_folder=model_folder, filename=f"model_{date}", dummy_input=dataset_training[0])

    if ((epoch % 5 == 0) or (epoch + 1 == last_epoch)):
        print("Store Diagrams")

        val_loss, pred, y, weight, PU_info = validate(model, test_dl, epoch, loss_obj=loss_obj.focal, weighted="raw_energy")
        threshold = get_best_threshold(pred, y, weight)
        model.threshold = threshold

        print("weighted by raw energy:")
        plot_binned_validation_results(pred, y, weight, thres=threshold, output_folder=model_folder, file_suffix=f"epoch_{epoch}_date_{date}")
        plot_validation_results(pred, y, save=True, output_folder=model_folder, file_suffix=f"epoch_{epoch}_date_{date}", weight=weight)

    early_stopping(model, val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping after {epoch} epochs")
        early_stopping.load_best_model(model)

        save_model(model, epoch, optimizer, train_loss_hist, val_loss_hist, output_folder=model_folder, filename=f"model_{date}_final_loss_{-early_stopping.best_score:.4f}", dummy_input=dataset_training[0])
        break

    scheduler.step()
    plt.close()
