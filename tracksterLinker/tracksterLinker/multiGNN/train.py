from tqdm import tqdm
import numpy as np

import torch

from tracksterLinker.GNN.LossFunctions import FocalLoss
from tracksterLinker.datasets.NeoGNNDataset import NeoGNNDataset
from tracksterLinker.utils.dataUtils import calc_weights

def train(model, opt, loader, epoch, weighted="raw_energy", emb_out=False, loss_obj=FocalLoss(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    epoch_loss = 0

    model.train()
    for sample in tqdm(loader, desc=f"Training Epoch {epoch}"):
        # reset optimizer and enable training mode
        opt.zero_grad()
        z = model.run(sample.x, sample.edge_features, sample.edge_index, device=device)
        weights = sample.x[:, NeoGNNDataset.node_feature_dict[weighted]]

        # compute the loss
        loss = loss_obj(z.squeeze(-1), sample.isPU.float(), weights)

        # back-propagate and update the weight
        loss.backward()
        opt.step()
        epoch_loss += loss

    return float(epoch_loss)/len(loader)


def test(model, loader, epoch, weighted="raw_energy", loss_obj=FocalLoss(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    with torch.set_grad_enabled(False):
        model.eval()
        val_loss = 0.0

        # 0: tp, 1: fp, 2: fn, 3: tn
        stats = torch.zeros(4, device=device)
            
        for sample in tqdm(loader, desc=f"Validation Epoch {epoch}"):
            nn_pred = model.run(sample.x, sample.edge_features, sample.edge_index, device=device)
            weights = sample.x[:, NeoGNNDataset.node_feature_dict[weighted]]

            y_pred = (nn_pred > model.threshold).squeeze()
            y_true = (sample.isPU == 1).squeeze()

            stats[0] += torch.sum(weights * (y_true & y_pred)).item()
            stats[1] += torch.sum(weights * (~y_true & y_pred)).item()
            stats[2] += torch.sum(weights * (y_true & ~y_pred)).item()
            stats[3] += torch.sum(weights * (~y_true & ~y_pred)).item()

            val_loss += loss_obj(nn_pred.squeeze(-1), sample.isPU.float(), weights).item()

        val_loss /= len(loader)
        return val_loss, stats


def validate(model, loader, epoch, weighted="raw_energy", loss_obj=FocalLoss(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    with torch.set_grad_enabled(False):
        model.eval()
        val_loss = 0.0

        pred, y, weights = [], [], []
            
        for sample in tqdm(loader, desc=f"Validation Epoch {epoch}"):
            nn_pred = model.run(sample.x, sample.edge_features, sample.edge_index, device=device)
            pred += nn_pred.squeeze(-1).tolist()
            y += sample.isPU.tolist()
            weight = sample.x[:, NeoGNNDataset.node_feature_dict[weighted]]
            weights += weight.tolist()
            
            val_loss += loss_obj(nn_pred.squeeze(-1), sample.isPU.float(), weight).item()

        val_loss /= len(loader)
    return val_loss, torch.tensor(pred), torch.tensor(y), torch.tensor(weights)
