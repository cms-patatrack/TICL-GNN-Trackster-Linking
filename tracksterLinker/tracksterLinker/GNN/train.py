from tqdm import tqdm
import numpy as np

import torch

from tracksterLinker.GNN.TrackLinkingNet import FocalLoss
from tracksterLinker.datasets.GNNDataset import GNNDataset
from tracksterLinker.utils.dataUtils import calc_weights

def train(model, opt, loader, epoch, emb_out=False, loss_obj=FocalLoss(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    epoch_loss = 0

    model.train()
    for sample in tqdm(loader, desc=f"Training Epoch {epoch}"):
        # reset optimizer and enable training mode
        opt.zero_grad()
        z = model.run(sample.x, sample.edge_features, sample.edge_index, device=device)

        # compute the loss
        loss = loss_obj(z.squeeze(-1), sample.y)

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
        cross_edges = torch.zeros(4, device=device)
        signal_edges = torch.zeros(4, device=device)
        pu_edges = torch.zeros(4, device=device)
            
        for sample in tqdm(loader, desc=f"Validation Epoch {epoch}"):
            nn_pred = model.run(sample.x, sample.edge_features, sample.edge_index, device=device)
            weights = calc_weights(sample.edge_index, sample.x, GNNDataset.node_feature_dict, name=weighted)

            y_pred = (nn_pred > model.threshold).squeeze()
            y_true = (sample.y > 0).squeeze()

            cross_edges[0] += torch.sum(weights[sample.PU_info[:, 0]] * (y_true[sample.PU_info[:, 0]] & y_pred[sample.PU_info[:, 0]])).item()
            cross_edges[1] += torch.sum(weights[sample.PU_info[:, 0]] * (~y_true[sample.PU_info[:, 0]] & y_pred[sample.PU_info[:, 0]])).item()
            cross_edges[2] += torch.sum(weights[sample.PU_info[:, 0]] * (y_true[sample.PU_info[:, 0]] & ~y_pred[sample.PU_info[:, 0]])).item()
            cross_edges[3] += torch.sum(weights[sample.PU_info[:, 0]] * (~y_true[sample.PU_info[:, 0]] & ~y_pred[sample.PU_info[:, 0]])).item()

            signal_edges[0] += torch.sum(weights[sample.PU_info[:, 1]] * (y_true[sample.PU_info[:, 1]] & y_pred[sample.PU_info[:, 1]])).item()
            signal_edges[1] += torch.sum(weights[sample.PU_info[:, 1]] * (~y_true[sample.PU_info[:, 1]] & y_pred[sample.PU_info[:, 1]])).item()
            signal_edges[2] += torch.sum(weights[sample.PU_info[:, 1]] * (y_true[sample.PU_info[:, 1]] & ~y_pred[sample.PU_info[:, 1]])).item()
            signal_edges[3] += torch.sum(weights[sample.PU_info[:, 1]] * (~y_true[sample.PU_info[:, 1]] & ~y_pred[sample.PU_info[:, 1]])).item()
            
            pu_edges[0] += torch.sum(weights[sample.PU_info[:, 2]] * (y_true[sample.PU_info[:, 2]] & y_pred[sample.PU_info[:, 2]])).item()
            pu_edges[1] += torch.sum(weights[sample.PU_info[:, 2]] * (~y_true[sample.PU_info[:, 2]] & y_pred[sample.PU_info[:, 2]])).item()
            pu_edges[2] += torch.sum(weights[sample.PU_info[:, 2]] * (y_true[sample.PU_info[:, 2]] & ~y_pred[sample.PU_info[:, 2]])).item()
            pu_edges[3] += torch.sum(weights[sample.PU_info[:, 2]] * (~y_true[sample.PU_info[:, 2]] & ~y_pred[sample.PU_info[:, 2]])).item()
            
            val_loss += loss_obj(nn_pred.squeeze(-1), sample.y.float()).item()

        val_loss /= len(loader)
        return val_loss, cross_edges, signal_edges, pu_edges 


def validate(model, loader, epoch, weighted="raw_energy", loss_obj=FocalLoss(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    with torch.set_grad_enabled(False):
        model.eval()
        val_loss = 0.0

        pred, y, weights = [], [], []
        PU_info = [[], [], []]
            
        for sample in tqdm(loader, desc=f"Validation Epoch {epoch}"):
            nn_pred = model.run(sample.x, sample.edge_features, sample.edge_index, device=device)
            pred += nn_pred.squeeze(-1).tolist()
            y += sample.y.tolist()
            weights += calc_weights(sample.edge_index, sample.x, GNNDataset.node_feature_dict, name=weighted).tolist()
            PU_info[0] += sample.PU_info[:, 0].tolist()
            PU_info[1] += sample.PU_info[:, 1].tolist()
            PU_info[2] += sample.PU_info[:, 2].tolist()
            
            val_loss += loss_obj(nn_pred.squeeze(-1), sample.y.float()).item()

        val_loss /= len(loader)
    return val_loss, torch.tensor(pred), torch.tensor(y), torch.tensor(weights), torch.tensor(PU_info)
