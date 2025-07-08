from tqdm import tqdm
import numpy as np

import torch

from tracksterLinker.GNN.TrackLinkingNet import FocalLoss


def train(model, opt, loader, epoch, emb_out=False, loss_obj=FocalLoss(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    epoch_loss = 0

    model.train()
    for sample in tqdm(loader, desc=f"Training Epoch {epoch}"):
        # reset optimizer and enable training mode
        opt.zero_grad()


        if emb_out:
            z, _ = model(sample.x, sample.edge_features, sample.edge_index, device=device, emb_out=True)
        else:
            z = model(sample.x, sample.edge_features, sample.edge_index, device=device)

        # compute the loss
        loss = loss_obj(z.squeeze(-1), sample.y)

        # back-propagate and update the weight
        loss.backward()
        opt.step()
        epoch_loss += loss

    return float(epoch_loss)/len(loader)


def test(model, loader, epoch, loss_obj=FocalLoss(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    with torch.set_grad_enabled(False):
        model.eval()
        pred, y = [], []
        val_loss = 0.0

        for sample in tqdm(loader, desc=f"Validation Epoch {epoch}"):
            nn_pred = model(sample.x, sample.edge_features, sample.edge_index, device=device)
            pred += nn_pred.squeeze(-1).tolist()
            y += sample.y.tolist()
            val_loss += loss_obj(nn_pred, sample.y.float()).item()

        val_loss /= len(loader)
    return val_loss, np.array(pred), np.array(y)
