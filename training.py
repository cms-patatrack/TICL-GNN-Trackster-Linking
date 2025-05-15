import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch

from test import *
from GNN_TrackLinkingNet import prepare_network_input_data, FocalLoss


def train(model, opt, loader, epoch, edge_features=True, emb_out=False, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), loss_obj=FocalLoss()):

    epoch_loss = 0
    model.train()
    for sample in tqdm(loader, desc=f"Training epoch {epoch}"):
        # reset optimizer and enable training mode
        opt.zero_grad()

        # move data to the device
        sample = sample.to(device)

        # get the prediction tensor
        if edge_features:
            if sample.edge_index.shape[1] != sample.edges_features.shape[0]:
                continue
            data = prepare_network_input_data(sample.x, sample.edge_index, edge_features=sample.edges_features)
        else:
            data = prepare_network_input_data(sample.x, sample.edge_index)

        if emb_out:
            z, _ = model(*data, device=device)
        else:
            z = model(*data, device=device)

        # compute the loss
        loss = loss_obj(z, sample.y.float())

        # back-propagate and update the weight
        loss.backward()
        opt.step()
        epoch_loss += loss

    return float(epoch_loss)/len(loader)
