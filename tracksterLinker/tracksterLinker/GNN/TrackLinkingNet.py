import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tracksterLinker.GNN.EdgeConvBlock import EdgeConvBlock
from tracksterLinker.datasets.GNNDataset import GNNDataset


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, model, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

class GNN_TrackLinkingNet(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=16, output_dim=1, niters=2, dropout=0.2,
                 edge_feature_dim=12, edge_hidden_dim=16, weighted_aggr=True, default_thresh = 0.6,
                 node_scaler=None, edge_scaler=None):
        super(GNN_TrackLinkingNet, self).__init__()

        self.niters = niters
        self.input_dim = input_dim
        self.edge_feature_dim = edge_feature_dim
        self.weighted_aggr = weighted_aggr
        self.threshold = default_thresh

        if (node_scaler is None):
            node_scaler = torch.ones(input_dim)
        self.register_buffer("node_scaler", node_scaler)

        if (edge_scaler is None):
            edge_scaler = torch.ones(edge_feature_dim)
        self.register_buffer("edge_scaler", edge_scaler)

        # Feature transformation to latent space
        self.inputnetwork = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )

        # Edge Feature transformation to latent space
        self.edge_inputnetwork = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU()
        )

        self.attention_direct = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, 1),
            nn.Sigmoid()
        )

        self.attention_reverse = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, 1),
            nn.Sigmoid()
        )

        # EdgeConv
        self.graphconvs = nn.ModuleList()
        for i in range(niters):
            self.graphconvs.append(EdgeConvBlock(in_feat=hidden_dim, out_feats=[2*hidden_dim, hidden_dim], weighted_aggr=self.weighted_aggr, dropout=dropout))

        # Edge features from node embeddings for classification
        self.edgenetwork = nn.Sequential(
            nn.Linear(2* hidden_dim + edge_feature_dim +
                      edge_hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Sigmoid()
            nn.LeakyReLU()
        )

        self.outputnetwork = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def run(self, X, edge_features, edge_index):
        edge_features = (edge_features + 10e-5) / self.edge_scaler
        X = X / self.node_scaler
        edge_features_NN = self.edge_inputnetwork(edge_features)
        
        alpha_dir = self.attention_direct(edge_features_NN)
        alpha_rev = self.attention_reverse(edge_features_NN)
        alpha = torch.cat([alpha_dir, alpha_rev], dim=0)

        # Feature transformation to latent space
        node_emb = self.inputnetwork(X)
        empty = torch.zeros(X.shape[0], dtype=torch.int, device=X.device)
        src, dst = edge_index.unbind(1)
        ind_p1 = torch.cat((empty, src, dst))
        ind_p2 = torch.cat((empty, dst, src))

        # Niters x EdgeConv block
        for graphconv in self.graphconvs:
            node_emb = graphconv(node_emb, ind_p1, ind_p2, alpha=alpha, device=X.device)
        
        src_emb = node_emb.index_select(0, src)
        dst_emb = node_emb.index_select(0, dst)

        edge_emb = torch.cat([src_emb, dst_emb, edge_features_NN, edge_features], dim=-1)
        embedding = self.edgenetwork(edge_emb)
        return embedding, self.outputnetwork(embedding)
        # return None, embedding

    def forward(self, X, edge_features, edge_index):
        embedding, pred = self.run(X, edge_features, edge_index, device)
        return pred
