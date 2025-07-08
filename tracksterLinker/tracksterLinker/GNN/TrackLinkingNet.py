import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

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
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.4):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions, targets):
        """Binary focal loss, mean.

        Per https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/5 with
        improvements for alpha.
        :param bce_loss: Binary Cross Entropy loss, a torch tensor.
        :param targets: a torch tensor containing the ground truth, 0s and 1s.
        :param gamma: focal loss power parameter, a float scalar.
        :param alpha: weight of the class indicated by 1, a float scalar.
        """
        ce_loss = F.binary_cross_entropy(
            predictions, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        alpha_tensor = (1 - self.alpha) + targets * (2 * self.alpha - 1)
        # alpha if target = 1 and 1 - alpha if target = 0
        f_loss = (alpha_tensor * (1 - p_t) ** self.gamma * ce_loss).mean()
        return f_loss * 1000


class GNN_TrackLinkingNet(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=16, output_dim=1, niters=2, dropout=0.2,
                 edge_feature_dim=12, edge_hidden_dim=16, weighted_aggr=True,
                 node_scaler=None, edge_scaler=None):
        super(GNN_TrackLinkingNet, self).__init__()

        self.niters = niters
        self.input_dim = input_dim
        self.edge_feature_dim = edge_feature_dim
        self.weighted_aggr = weighted_aggr

        if (node_scaler is not None):
            self.node_scaler = node_scaler
        else:
            self.node_scaler = torch.ones(input_dim)


        if (edge_scaler is not None):
            self.edge_scaler = edge_scaler
        else:
            self.node_scaler = torch.ones(edge_feature_dim)

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
            self.graphconvs.append(EdgeConvBlock(in_feat=hidden_dim,
                                                 out_feats=[2*hidden_dim, hidden_dim], dropout=dropout,
                                                 weighted_aggr=weighted_aggr))

        # Edge features from node embeddings for classification
        self.edgenetwork = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feature_dim +
                      edge_hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, X, edge_features, edge_index, return_emb=False, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        edge_features/= (edge_features + 10e-5) / self.edge_scaler
        X /= self.node_scaler
        edge_features_NN = self.edge_inputnetwork(edge_features)

        alpha_dir = self.attention_direct(edge_features_NN)
        alpha_rev = self.attention_reverse(edge_features_NN)
        alpha = torch.cat([alpha_dir, alpha_rev], dim=0).float()
        # Feature transformation to latent space
        node_emb = self.inputnetwork(X)
        ind_p1 = torch.cat((torch.arange(0, X.shape[0], dtype=int, device=device), edge_index[:, 0], edge_index[:, 1]))
        ind_p2 = torch.cat((torch.arange(0, X.shape[0], dtype=int, device=device), edge_index[:, 1], edge_index[:, 0]))

        # Niters x EdgeConv block
        for graphconv in self.graphconvs[:2]:
            node_emb = graphconv(node_emb, ind_p1, ind_p2, alpha=alpha, device=device)
        #node_emb = self.graphconvs[0](node_emb, ind_p1, ind_p2, alpha=alpha, device=device)
       
        edge_emb = torch.cat([node_emb[edge_index[:, 0]], node_emb[edge_index[:, 1]], edge_features_NN, edge_features], dim=-1)
        pred = self.edgenetwork(edge_emb)
        if not return_emb:
            return pred
        return pred, node_emb
