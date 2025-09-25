import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tracksterLinker.GNN.EdgeConvBlock import EdgeConvBlock
from tracksterLinker.transformer.Transformer import EncoderLayer
from tracksterLinker.datasets.GNNDataset import GNNDataset


class PUNet(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=16, output_dim=1, niters=2, dropout=0.2, num_heads=12, num_layers=3,
                 edge_feature_dim=12, edge_hidden_dim=16, weighted_aggr=True, default_thresh = 0.6,
                 node_scaler=None, edge_scaler=None):
        super(PUNet, self).__init__()

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
        self.encoder_layers = nn.ModuleList([EncoderLayer(hidden_dim, num_heads, hidden_dim*2, dropout) for _ in range(num_layers)])

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
        self.pu_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

        self.edgenetwork = nn.Sequential(
            nn.Linear(2* hidden_dim + edge_feature_dim +
                      edge_hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
            # nn.LeakyReLU()
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
        node_emb = node_emb.unsqueeze(0)

        for enc_layer in self.encoder_layers:
            node_emb = enc_layer(node_emb)

        node_emb = node_emb.squeeze(0)

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
        # return embedding, self.outputnetwork(embedding)
        return None, embedding

    def forward(self, X, edge_features, edge_index):
        emb, pred = self.run(X, edge_features, edge_index)
        return pred
