import torch
import torch.nn as nn


class EdgeConvBlock(nn.Module):
    """EdgeConv layer.
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """

    def __init__(self, in_feat, out_feats, activation=True, dropout=0.2, weighted_aggr=False):
        super(EdgeConvBlock, self).__init__()
        self.activation = activation
        self.num_layers = len(out_feats)
        self.weighted_aggr = weighted_aggr

        self.drop = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Linear(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i]))

        self.acts = nn.ModuleList()
        for i in range(self.num_layers):
            self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Linear(in_feat, out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, features, ind_p1, ind_p2, alpha=None, device='cpu'):
        EDGE_EMB_p1 = features[ind_p1, :]
        EDGE_EMB_p2 = features[ind_p2, :] - EDGE_EMB_p1

        x = torch.cat((EDGE_EMB_p1, EDGE_EMB_p2), dim=1)
        N = features.shape[0]
        i = 0
        
        for conv, act in zip(self.convs, self.acts):
            x = conv(x)
            if self.activation:
                x = act(x)
            if i == 0:
                x = self.drop(x)
            i += 1

        # Do aggregation
        if self.weighted_aggr and alpha is not None:
            alpha_vec = torch.cat((torch.ones(N, device=device, dtype=float).float(), torch.squeeze(alpha)), dim=0)
            x = torch.mul(alpha_vec, x.transpose(0, 1)).transpose(0, 1)

        # Create a destination tensor to store the summed rows
        summed_matrix = torch.zeros(N, x.size(1), device=device).float()
        # Sum the rows based on the index using torch.scatter_add
        x = torch.scatter_add(summed_matrix, 0, ind_p1.unsqueeze(1).repeat(1, x.size(1)), x)

        # Skip connection:
        if self.sc:
            sc = self.sc(features)
        else:
            sc = features
        out = self.sc_act(sc + x)

        return out
