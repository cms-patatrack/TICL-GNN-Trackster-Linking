import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ClusterDataset import ClusterDataset
from Transformer import Transformer
from lang import Lang


class EventGrouping(nn.Module):

    node_feature_keys = ["barycenter_x", "barycenter_y", "barycenter_z", "barycenter_eta", "barycenter_phi", "eVector0_x", "eVector0_y", "eVector0_z", "EV1", "EV2", "EV3",
                         "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "num_LCs", "num_hits", "raw_energy", "raw_em_energy", "photon_prob", "electron_prob", "muon_prob",
                         "neutral_pion_prob", "charged_hadron_prob", "neutral_hadron_prob", "z_min", "z_max", "LC_density", "trackster_density", "time", "idx"]
    node_feature_dict = {k: v for v, k in enumerate(node_feature_keys)}
    model_feature_keys = np.array(["idx", "barycenter_eta", "barycenter_phi", "raw_energy"])

    def __init__(self, converter, transformer, neighborhood=4, seq_length=60, max_nodes=66):
        super(EventGrouping, self).__init__()

        self.neighborhood = neighborhood
        self.seq_length = seq_length
        self.max_nodes = max_nodes

        self.converter = converter
        self.transformer = transformer

    def build_subgraph(self, graph, root, neighborhood=1):
        neighbors = graph[1][graph[0] == root]

        if (neighborhood == 0):
            return neighbors
        subgraph = np.copy(neighbors)

        for n in neighbors:
            subgraph = np.append(subgraph, self.build_subgraph(graph, n, neighborhood-1))

        return np.unique(subgraph)

    def forward(self, data):
        num_nodes = data.x.shape[0]
        visited = []
        seqs = []
        for root in data.roots:
            step = 0
            root_subgraph = np.append(self.build_subgraph(data.edge_index, root, self.neighborhood), root)
            root_subgraph = np.array(root_subgraph, dtype=int)

            sample_seq = torch.from_numpy(self.converter.starting_seq(root.item(), self.seq_length)).long()

            X = data.x[root_subgraph].float()
            X = F.pad(X, pad=(0, 0, self.max_nodes - data.x.shape[0], 0), value=self.converter.word2index["<PAD>"])
            X = X[:, list(map(self.node_feature_dict.get, self.model_feature_keys))]

            predictions = self.transformer(torch.unsqueeze(X, dim=0), torch.unsqueeze(sample_seq, dim=0))
            predicted_index = predictions.argmax(-1)
            predicted_number = int(predicted_index[0, -1].item())
            print(predicted_number)

            while (predicted_number != converter.word2index["<EOS>"] and step < num_nodes):
                sample_seq = torch.roll(sample_seq, -1, dims=0)
                sample_seq[-1] = predicted_number

                if (sample_seq[-1] > self.converter.word2index[";"] and predicted_number < num_nodes):
                    new_root = int(self.converter.index2word[sample_seq[-1].item()])

                    subgraph = np.append(self.build_subgraph(data.edge_index, new_root, self.neighborhood), new_root)
                    subgraph = np.array(subgraph, dtype=int)
                    subgraph = np.setdiff1d(subgraph, visited)
                    visited.append(new_root)

                    X = data.x[subgraph].float()
                    X = F.pad(X, pad=(0, 0, self.max_nodes - data.x.shape[0], 0), value=self.converter.word2index["<PAD>"])
                    X = X[:, list(map(self.node_feature_dict.get, self.model_feature_keys))]
                    predictions = self.transformer(torch.unsqueeze(X, dim=0), torch.unsqueeze(sample_seq, dim=0))
                else:
                    X = data.x[root_subgraph].float()
                    X = F.pad(X, pad=(0, 0, self.max_nodes - data.x.shape[0], 0), value=self.converter.word2index["<PAD>"])
                    X = X[:, list(map(self.node_feature_dict.get, self.model_feature_keys))]
                    predictions = self.transformer(torch.unsqueeze(X, dim=0), torch.unsqueeze(sample_seq, dim=0))

                predicted_index = predictions.argmax(-1)
                predicted_number = predicted_index[0, -1].item()
                print(predicted_number)
                step += 1

            seqs.append(sample_seq)
        return seqs


if __name__ == "__main__":
    data_folder_test = "/Users/chrisizeh/cernbox/histo"
    store_folder_training = "/Users/chrisizeh/cernbox/graph_data_test"
    model_path = "/Users/chrisizeh/cernbox/tranformer_4.pt"

    max_nodes = 66
    input_length = 60
    converter = Lang(max_nodes)
    vocab_size = converter.n_words
    dataset = ClusterDataset(store_folder_training, data_folder_test, test=True)

    d_model = 16
    num_heads = 2
    num_layers = 4
    d_ff = 32
    dropout = 0
    padding = converter.word2index["<PAD>"]
    feature_num = 4

    model = Transformer(padding, vocab_size, d_model, num_heads, num_layers, d_ff, feature_num, max_nodes, input_length, dropout)

    weights = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(weights["model_state_dict"])

    runner = EventGrouping(converter, model, neighborhood=1, seq_length=input_length, max_nodes=max_nodes)
    print(runner(dataset.get(3)))
