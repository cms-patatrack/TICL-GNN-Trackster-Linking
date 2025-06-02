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
    model_feature_keys = np.array(["barycenter_eta", "barycenter_phi", "raw_energy"])

    def __init__(self, transformer, seq_length=60, max_nodes=66, scale=None):
        super(EventGrouping, self).__init__()

        self.seq_length = seq_length
        self.max_nodes = max_nodes

        self.transformer = transformer
        self.scale = scale

    def forward(self, data, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        num_nodes = data["nTrackster"]
        converter = Lang(trackster_list=data["lang"])
        visited = []
        step = 2
        sample_seq = converter.starting_seq(data["root"], self.seq_length).to(device)

        X = data["x"].float()
        X = F.pad(X, pad=(0, 0, self.max_nodes - num_nodes, 0), value=converter.word2index["<PAD>"])

        if(self.scale is not None):
            X /= self.scale

        X = X[:, list(map(self.node_feature_dict.get, self.model_feature_keys))]

        with torch.set_grad_enabled(False):
            self.transformer.eval()
            predictions = self.transformer(torch.unsqueeze(X, dim=0), torch.unsqueeze(sample_seq, dim=0))
            predicted_index = int(torch.argsort(-predictions[0, -1, :num_nodes], dim=0)[0].item())
            
            while (predicted_index != converter.word2index["<EOS>"] and step < self.seq_length-1):
                sample_seq[step] = predicted_index

                predictions = self.transformer(torch.unsqueeze(X, dim=0), torch.unsqueeze(sample_seq, dim=0))
                predicted_index = int(torch.argsort(-predictions[0, -1, :num_nodes], dim=0)[0].item())
                step += 1

        sample_seq[step] = predicted_index
        return sample_seq


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
