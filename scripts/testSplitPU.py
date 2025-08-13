from glob import glob
import os.path as osp

from torch_geometric.loader.dataloader import DataLoader

import tracksterLinker
from tracksterLinker.utils.graphUtils import *
from tracksterLinker.datasets.NeoGNNDataset import NeoGNNDataset, load_branch_with_highest_cycle

import networkx as nx
import numpy as np
from tqdm import tqdm

base_folder = "/home/czeh"
hist_folder = osp.join(base_folder, "histo_fullPU")
data_folder_test = osp.join(base_folder, "GNN/datasetPU_test")
batch_size = 1

dataset_test = NeoGNNDataset(data_folder_test, hist_folder, test=True)
test_dl = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

meanComps = []
maxComps = []
for sample in tqdm(test_dl):
    G = nx.Graph()
    G.add_nodes_from(list(range(sample.x.shape[0])))
    new_edge_index = sample.edge_index[~sample.PU_info[:, 0]].detach().cpu().numpy()

    G.add_edges_from(new_edge_index)

    #print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
    meanComps.append(np.mean([len(c) for c in nx.connected_components(G)]))
    maxComps.append(np.max([len(c) for c in nx.connected_components(G)]))

print(np.mean(meanComps))
print(np.max(maxComps))
