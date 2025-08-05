from glob import glob
import tracksterLinker
from tracksterLinker.utils.graphUtils import *
from tracksterLinker.datasets.GNNDataset import GNNDataset
from tracksterLinker.datasets.NeoGNNDataset import NeoGNNDataset, load_branch_with_highest_cycle

from tracksterLinker.utils.dataUtils import *
import os.path as osp
import multiprocessing as mp

import uproot as uproot

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    base_folder = "/home/czeh"
    hist_folder = osp.join(base_folder, "histo_preparedGNN_0PU")
    data_folder_prepd = osp.join(base_folder, "compare/dataset_prepd")
    data_folder_hand = osp.join(base_folder, "compare/dataset_hand")


    file = uproot.open("/home/czeh/histo_preparedGNN_0PU/train/histo.root")
    dataset_prepd = NeoGNNDataset(data_folder_prepd, hist_folder)
    dataset_hand = GNNDataset(data_folder_hand, hist_folder, num_workers=1)

    nodes_hand = []
    nodes_prepd = []

    edges_hand = []
    edges_prepd = []
    for i in range(len(dataset_hand)):
        nodes_hand.append((dataset_hand.get(i).num_nodes, dataset_hand.get(i).edge_index.shape[0]))
        nodes_prepd.append((dataset_prepd.__getitem__(i).num_nodes, dataset_prepd.__getitem__(i).edge_index.shape[0]))

    for i in range(len(dataset_hand)):
        j = nodes_prepd.index(nodes_hand[i])
        assert torch.allclose(dataset_hand.get(i).x, dataset_prepd.__getitem__(j).x, atol=1e-4)
        assert torch.allclose(dataset_hand.get(i).edge_features, dataset_prepd.__getitem__(j).edge_features, atol=1e-4)
        assert torch.allclose(dataset_hand.get(i).edge_index, dataset_prepd.__getitem__(j).edge_index, atol=1e-4)
        assert torch.allclose(dataset_hand.get(i).y, dataset_prepd.__getitem__(j).y, atol=1e-6)

    print("Dataset is the same")
