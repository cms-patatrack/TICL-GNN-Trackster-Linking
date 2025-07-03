import tracksterLinker
from tracksterLinker.utils.graphUtils import *
from tracksterLinker.datasets.GNNDataset import GNNDataset

import os.path as osp
import multiprocessing as mp


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    base_folder = "/home/czeh"
    model_folder = osp.join(base_folder, "GNN/model")
    hist_folder = osp.join(base_folder, "histo_CloseByMP_0PU")
    data_folder_training = osp.join(base_folder, "GNN/dataset")
    data_folder_test = osp.join(base_folder, "GNN/dataset_test")

    dataset_training = GNNDataset(data_folder_training, hist_folder)
    dataset_test = GNNDataset(data_folder_test, hist_folder, test=True, node_scaler=dataset_training.node_scaler, edge_scaler=dataset_training.edge_scaler)

    print("Training Dataset done. Statistics:")
    print_graph_statistics(dataset_training)

    print("Test Dataset done. Statistics:")
    print_graph_statistics(dataset_test)
