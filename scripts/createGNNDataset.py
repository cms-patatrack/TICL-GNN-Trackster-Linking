import tracksterLinker
from tracksterLinker.utils.graphUtils import *
from tracksterLinker.datasets.GNNDataset import GNNDataset
import os.path as osp


base_folder = "/home/czeh"
model_folder = osp.join(base_folder, "GNN/model")
hist_folder = osp.join(base_folder, "histo_CloseByMP_0PU")
data_folder_training = osp.join(base_folder, "GNN/dataset")
data_folder_test = osp.join(base_folder, "GNN/dataset_test")

dataset_test = GNNDataset(data_folder_test, hist_folder, test=True)
dataset_training = GNNDataset(data_folder_training, hist_folder)

print("Training Dataset done. Statistics:")
print_graph_statistics(dataset_training)

print("Test Dataset done. Statistics:")
print_graph_statistics(dataset_test)
