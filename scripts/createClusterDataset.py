import os.path as osp

import torch

from tracksterLinker.datasets.ClusterDatasetBuilder import ClusterDatasetBuilder

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, number of devices: {torch.cuda.device_count()}")

torch.multiprocessing.set_start_method('spawn')

base_folder = "/data/czeh"
data_folder_training = osp.join(base_folder, "linking_dataset/dataset_hardronics")
store_folder_training = osp.join(base_folder, "linking_dataset/dataset_hardronics_tokens")
data_folder_test = osp.join(base_folder, "linking_dataset/dataset_hardronics_test")
store_folder_test = osp.join(base_folder, "linking_dataset/dataset_hardronics_test_tokens")

testBuilder = ClusterDatasetBuilder(store_folder_test, data_folder_test, input_length=input_length)

if not testBuilder.metadata_exists():
    testBuilder.generate(24, device)

trainBuilder = ClusterDatasetBuilder(store_folder_training, data_folder_training, input_length=input_length)

if not trainBuilder.metadata_exists():
    trainBuilder.generate(24, device)
