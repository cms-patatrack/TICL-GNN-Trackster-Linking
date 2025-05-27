from ClusterDataset import ClusterDataset as GNNDataset
from ClusterDatasetTransformer import ClusterDataset

input_length = 60
max_seq_length = 60
batch_size = 64
max_nodes = 66

converter = Lang(max_nodes)
vocab_size = converter.n_words

# Load the dataset
hist_folder = "/eos/user/c/czeh/histo_10pion0PU/"
data_folder_training = "/eos/user/c/czeh/graph_data"
data_folder_test = "/eos/user/c/czeh/graph_data_test"

dataset_training = GNNDataset(data_folder_training, hist_folder)
dataset_test = GNNDataset(data_folder_test, hist_folder, test=True)

data_folder_training = "/eos/user/c/czeh/graph_data/processed"
store_folder_training = "/eos/user/c/czeh/graph_data_trans"
data_folder_test = "/eos/user/c/czeh/graph_data_test/processed"
store_folder_test = "/eos/user/c/czeh/graph_data_trans_test"

dataset_training = ClusterDataset(converter, store_folder_training, data_folder_training, max_nodes=max_nodes, input_length=input_length)
dataset_test = ClusterDataset(converter, store_folder_test, data_folder_test, max_nodes=max_nodes, input_length=input_length)
