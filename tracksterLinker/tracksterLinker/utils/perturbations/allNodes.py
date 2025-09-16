import torch

from tracksterLinker.datasets.GNNDataset import GNNDataset


# num_data is the maximal value, it can be less than that
def perturbate(node_features, feature_name, max_val=0.001, num_data=100, device=torch.device('cuda' if torch.cuda.is_available() else "cpu")):
    feature_idx = GNNDataset.node_feature_dict[feature_name] 

    random_values = torch.unique(torch.rand(num_data, device=device) * max_val * 2 - max_val)
    data = torch.clone(torch.broadcast_to(node_features, (random_values.shape[0], node_features.shape[0], node_features.shape[1])))  

    data[:, :, feature_idx] += random_values.expand(node_features.shape[0], random_values.shape[0]).T
    return random_values, data

