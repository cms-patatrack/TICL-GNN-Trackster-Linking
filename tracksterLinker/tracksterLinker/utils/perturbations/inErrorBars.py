
import torch 
import torch.distributions as dist

from tracksterLinker.datasets.GNNDataset import GNNDataset


# num_data is the maximal value, it can be less than that
def perturbate(node_features, num_samples=100, with_z=True, device=torch.device('cuda' if torch.cuda.is_available() else "cpu")):
    pca_values = torch.clamp(node_features[:, GNNDataset.node_feature_dict["sigmaPCA1"]:GNNDataset.node_feature_dict["sigmaPCA3"]+1], min=1e-6)
    eigenv = node_features[:, GNNDataset.node_feature_dict["eVector0_x"]:GNNDataset.node_feature_dict["eVector0_z"]+1]

    normal_dist = dist.Normal(torch.zeros(pca_values.shape, device=device), pca_values)
    
    data = torch.clone(torch.broadcast_to(node_features, (num_samples, node_features.shape[0], node_features.shape[1])))  
    multiple_samples = normal_dist.sample((num_samples,))
    perts = multiple_samples * eigenv

    if with_z:
        data[:, :, GNNDataset.node_feature_dict["barycenter_x"]:GNNDataset.node_feature_dict["barycenter_y"]+1] += perts[:, :, :2]
    else:
        data[:, :, GNNDataset.node_feature_dict["barycenter_x"]:GNNDataset.node_feature_dict["barycenter_z"]+1] += perts
    return data

