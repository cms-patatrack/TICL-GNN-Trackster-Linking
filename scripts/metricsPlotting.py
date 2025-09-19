import os.path as osp
import os
from glob import glob

import awkward as ak

import torch
import matplotlib.pyplot as plt

from tracksterLinker.datasets.NeoGNNDataset import NeoGNNDataset
from tracksterLinker.utils.graphHeatMap import GraphHeatmap


output_folder = "/home/czeh/newGNNMetrics"

# CUDA Setup
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

files = glob(osp.join(output_folder, f"*.pt"))
hm_dU_signal = GraphHeatmap(resolution=250)
hm_dO_signal = GraphHeatmap(resolution=250)
hm_dU_PU = GraphHeatmap(resolution=250)
hm_dO_PU = GraphHeatmap(resolution=250)

i = 0
print(files)

for file in files:
    print(file)
    metrics = torch.load(file, weights_only=False)
    
    hm_dU_signal.add_graph(metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_eta"]].cpu(), metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(), metrics["comp_dU_Signal"])
    hm_dO_signal.add_graph(metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_eta"]].cpu(), metrics["features"][~metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(), metrics["comp_dO_Signal"])
    hm_dU_PU.add_graph(metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_eta"]].cpu(), metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(), metrics["comp_dU_PU"])
    hm_dO_PU.add_graph(metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_eta"]].cpu(), metrics["features"][metrics["isPU"], NeoGNNDataset.node_feature_dict["barycenter_phi"]].cpu(), metrics["comp_dO_PU"])

    i += 1

hm_dU_signal.plot(show_nodes=False, file="multi_heat_dU_signal", folder=output_folder)
hm_dO_signal.plot(show_nodes=False, file="multi_heat_dO_signal", folder=output_folder)
hm_dU_PU.plot(show_nodes=False, file="multi_heat_dU_PU", folder=output_folder)
hm_dO_PU.plot(show_nodes=False, file="multi_heat_dO_PU", folder=output_folder)
