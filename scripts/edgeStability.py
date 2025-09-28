import os.path as osp
import os
from datetime import datetime
import json

import awkward as ak

import torch
from torch import jit
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader.dataloader import DataLoader
import matplotlib.pyplot as plt

from tracksterLinker.datasets.NeoGNNDataset import NeoGNNDataset
from tracksterLinker.utils.dataStatistics import *
from tracksterLinker.utils.dataUtils import *
from tracksterLinker.utils.graphUtils import *
from tracksterLinker.utils.graphMetric import *
from tracksterLinker.utils.graphHeatMap import GraphHeatmap

from tracksterLinker.utils.perturbations.inErrorBars import perturbate

def wait_some(futures):
    """Wait until at least one future completes, return (done, not_done)."""
    done, not_done = wait(futures, return_when=FIRST_COMPLETED)
    return done, list(not_done)

def compute_and_save(graph_true, graph_pred, data, isPU, device, verbose, path, extra_metrics=None):
    metrics = graph_dist(graph_true, graph_pred, data, isPU, device=device, verbose=verbose)
    if extra_metrics is not None:
        metrics.update(extra_metrics)
    torch.save(metrics, path)
    return path

if __name__ == "__main__":
    model_name = "model_2025-09-24_traced"
    model_thresh = 0.2
    model_comp_name = "newGNN_focal"
    model_comp_thresh = 0.2

    base_folder = "/home/czeh"
    model_folder = osp.join(base_folder, "GNN/modelfocal_small")
    output_folder = "/home/czeh/stability/model_comparison"
    hist_folder = osp.join(base_folder, "GNN/full_PU")
    data_folder = osp.join(base_folder, "GNN/dataset_hardronics_test")
    os.makedirs(output_folder, exist_ok=True)

    # Prepare Dataset
    batch_size = 1
    dataset = NeoGNNDataset(data_folder, hist_folder, test=True)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    # CUDA Setup
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    # Prepare Model
    model = jit.load(osp.join(model_folder, f"{model_name}.pt"))
    model = model.to(device)
    model.eval()

    model_comp = jit.load(osp.join(model_folder, f"{model_comp_name}.pt"))
    model_comp = model.to(device)
    model_comp.eval()
    i = 0

    n_perturb = 30
    n_graphs = 50
    futures = []
    mapp = GraphHeatmap(resolution=250, axis_names=["x", "y"], axis_values=None, mean=False)
    mapp_model_comp = GraphHeatmap(resolution=250, axis_names=["x", "y"], axis_values=None, mean=False)
    mapp_model_direct_comp = GraphHeatmap(resolution=250, axis_names=["x", "y"], axis_values=None, mean=False)
    for sample in data_loader:
        print(f"Graph {i}")
        nn_pred = model.forward(sample.x, sample.edge_features, sample.edge_index, device=device)
        nn_pred_comp = model_comp.forward(sample.x, sample.edge_features, sample.edge_index, device=device)
         
        y_true = sample.y
        y_base = (nn_pred > model.threshold).squeeze().int()
        y_base_comp = (nn_pred_comp > model.threshold).squeeze().int()

        data_x = ((sample.x[sample.edge_index[:, 1], NeoGNNDataset.node_feature_dict["barycenter_x"]] + sample.x[sample.edge_index[:, 0], NeoGNNDataset.node_feature_dict["barycenter_x"]])/2).cpu()
        data_y = ((sample.x[sample.edge_index[:, 1], NeoGNNDataset.node_feature_dict["barycenter_y"]] + sample.x[sample.edge_index[:, 0], NeoGNNDataset.node_feature_dict["barycenter_y"]])/2).cpu()
        #mapp.add_graph(data_x, data_y, (sample.PU_info[:, 1]).cpu())
        perturbated_data = perturbate(sample.x, num_samples=n_perturb, with_z=True)

        for j, data in enumerate(perturbated_data):
            print(f"{i}, {j}")
            nn_pred = model.forward(data, sample.edge_features, sample.edge_index, device=device)
            y_pred = (nn_pred > model_thresh).squeeze().int()
            
            mapp.add_graph(data_x, data_y, (y_base & y_pred).cpu())

            nn_pred = model_comp.forward(data, sample.edge_features, sample.edge_index, device=device)
            y_pred_comp = (nn_pred > model_comp_thresh).squeeze().int()
            
            mapp_model_comp.add_graph(data_x, data_y, (y_base_comp & y_pred_comp).cpu())
            mapp_model_direct_comp.add_graph(data_x, data_y, (~y_pred_comp & y_pred).cpu())


        i += 1
        if i == n_graphs:
            break

    mapp.plot(show_nodes=True, max_distance=5, file="contr_error_bars_edge_pos", folder=output_folder)
    mapp.plot(show_nodes=False, max_distance=5, file="contr_error_bars", folder=output_folder)
    mapp_model_comp.plot(show_nodes=False, max_distance=5, file="focal_error_bars", folder=output_folder)
    mapp_model_direct_comp.plot(show_nodes=False, max_distance=5, file="contr_focal_direct_error_bars", folder=output_folder)
