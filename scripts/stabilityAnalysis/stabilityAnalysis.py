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

from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp

from tracksterLinker.datasets.NeoGNNDataset import NeoGNNDataset
from tracksterLinker.utils.dataStatistics import *
from tracksterLinker.utils.graphUtils import *
from tracksterLinker.utils.graphMetric import *

from tracksterLinker.utils.perturbations.allNodes import perturbate

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
    mp.set_start_method("spawn", force=True)
    model_name = "model-08-21"

    base_folder = "/home/czeh"
    model_folder = osp.join(base_folder, "GNN/model")
    output_folder = "/eos/user/c/czeh/stabilityCheck/energy_perturbations"
    hist_folder = osp.join(base_folder, "GNN/full_PU")
    data_folder = osp.join(base_folder, "GNN/datasetPU")
    os.makedirs(model_folder, exist_ok=True)

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
    i = 0

    n_perturb = 30
    futures = []
    with ProcessPoolExecutor(max_workers=min(32, n_perturb+1)) as executor:
        for sample in data_loader:
            print(f"Graph {i}")
            nn_pred = model.forward(sample.x, sample.edge_features, sample.edge_index, device=device)
             
            y_pred = (nn_pred > model.threshold).squeeze()
            y_true = (sample.y > 0).squeeze()

            graph_true = sample.edge_index[y_true]
            graph_pred = sample.edge_index[y_pred]

            os.makedirs(osp.join(output_folder, f"{i}"), exist_ok=True)

            futures.append(executor.submit(
                compute_and_save,
                graph_true, graph_pred, sample.x, sample.isPU, device, True,
                osp.join(output_folder, f"{i}", "baseline.pt"),
            ))

            #metrics = graph_dist(graph_true, graph_pred, sample.x, sample.isPU, device=device, verbose=True)
            #torch.save(metrics, osp.join(output_folder, f"{i}", f"baseline.pt"))

            random_values, perturbated_data = perturbate(sample.x, "raw_energy", max_val=10, num_data=n_perturb)

            for j, data in enumerate(perturbated_data):
                print(f"Perturbated Graph {j}: {random_values[j]}")
                nn_pred = model.forward(data, sample.edge_features, sample.edge_index, device=device)
                 
                y_pred = (nn_pred > model.threshold).squeeze()
                y_true = (sample.y > 0).squeeze()

                graph_true = sample.edge_index[y_true]
                graph_pred = sample.edge_index[y_pred]

                futures.append(executor.submit(
                    compute_and_save,
                    graph_true, graph_pred, data, sample.isPU, device, True,
                    osp.join(output_folder, f"{i}", f"graph_{j}.pt"),
                    {"allNodes_perturb": random_values[j]},
                ))

            if len(futures) > 2 * max_workers:
                done, futures = wait_some(futures)
                for f in done:
                    try:
                        print(f"[OK] Saved: {f.result()}")
                    except Exception as e:
                        print(f"[ERROR in worker] {e}")

            i += 1
            if i == 100:
                break


        for f in as_completed(futures):
            f.result()
            pbar.update()
