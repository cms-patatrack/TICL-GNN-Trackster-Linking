import argparse
import csv
import json
import os
import os.path as osp
import sys
import tempfile
import types
from glob import glob

os.environ.setdefault("MPLCONFIGDIR", osp.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, osp.join(REPO_ROOT, "tracksterLinker"))


NODE_FEATURE_KEYS = [
    "barycenter_x",
    "barycenter_y",
    "barycenter_z",
    "barycenter_eta",
    "barycenter_phi",
    "eVector0_x",
    "eVector0_y",
    "eVector0_z",
    "EV1",
    "EV2",
    "EV3",
    "sigmaPCA1",
    "sigmaPCA2",
    "sigmaPCA3",
    "num_LCs",
    "num_hits",
    "raw_energy",
    "raw_em_energy",
    "photon_prob",
    "electron_prob",
    "muon_prob",
    "neutral_pion_prob",
    "charged_hadron_prob",
    "neutral_hadron_prob",
    "z_min",
    "z_max",
    "LC_density",
    "trackster_density",
    "time",
]
NODE_FEATURE_DICT = {key: idx for idx, key in enumerate(NODE_FEATURE_KEYS)}


def install_analysis_import_stubs():
    """Allow checkpoint-only analysis on machines without CuPy/uproot installed."""
    if "cupy" not in sys.modules:
        try:
            __import__("cupy")
        except ModuleNotFoundError:
            cupy_stub = types.ModuleType("cupy")
            cupy_stub.pi = np.pi
            cupy_stub.newaxis = np.newaxis
            cupy_stub.ndarray = np.ndarray
            sys.modules["cupy"] = cupy_stub

    if "uproot" not in sys.modules:
        try:
            __import__("uproot")
        except ModuleNotFoundError:
            sys.modules["uproot"] = types.ModuleType("uproot")


def read_json(path):
    if not osp.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sort_key(path):
    stem = osp.splitext(osp.basename(path))[0]
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return stem


def latest_checkpoint(model_folder, model_name):
    model_dir = osp.join(model_folder, model_name)
    candidates = sorted(glob(osp.join(model_dir, "*_dict.pt")), key=osp.getmtime)
    if not candidates:
        candidates = sorted(glob(osp.join(model_dir, "*.pt")), key=osp.getmtime)
        candidates = [
            path
            for path in candidates
            if not path.endswith("_traced.pt") and not path.endswith("_diff_traced.pt")
        ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {model_dir}")
    return candidates[-1]


def default_base_folder():
    parent_data = osp.abspath(osp.join(REPO_ROOT, "..", "data"))
    repo_data = osp.abspath(osp.join(REPO_ROOT, "data"))
    if osp.isdir(parent_data):
        return parent_data
    if osp.isdir(repo_data):
        return repo_data
    return parent_data


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep dummy-reconstruction thresholds for a trained model.")
    parser.add_argument("--base-folder", default=default_base_folder())
    parser.add_argument("--run-name", default="dummy_reco_experiment")
    parser.add_argument("--model-name", default="focal_contrastive")
    parser.add_argument("--model-folder", default=None)
    parser.add_argument("--data-folder", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument("--limit-graphs", type=int, default=None)
    parser.add_argument("--selection", default="signal", choices=["signal", "all"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def split_folder_name(split):
    if split == "train":
        return "dataset_dummy_reco"
    return f"dataset_dummy_reco_{split}"


def build_model(architecture, state_dict, device):
    install_analysis_import_stubs()
    from tracksterLinker.GNN.TrackLinkingNet import GNN_TrackLinkingNet
    from tracksterLinker.multiGNN.PUNet import PUNet

    input_dim = int(state_dict["node_scaler"].numel())
    edge_dim = int(state_dict["edge_scaler"].numel())
    kwargs = {
        "input_dim": input_dim,
        "edge_feature_dim": edge_dim,
        "niters": 2,
        "weighted_aggr": True,
        "dropout": 0.3,
        "node_scaler": state_dict["node_scaler"].detach().clone(),
        "edge_scaler": state_dict["edge_scaler"].detach().clone(),
    }
    if architecture == "punet":
        model = PUNet(edge_hidden_dim=16, hidden_dim=32, num_heads=2, **kwargs)
    elif architecture == "gnn":
        model = GNN_TrackLinkingNet(edge_hidden_dim=32, hidden_dim=64, **kwargs)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def calc_weights(edge_index, x, node_feature_dict, name="raw_energy"):
    feature_index = node_feature_dict[name]
    return torch.maximum(x[edge_index[:, 0], feature_index], x[edge_index[:, 1], feature_index]).abs()


def load_graph_records(model, graph_files, device, limit_graphs=None):
    from tracksterLinker.utils.reco_metrics import (
        edge_classification_metrics,
        reconstruction_metrics_from_components,
    )

    records = []
    all_scores = []
    all_labels = []
    all_weights = []

    selected_files = graph_files[:limit_graphs] if limit_graphs is not None else graph_files
    for graph_idx, graph_file in enumerate(selected_files, start=1):
        print(f"Scoring graph {graph_idx}/{len(selected_files)}: {osp.basename(graph_file)}")
        sample = torch.load(graph_file, map_location=device, weights_only=False).to(device)
        with torch.no_grad():
            _, logits = model.run(sample.x, sample.edge_features, sample.edge_index)
            scores = model.scale(logits).squeeze(-1)
            weights = calc_weights(sample.edge_index, sample.x, NODE_FEATURE_DICT, name="raw_energy")

        records.append(
            {
                "scores": scores.detach().cpu(),
                "edge_index": sample.edge_index.detach().cpu(),
                "num_nodes": int(sample.x.shape[0]),
                "cluster": sample.cluster.detach().cpu(),
                "energy": sample.x[:, NODE_FEATURE_DICT["raw_energy"]].detach().cpu(),
                "isPU": sample.isPU.detach().cpu() if hasattr(sample, "isPU") else None,
            }
        )
        all_scores.append(scores.detach().cpu())
        all_labels.append(sample.y.detach().cpu())
        all_weights.append(weights.detach().cpu())

    # Keep imports live in this function so static tools know these utilities are intentionally used.
    _ = edge_classification_metrics, reconstruction_metrics_from_components
    return records, torch.cat(all_scores), torch.cat(all_labels), torch.cat(all_weights)


def metrics_for_thresholds(records, all_scores, all_labels, all_weights, thresholds, selection):
    from tracksterLinker.utils.reco_metrics import (
        aggregate_metric_dicts,
        connected_components_from_edges,
        edge_classification_metrics,
        reconstruction_metrics_from_components,
    )

    rows = []
    for threshold in thresholds:
        print(f"Evaluating threshold {threshold:.3f}")
        row = edge_classification_metrics(all_scores, all_labels, all_weights, float(threshold))
        reco_rows = []
        for record in records:
            components = connected_components_from_edges(
                record["edge_index"],
                record["num_nodes"],
                record["scores"] >= threshold,
            )
            reco_rows.append(
                reconstruction_metrics_from_components(
                    components,
                    record["cluster"],
                    record["energy"],
                    is_pu=record["isPU"],
                    selection=selection,
                )
            )
        row.update(aggregate_metric_dicts(reco_rows))
        rows.append(row)
    return rows


def write_outputs(rows, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    json_path = osp.join(output_dir, f"{model_name}_threshold_sweep.json")
    csv_path = osp.join(output_dir, f"{model_name}_threshold_sweep.csv")
    plot_path = osp.join(output_dir, f"{model_name}_threshold_sweep.png")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    keys = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    thresholds = [row["threshold"] for row in rows]
    metrics = ["edge_f1", "b3_f1", "mean_best_iou", "energy_weighted_iou", "fake_rate", "merge_rate"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in metrics:
        ax.plot(thresholds, [row.get(metric, np.nan) for row in rows], marker="o", label=metric)
    ax.set_xlabel("edge threshold")
    ax.set_ylabel("metric")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return json_path, csv_path, plot_path


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_folder = args.model_folder or osp.join(args.base_folder, "training_data", args.run_name)
    data_folder = args.data_folder or osp.join(args.base_folder, "linking_dataset", args.run_name)
    output_dir = args.output_dir or osp.join(model_folder, args.model_name, "threshold_sweep")

    checkpoint_path = args.checkpoint or latest_checkpoint(model_folder, args.model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    metadata = read_json(osp.join(osp.dirname(checkpoint_path), "metadata.json"))
    config = read_json(osp.join(model_folder, "config.json"))
    architecture = checkpoint.get("architecture") or metadata.get("architecture") or config.get("architecture", "gnn")
    model = build_model(architecture, state_dict, device)

    processed_dir = osp.join(data_folder, split_folder_name(args.split), "processed")
    graph_files = sorted(glob(osp.join(processed_dir, "data_*.pt")), key=file_sort_key)
    if not graph_files:
        raise FileNotFoundError(f"No processed graph files found in {processed_dir}")

    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + 0.5 * args.threshold_step,
        args.threshold_step,
    )
    records, all_scores, all_labels, all_weights = load_graph_records(
        model,
        graph_files,
        device,
        limit_graphs=args.limit_graphs,
    )
    rows = metrics_for_thresholds(records, all_scores, all_labels, all_weights, thresholds, args.selection)
    json_path, csv_path, plot_path = write_outputs(rows, output_dir, args.model_name)

    best_edge = max(rows, key=lambda row: row.get("edge_f1", float("-inf")))
    best_b3 = max(rows, key=lambda row: row.get("b3_f1", float("-inf")))
    best_iou = max(rows, key=lambda row: row.get("energy_weighted_iou", float("-inf")))
    print(f"Best edge_f1 threshold: {best_edge['threshold']:.3f} edge_f1={best_edge['edge_f1']:.4f}")
    print(f"Best b3_f1 threshold: {best_b3['threshold']:.3f} b3_f1={best_b3['b3_f1']:.4f}")
    print(
        "Best energy_weighted_iou threshold: "
        f"{best_iou['threshold']:.3f} energy_weighted_iou={best_iou['energy_weighted_iou']:.4f}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
