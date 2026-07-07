import argparse
import json
import os
import os.path as osp
import sys
import tempfile
from glob import glob

os.environ.setdefault("MPLCONFIGDIR", osp.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
sys.path.insert(0, osp.join(REPO_ROOT, "tracksterLinker"))


POSTER_BLUE = "#0033A0"
POSTER_LIGHT_BLUE = "#6EA6D9"
POSTER_RED = "#B0405A"
POSTER_GRAY = "#F4F6F8"
TEXT_COLOR = "#1F2933"


# Dummy outputs default to ../data while keeping the existing training_data/linking_dataset layout.
base_folder = osp.abspath(osp.join(REPO_ROOT, "..", "data"))
run_name = "dummy_reco_experiment"
model_folder = osp.join(base_folder, "training_data", run_name)
data_folder = osp.join(base_folder, "linking_dataset", run_name)
raw_data_folder = osp.join(data_folder, "histo")
data_folder_test = osp.join(data_folder, "dataset_dummy_reco_test")
output_folder = osp.join(base_folder, "training_data", f"{run_name}_edge_stability")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create poster-ready edge-stability heatmaps for the dummy reconstruction "
            "experiment. The plots compare focal-only and focal+contrastive checkpoints "
            "under the same transverse PCA perturbations."
        )
    )
    parser.add_argument("--base-folder", default=base_folder, help="Root folder for dummy data and outputs. Default: ../data.")
    parser.add_argument("--run-name", default=run_name)
    parser.add_argument("--model-folder", "--work-dir", dest="model_folder", default=None, help="Override model/checkpoint folder.")
    parser.add_argument(
        "--data-folder",
        "--data-dir",
        dest="data_folder",
        default=None,
        help="Override dataset root.",
    )
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--processed-data-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="Override plot output folder. Default: <base-folder>/training_data/<run-name>_edge_stability.")
    parser.add_argument("--focal-checkpoint", default=None)
    parser.add_argument("--contrastive-checkpoint", default=None)
    parser.add_argument("--num-graphs", type=int, default=100)
    parser.add_argument("--num-perturbations", type=int, default=50)
    parser.add_argument("--bins", type=int, default=95)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, or cpu.")
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def dummy_data_paths(args):
    use_script_paths = args.base_folder == base_folder and args.run_name == run_name
    default_model_folder = model_folder if use_script_paths else osp.join(args.base_folder, "training_data", args.run_name)
    default_data_folder = data_folder if use_script_paths else osp.join(args.base_folder, "linking_dataset", args.run_name)
    selected_model_folder = args.model_folder or default_model_folder
    selected_data_folder = args.data_folder or default_data_folder
    raw_data_dir = args.raw_data_dir or (raw_data_folder if use_script_paths and args.data_folder is None else osp.join(selected_data_folder, "histo"))
    test_folder = (
        osp.join(args.processed_data_dir, "dataset_dummy_reco_test")
        if args.processed_data_dir is not None
        else data_folder_test if use_script_paths and args.data_folder is None
        else osp.join(selected_data_folder, "dataset_dummy_reco_test")
    )
    return {
        "model": selected_model_folder,
        "stability": args.output_dir or (output_folder if use_script_paths else osp.join(args.base_folder, "training_data", f"{args.run_name}_edge_stability")),
        "data": selected_data_folder,
        "raw": raw_data_dir,
        "test": test_folder,
    }


def read_json(path):
    if not osp.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def newest_checkpoint(folder, model_name):
    model_dir = osp.join(folder, model_name)
    candidates = sorted(glob(osp.join(model_dir, "*_dict.pt")), key=osp.getmtime)
    if not candidates:
        candidates = [
            path
            for path in sorted(glob(osp.join(model_dir, "*.pt")), key=osp.getmtime)
            if not path.endswith("_traced.pt") and not path.endswith("_diff_traced.pt")
        ]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under {model_dir}. "
            "Run scripts/run_dummy_reco_experiment.py first or pass an explicit checkpoint path."
        )
    return candidates[-1]


def build_model(architecture, input_dim, edge_dim, device):
    from tracksterLinker.GNN.TrackLinkingNet import GNN_TrackLinkingNet
    from tracksterLinker.multiGNN.PUNet import PUNet

    kwargs = {
        "input_dim": input_dim,
        "edge_feature_dim": edge_dim,
        "niters": 4,
        "weighted_aggr": True,
        "dropout": 0.3,
    }
    if architecture == "punet":
        return PUNet(edge_hidden_dim=64, hidden_dim=128, num_heads=8, **kwargs).to(device)
    if architecture == "gnn":
        return GNN_TrackLinkingNet(edge_hidden_dim=32, hidden_dim=64, **kwargs).to(device)
    raise ValueError(f"Unsupported architecture in checkpoint: {architecture}")


def load_model(checkpoint_path, sample, input_dim, device, args):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = read_json(osp.join(osp.dirname(checkpoint_path), "metadata.json"))
    paths = dummy_data_paths(args)
    config = read_json(osp.join(paths["model"], "config.json"))
    architecture = checkpoint.get("architecture") or metadata.get("architecture") or config.get("architecture", "gnn")
    model = build_model(architecture, input_dim, sample.edge_features.shape[1], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = checkpoint.get("threshold", metadata.get("threshold", model.threshold))
    model.threshold = float(threshold)
    model.eval()
    return model, checkpoint


def load_test_dataset(args):
    from tracksterLinker.datasets.DummyDataset import DummyDataset

    paths = dummy_data_paths(args)
    raw_data_dir = paths["raw"]
    processed_root = paths["test"]
    if not osp.isdir(raw_data_dir) and not osp.isdir(processed_root):
        raise FileNotFoundError(
            f"Could not find dummy data in {raw_data_dir} or processed data in {processed_root}. "
            "Run scripts/run_dummy_reco_experiment.py --generate-data first."
        )

    dataset = DummyDataset(
        processed_root,
        raw_data_dir,
        split="test",
        num_workers=args.num_workers,
        device=torch.device("cpu"),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No test graphs found in {processed_root}")
    return dataset, DummyDataset


def predict_edges(model, sample, node_features, threshold):
    scores = model(node_features, sample.edge_features, sample.edge_index).reshape(-1)
    return scores >= threshold


def append_records(records, key, x, y, gain, focal_flip, contrastive_flip, mask):
    if mask is None:
        mask_np = np.ones(len(gain), dtype=bool)
    else:
        mask_np = mask.detach().cpu().numpy().astype(bool)
    if not np.any(mask_np):
        return
    records[key]["x"].append(x[mask_np])
    records[key]["y"].append(y[mask_np])
    records[key]["gain"].append(gain[mask_np])
    records[key]["focal_flip"].append(focal_flip[mask_np])
    records[key]["contrastive_flip"].append(contrastive_flip[mask_np])


def concat_records(records):
    out = {}
    for key, values in records.items():
        out[key] = {}
        for name, chunks in values.items():
            if chunks:
                out[key][name] = np.concatenate(chunks)
            else:
                out[key][name] = np.array([], dtype=float)
    return out


def stability_records(args, dataset, dataset_cls, focal_model, contrastive_model, device):
    from tracksterLinker.utils.perturbations.inErrorBars import perturbate

    torch.manual_seed(args.seed)
    records = {
        "all": {"x": [], "y": [], "gain": [], "focal_flip": [], "contrastive_flip": []},
        "signal": {"x": [], "y": [], "gain": [], "focal_flip": [], "contrastive_flip": []},
    }

    x_idx = dataset_cls.node_feature_dict["barycenter_x"]
    y_idx = dataset_cls.node_feature_dict["barycenter_y"]
    n_graphs = min(args.num_graphs, len(dataset))

    for graph_idx in range(n_graphs):
        print(f"Stability graph {graph_idx + 1}/{n_graphs}")
        sample = dataset[graph_idx].to(device)
        if sample.edge_index.numel() == 0:
            continue

        with torch.no_grad():
            focal_base = predict_edges(focal_model, sample, sample.x, focal_model.threshold)
            contrastive_base = predict_edges(contrastive_model, sample, sample.x, contrastive_model.threshold)

            focal_flips = torch.zeros_like(focal_base, dtype=torch.float32)
            contrastive_flips = torch.zeros_like(contrastive_base, dtype=torch.float32)
            perturbed = perturbate(
                sample.x,
                num_samples=args.num_perturbations,
                with_z=True,
                device=device,
            )

            for perturbed_x in perturbed:
                focal_pred = predict_edges(focal_model, sample, perturbed_x, focal_model.threshold)
                contrastive_pred = predict_edges(contrastive_model, sample, perturbed_x, contrastive_model.threshold)
                focal_flips += (focal_pred != focal_base).float()
                contrastive_flips += (contrastive_pred != contrastive_base).float()

        focal_flip_rate = (focal_flips / max(1, args.num_perturbations)).detach().cpu().numpy()
        contrastive_flip_rate = (contrastive_flips / max(1, args.num_perturbations)).detach().cpu().numpy()
        gain = focal_flip_rate - contrastive_flip_rate

        src = sample.edge_index[:, 0]
        dst = sample.edge_index[:, 1]
        edge_x = ((sample.x[src, x_idx] + sample.x[dst, x_idx]) / 2).detach().cpu().numpy()
        edge_y = ((sample.x[src, y_idx] + sample.x[dst, y_idx]) / 2).detach().cpu().numpy()
        signal_mask = sample.PU_info[:, 1] if hasattr(sample, "PU_info") else None

        append_records(records, "all", edge_x, edge_y, gain, focal_flip_rate, contrastive_flip_rate, None)
        append_records(records, "signal", edge_x, edge_y, gain, focal_flip_rate, contrastive_flip_rate, signal_mask)

    return concat_records(records)


def symmetric_range(values, default=250.0):
    if values.size == 0:
        return (-default, default)
    limit = float(np.nanpercentile(np.abs(values), 99.0))
    limit = max(50.0, min(default, limit))
    return (-limit, limit)


def binned_mean(x, y, values, bins, x_range, y_range):
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x = x[finite]
    y = y[finite]
    values = values[finite]
    if len(values) == 0:
        return np.ma.masked_all((bins, bins)), x_range, y_range

    weighted, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[x_range, y_range],
        weights=values,
    )
    counts, _, _ = np.histogram2d(x, y, bins=bins, range=[x_range, y_range])
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = weighted / counts
    return np.ma.masked_where(counts.T == 0, mean.T), x_edges, y_edges


def plot_gain_heatmap(records, key, title, output_path, bins):
    x = records[key]["x"]
    y = records[key]["y"]
    gain = records[key]["gain"]
    focal = records[key]["focal_flip"]
    contrastive = records[key]["contrastive_flip"]

    x_range = symmetric_range(np.concatenate([records["all"]["x"], records["all"]["y"]]))
    y_range = x_range
    heatmap, x_edges, y_edges = binned_mean(x, y, gain, bins, x_range, y_range)

    cmap = LinearSegmentedColormap.from_list(
        "poster_stability_gain",
        [POSTER_RED, "#FFFFFF", POSTER_LIGHT_BLUE, POSTER_BLUE],
        N=256,
    )
    cmap.set_bad("#FFFFFF")
    finite_gain = gain[np.isfinite(gain)]
    if finite_gain.size:
        vmax = float(np.nanpercentile(np.abs(finite_gain), 98.0))
    else:
        vmax = 0.05
    vmax = max(0.03, vmax)

    mean_focal = float(np.nanmean(focal)) if focal.size else float("nan")
    mean_contrastive = float(np.nanmean(contrastive)) if contrastive.size else float("nan")
    mean_gain = float(np.nanmean(gain)) if gain.size else float("nan")

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    fig.patch.set_facecolor("white")
    im = ax.imshow(
        heatmap,
        origin="lower",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("edge midpoint x [cm]", color=TEXT_COLOR)
    ax.set_ylabel("edge midpoint y [cm]", color=TEXT_COLOR)
    ax.set_title(title, color=POSTER_BLUE, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#A8B0B8")

    summary = (
        f"mean flip: focal {mean_focal:.3f}, "
        f"contrastive {mean_contrastive:.3f}\n"
        f"mean gain {mean_gain:+.3f}"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=TEXT_COLOR,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": POSTER_GRAY, "edgecolor": "#CBD2D9"},
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("focal flip rate - contrastive flip rate", color=TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def write_summary(records, output_path, args, focal_checkpoint, contrastive_checkpoint):
    summary = {
        "model_folder": dummy_data_paths(args)["model"],
        "data_dir": dummy_data_paths(args)["data"],
        "focal_checkpoint": focal_checkpoint,
        "contrastive_checkpoint": contrastive_checkpoint,
        "num_graphs_requested": args.num_graphs,
        "num_perturbations": args.num_perturbations,
        "regions": {},
    }
    for key, values in records.items():
        gain = values["gain"]
        focal = values["focal_flip"]
        contrastive = values["contrastive_flip"]
        summary["regions"][key] = {
            "edges": int(len(gain)),
            "mean_focal_flip_rate": float(np.nanmean(focal)) if len(focal) else None,
            "mean_contrastive_flip_rate": float(np.nanmean(contrastive)) if len(contrastive) else None,
            "mean_stability_gain": float(np.nanmean(gain)) if len(gain) else None,
            "fraction_positive_gain": float(np.nanmean(gain > 0)) if len(gain) else None,
        }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paths = dummy_data_paths(args)
    os.makedirs(paths["stability"], exist_ok=True)

    focal_checkpoint = args.focal_checkpoint or newest_checkpoint(paths["model"], "focal")
    contrastive_checkpoint = args.contrastive_checkpoint or newest_checkpoint(paths["model"], "focal_contrastive")

    dataset, dataset_cls = load_test_dataset(args)
    sample = dataset[0]
    input_dim = len(dataset_cls.model_feature_keys)
    focal_model, _ = load_model(focal_checkpoint, sample, input_dim, device, args)
    contrastive_model, _ = load_model(contrastive_checkpoint, sample, input_dim, device, args)

    print(f"Using device: {device}")
    print(f"Focal checkpoint: {focal_checkpoint}")
    print(f"Contrastive checkpoint: {contrastive_checkpoint}")

    records = stability_records(args, dataset, dataset_cls, focal_model, contrastive_model, device)
    plot_gain_heatmap(
        records,
        "all",
        "All candidate edges: contrastive stability gain",
        osp.join(paths["stability"], "all_edge_stab.png"),
        args.bins,
    )
    plot_gain_heatmap(
        records,
        "signal",
        "Signal-only edges: contrastive stability gain",
        osp.join(paths["stability"], "signal_edge_stab.png"),
        args.bins,
    )
    write_summary(
        records,
        osp.join(paths["stability"], "dummy_edge_stability_summary.json"),
        args,
        focal_checkpoint,
        contrastive_checkpoint,
    )
    print(f"Saved stability plots to {paths['stability']}")


if __name__ == "__main__":
    main()
