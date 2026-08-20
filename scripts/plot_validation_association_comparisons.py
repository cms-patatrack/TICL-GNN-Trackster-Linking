#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_dummy_gnn_binned import (
    NODE_FEATURE,
    _best_device,
    _connected_components,
    _data_paths,
    _instantiate_model_from_state,
    _load_model_classes,
    _matched_component_records,
    _node_feature_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot validation-event 3D association comparisons: truth, focal, and focal+contrastive."
    )
    parser.add_argument(
        "--dataset",
        default="data/colliderml_ttbar_pu0_det11_14_graphutils/dataset_colliderml_reco_val",
        help="Processed validation dataset directory containing processed/data_*.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/colliderml_ttbar_pu0_det11_14_graphutils/association_comparisons",
        help="Output directory for event PNGs and summary JSON.",
    )
    parser.add_argument("--events", type=int, default=10, help="Number of validation events to plot.")
    parser.add_argument("--start", type=int, default=0, help="First processed event index to plot.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, mps, or cpu.")
    parser.add_argument(
        "--focal-checkpoint",
        default="data/colliderml_ttbar_pu0_det11_14_graphutils/focal/model_2026-08-20_epoch_30_dict.pt",
    )
    parser.add_argument(
        "--contrastive-checkpoint",
        default="data/colliderml_ttbar_pu0_det11_14_graphutils/focal_contrastive/model_2026-08-20_epoch_30_dict.pt",
    )
    parser.add_argument("--focal-threshold", type=float, default=0.05000000074505806)
    parser.add_argument("--contrastive-threshold", type=float, default=0.05000000074505806)
    parser.add_argument("--signal-only", action="store_true", help="Draw only nodes with isPU == 0 when available.")
    parser.add_argument("--positive-z-only", action="store_true", help="Draw only nodes with barycenter_z > 0.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = _resolve(repo_root, args.dataset)
    output_dir = _resolve(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _best_device(args.device)
    model_classes = _load_model_classes(repo_root)
    models = {
        "Truth": None,
        "Focal": _load_model(model_classes, _resolve(repo_root, args.focal_checkpoint), device),
        "Focal + contrastive": _load_model(model_classes, _resolve(repo_root, args.contrastive_checkpoint), device),
    }
    thresholds = {
        "Focal": float(args.focal_threshold),
        "Focal + contrastive": float(args.contrastive_threshold),
    }

    data_paths = _data_paths(dataset_dir, None)
    selected_paths = data_paths[args.start : args.start + args.events]
    if not selected_paths:
        raise SystemExit(f"No processed graphs found under {dataset_dir / 'processed'}")

    outputs = []
    with torch.inference_mode():
        for offset, path in enumerate(selected_paths):
            event_id = args.start + offset
            sample = torch.load(path, weights_only=False, map_location=device)
            _validate_sample_compatibility(sample, models, path)
            comparison = _event_associations(sample, models, thresholds, device)
            png_path = output_dir / f"event_{event_id:03d}_truth_focal_contrastive_associations.png"
            plot_counts = _plot_comparison(
                sample,
                comparison,
                event_id,
                png_path,
                signal_only=args.signal_only,
                positive_z_only=args.positive_z_only,
            )
            outputs.append(
                {
                    "event_id": event_id,
                    "event_path": str(path),
                    "png": str(png_path),
                    "nodes_drawn": plot_counts,
                    "summary": {name: values["summary"] for name, values in comparison.items() if name != "Truth"},
                }
            )
            print(f"wrote {png_path}", flush=True)

    summary_path = output_dir / "association_comparison_summary.json"
    _write_json(
        summary_path,
        {
            "dataset": str(dataset_dir),
            "events": len(outputs),
            "start": args.start,
            "signal_only": args.signal_only,
            "positive_z_only": args.positive_z_only,
            "models": {
                "Focal": {"checkpoint": str(_resolve(repo_root, args.focal_checkpoint)), "threshold": args.focal_threshold},
                "Focal + contrastive": {
                    "checkpoint": str(_resolve(repo_root, args.contrastive_checkpoint)),
                    "threshold": args.contrastive_threshold,
                },
            },
            "outputs": outputs,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "summary_json": str(summary_path), "events": len(outputs)}, indent=2))


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _load_model(model_classes: dict[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    model = _instantiate_model_from_state(model_classes, state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def _validate_sample_compatibility(sample: Any, models: dict[str, torch.nn.Module | None], path: Path) -> None:
    model = next((value for value in models.values() if value is not None), None)
    if model is None:
        return
    expected_nodes = _model_expected_dim(model, "node_scaler", "inputnetwork.0.weight")
    expected_edges = _model_expected_dim(model, "edge_scaler", "edge_inputnetwork.0.weight")
    actual_nodes = int(sample.x.shape[1])
    actual_edges = int(sample.edge_features.shape[1])
    errors = []
    if actual_nodes != expected_nodes:
        errors.append(f"node features: graph has {actual_nodes}, checkpoint expects {expected_nodes}")
    if actual_edges != expected_edges:
        errors.append(f"edge features: graph has {actual_edges}, checkpoint expects {expected_edges}")
    if errors:
        raise SystemExit(
            f"Processed graph {path} is not compatible with these checkpoints ({'; '.join(errors)}). "
            "Use the ColliderML validation split created with the same feature schema as the checkpoints."
        )


def _model_expected_dim(model: torch.nn.Module, scaler_name: str, weight_name: str) -> int:
    state = model.state_dict()
    if scaler_name in state:
        return int(state[scaler_name].numel())
    if weight_name in state:
        return int(state[weight_name].shape[1])
    raise ValueError(f"Could not infer expected dimension from model state: {scaler_name} or {weight_name}")


def _event_associations(
    sample: Any,
    models: dict[str, torch.nn.Module | None],
    thresholds: dict[str, float],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    feature_dict = _node_feature_dict(sample)
    x = sample.x.to(device)
    edge_index = sample.edge_index.to(device)
    edge_features = sample.edge_features.to(device)
    y_true = (sample.y.to(device) > 0).bool()
    truth_components = _connected_components(edge_index[y_true], x.shape[0])
    truth_labels = _component_labels(truth_components, x.shape[0])

    out: dict[str, dict[str, Any]] = {
        "Truth": {
            "labels": truth_labels,
            "components": truth_components,
            "summary": {"n_components": float(len(truth_components))},
        }
    }

    for name, model in models.items():
        if name == "Truth":
            continue
        assert model is not None
        _, logits = model.run(x, edge_features, edge_index)
        scores = model.scale(logits).squeeze(-1)
        y_pred = scores > thresholds[name]
        reco_components = _connected_components(edge_index[y_pred], x.shape[0])
        records = _matched_component_records(0, x, truth_components, reco_components, feature_dict)
        reco_labels = _truth_aligned_reco_labels(records, reco_components, truth_labels, x.shape[0])
        out[name] = {
            "labels": reco_labels,
            "components": reco_components,
            "summary": _summarize(records, truth_components, reco_components),
        }
    return out


def _truth_aligned_reco_labels(
    records: list[dict[str, Any]],
    reco_components: list[list[int]],
    truth_labels: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    reco_records = [record for record in records if record["object_type"] == "reco"]
    labels = np.full(num_nodes, -1, dtype=int)
    conflict_base = max(int(truth_labels.max()) + 1, 0) if truth_labels.size else 0
    for reco_id, component in enumerate(reco_components):
        nodes = np.asarray(component, dtype=int)
        truth_id = -1
        if reco_id < len(reco_records):
            truth_id = int(reco_records[reco_id].get("matched_truth_id", -1))
        component_truth = truth_labels[nodes]
        labels[nodes] = truth_id
        conflict_mask = (component_truth >= 0) & (component_truth != truth_id)
        if conflict_mask.any():
            labels[nodes[conflict_mask]] = conflict_base + component_truth[conflict_mask]
    return labels


def _summarize(
    records: list[dict[str, Any]],
    truth_components: list[list[int]],
    reco_components: list[list[int]],
) -> dict[str, float]:
    truth = [record for record in records if record["object_type"] == "truth"]
    reco = [record for record in records if record["object_type"] == "reco"]
    return {
        "n_truth_components": float(len(truth_components)),
        "n_reco_components": float(len(reco_components)),
        "efficiency": _weighted_mean(truth, "efficiency"),
        "sim_completeness": _weighted_mean(truth, "sim_completeness"),
        "fragmentation": _weighted_mean(truth, "fragmentation"),
        "split_rate": _weighted_mean(truth, "split_rate"),
        "reco_purity": _weighted_mean(reco, "reco_purity"),
        "fake_rate": _weighted_mean(reco, "fake_rate"),
        "merge_rate": _weighted_mean(reco, "merge_rate"),
        "duplicate_rate": _weighted_mean(reco, "duplicate_rate"),
    }


def _weighted_mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    values = np.asarray([float(record[key]) for record in records], dtype=float)
    weights = np.asarray([max(float(record.get("weight", 1.0)), 0.0) for record in records], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(values)
    return float(np.average(values, weights=weights))


def _plot_comparison(
    sample: Any,
    comparison: dict[str, dict[str, Any]],
    event_id: int,
    path: Path,
    *,
    signal_only: bool,
    positive_z_only: bool,
) -> dict[str, int]:
    feature_dict = _node_feature_dict(sample)
    x_tensor = sample.x.detach().cpu()
    eta = x_tensor[:, feature_dict["barycenter_eta"]]
    phi = x_tensor[:, feature_dict["barycenter_phi"]]
    z = x_tensor[:, feature_dict["barycenter_z"]]
    x_coord, y_coord = _eta_phi_to_xy(eta, phi, z)
    z_coord = z.numpy()
    x_coord = x_coord.numpy()
    y_coord = y_coord.numpy()
    energy = x_tensor[:, feature_dict["raw_energy"]].abs().numpy()
    sizes = _marker_sizes(energy)
    draw_mask = _draw_mask(sample, feature_dict, signal_only, positive_z_only)

    label_to_color = _label_colors([values["labels"] for values in comparison.values()])
    fig = plt.figure(figsize=(16.5, 5.4), dpi=180)
    plot_counts: dict[str, int] = {}
    names = list(comparison)
    for index, name in enumerate(names, start=1):
        ax = fig.add_subplot(1, len(names), index, projection="3d")
        labels = comparison[name]["labels"]
        colors = np.asarray([label_to_color[int(label)] for label in labels])
        ax.scatter(
            z_coord[draw_mask],
            y_coord[draw_mask],
            x_coord[draw_mask],
            c=colors[draw_mask],
            s=sizes[draw_mask],
            linewidth=0.15,
            edgecolors="black",
            alpha=0.86,
            depthshade=False,
        )
        active = int(np.unique(labels[draw_mask & (labels >= 0)]).size)
        plot_counts[name] = int(draw_mask.sum())
        ax.set_title(_panel_title(name, comparison[name]["summary"], active), fontsize=10)
        ax.set_xlabel("z [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_zlabel("x [cm]")
        ax.view_init(elev=22, azim=-64)
        _set_equal_3d_axes(ax, z_coord[draw_mask], y_coord[draw_mask], x_coord[draw_mask])

    scope = []
    if signal_only:
        scope.append("signal")
    if positive_z_only:
        scope.append("z > 0")
    scope_text = ", ".join(scope) if scope else "all nodes"
    fig.suptitle(
        f"Validation event {event_id}: truth-aligned associations ({scope_text})\n"
        "Reco colors indicate associated truth component; grey means unmatched.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return plot_counts


def _panel_title(name: str, summary: dict[str, float], active_components: int) -> str:
    if name == "Truth":
        return f"Truth\ncomponents={int(summary['n_components'])}, shown={active_components}"
    return "\n".join(
        [
            name,
            f"reco={int(summary['n_reco_components'])}, shown assoc={active_components}",
            f"comp={summary['sim_completeness']:.3f}, frag={summary['fragmentation']:.3f}, fake={summary['fake_rate']:.3f}",
        ]
    )


def _eta_phi_to_xy(eta: torch.Tensor, phi: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 2 * torch.arctan(torch.exp(-eta))
    radius = torch.abs(z) * torch.tan(theta)
    return radius * torch.cos(phi), radius * torch.sin(phi)


def _component_labels(components: list[list[int]], num_nodes: int) -> np.ndarray:
    labels = np.full(num_nodes, -1, dtype=int)
    for component_id, component in enumerate(components):
        labels[np.asarray(component, dtype=int)] = component_id
    return labels


def _label_colors(label_arrays: list[np.ndarray]) -> dict[int, tuple[float, float, float, float]]:
    labels = sorted(set(np.concatenate([np.asarray(values).ravel() for values in label_arrays]).tolist()) - {-1})
    cmap = plt.get_cmap("tab20", max(len(labels), 1))
    colors = {int(label): cmap(index % cmap.N) for index, label in enumerate(labels)}
    colors[-1] = (0.68, 0.68, 0.68, 0.35)
    return colors


def _marker_sizes(energy: np.ndarray) -> np.ndarray:
    finite = energy[np.isfinite(energy)]
    if finite.size == 0 or finite.max() <= 0:
        return np.full_like(energy, 14.0, dtype=float)
    high = max(float(np.percentile(finite, 98)), 1e-9)
    scaled = np.sqrt(np.clip(energy, 0.0, high) / high)
    return 8.0 + 55.0 * scaled


def _draw_mask(sample: Any, feature_dict: dict[str, int], signal_only: bool, positive_z_only: bool) -> np.ndarray:
    num_nodes = int(sample.x.shape[0])
    mask = np.ones(num_nodes, dtype=bool)
    if signal_only:
        if not hasattr(sample, "isPU"):
            raise ValueError("Cannot use --signal-only because this sample has no isPU field")
        mask &= ~(sample.isPU.detach().cpu().bool().numpy().reshape(-1))
    if positive_z_only:
        z = sample.x[:, feature_dict["barycenter_z"]].detach().cpu().numpy()
        mask &= z > 0
    if not mask.any():
        raise ValueError("The selected draw filters removed every node")
    return mask


def _set_equal_3d_axes(ax, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
    ranges = []
    centers = []
    for values in (xs, ys, zs):
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
        ranges.append(high - low)
        centers.append(0.5 * (low + high))
    radius = 0.5 * max(max(ranges), 1.0)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
