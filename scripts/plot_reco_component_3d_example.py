#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_dummy_gnn_binned import (
    NODE_FEATURE,
    _connected_components,
    _data_paths,
    _instantiate_model_from_state,
    _load_punet_class,
    _matched_component_records,
)


@dataclass(frozen=True)
class ModelRun:
    label: str
    checkpoint: Path
    threshold: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the validation event where two reconstructions differ most and plot predicted 3D components."
    )
    parser.add_argument("--dataset", default="data/gnn_dataset/dataset_dummy_reco_val_combined")
    parser.add_argument("--output-dir", default="data/dummy_gnn_rootlike_20260814_145108/validation_examples")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--event-id", type=int, default=None, help="Plot this processed data_<id>.pt event instead of scanning for the max difference.")
    parser.add_argument("--signal-only", action="store_true", help="Only draw nodes with isPU == 0; component colors still come from the full reconstruction.")
    parser.add_argument("--positive-z-only", action="store_true", help="Only draw nodes with barycenter_z > 0.")
    parser.add_argument("--focal-checkpoint", default="data/dummy_gnn_rootlike_20260814_145108/focal/model_2026-08-15_epoch_30_dict.pt")
    parser.add_argument(
        "--contrastive-checkpoint",
        default="data/dummy_gnn_rootlike_20260814_145108/focal_contrastive/model_2026-08-15_epoch_30_dict.pt",
    )
    parser.add_argument("--focal-threshold", type=float, default=0.75)
    parser.add_argument("--contrastive-threshold", type=float, default=0.75)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = _resolve(repo_root, args.dataset)
    output_dir = _resolve(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_runs = [
        ModelRun("Focal", _resolve(repo_root, args.focal_checkpoint), args.focal_threshold),
        ModelRun("Focal + contrastive", _resolve(repo_root, args.contrastive_checkpoint), args.contrastive_threshold),
    ]
    data_paths = _data_paths(dataset_dir, args.limit)
    if not data_paths:
        raise SystemExit(f"No processed graphs found under {dataset_dir / 'processed'}")
    models = _load_models(repo_root, model_runs, torch.device(args.device))

    if args.event_id is None:
        event_rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        with torch.inference_mode():
            for event_id, path in enumerate(data_paths):
                sample = torch.load(path, weights_only=False, map_location=args.device)
                sample_result = _evaluate_sample(sample, models, model_runs, torch.device(args.device))
                summaries = {label: result["summary"] for label, result in sample_result.items()}
                score = _difference_score(summaries[model_runs[0].label], summaries[model_runs[1].label])
                row = {
                    "event_id": event_id,
                    "path": str(path),
                    "difference_score": score,
                    **_flat_summary(model_runs[0].label, summaries[model_runs[0].label]),
                    **_flat_summary(model_runs[1].label, summaries[model_runs[1].label]),
                }
                event_rows.append(row)
                if best is None or score > best["difference_score"]:
                    best = {
                        "event_id": event_id,
                        "path": path,
                        "difference_score": score,
                        "sample": sample,
                        "result": sample_result,
                    }
        assert best is not None
    else:
        if args.event_id < 0 or args.event_id >= len(data_paths):
            raise SystemExit(f"--event-id {args.event_id} is outside the available range 0..{len(data_paths) - 1}")
        event_rows = []
        path = data_paths[args.event_id]
        with torch.inference_mode():
            sample = torch.load(path, weights_only=False, map_location=args.device)
            sample_result = _evaluate_sample(sample, models, model_runs, torch.device(args.device))
        summaries = {label: result["summary"] for label, result in sample_result.items()}
        best = {
            "event_id": args.event_id,
            "path": path,
            "difference_score": _difference_score(summaries[model_runs[0].label], summaries[model_runs[1].label]),
            "sample": sample,
            "result": sample_result,
        }

    csv_path = output_dir / "event_difference_ranking.csv"
    if event_rows:
        _write_event_rows(csv_path, event_rows)
    suffix_parts = []
    if args.signal_only:
        suffix_parts.append("signal")
    if args.positive_z_only:
        suffix_parts.append("zpos")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    png_path = output_dir / f"event_{best['event_id']:03d}_reco_component_3d{suffix}.png"
    json_path = output_dir / f"event_{best['event_id']:03d}_reco_component_3d{suffix}.json"
    plot_counts = _plot_event(
        best["sample"],
        best["result"],
        model_runs,
        best["event_id"],
        best["difference_score"],
        png_path,
        signal_only=args.signal_only,
        positive_z_only=args.positive_z_only,
    )
    _write_json(
        json_path,
        {
            "dataset": str(dataset_dir),
            "event_id": best["event_id"],
            "event_path": str(best["path"]),
            "difference_score": best["difference_score"],
            "signal_only": args.signal_only,
            "positive_z_only": args.positive_z_only,
            "nodes_drawn": plot_counts,
            "difference_score_definition": (
                "|delta fragmentation| + |delta duplicate_rate| + |delta merge_rate| + "
                "0.02 * |delta n_reco_components|"
            ),
            "models": {
                run.label: {
                    "checkpoint": str(run.checkpoint),
                    "threshold": run.threshold,
                    "summary": best["result"][run.label]["summary"],
                }
                for run in model_runs
            },
            "outputs": {"png": str(png_path), "ranking_csv": str(csv_path) if csv_path.exists() else None},
        },
    )
    print(
        json.dumps(
            {
                "event_id": best["event_id"],
                "difference_score": best["difference_score"],
                "png": str(png_path),
                "json": str(json_path),
                "ranking_csv": str(csv_path) if csv_path.exists() else None,
                "signal_only": args.signal_only,
                "positive_z_only": args.positive_z_only,
                "nodes_drawn": plot_counts,
                "summaries": {run.label: best["result"][run.label]["summary"] for run in model_runs},
            },
            indent=2,
        )
    )


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _load_models(repo_root: Path, model_runs: list[ModelRun], device: torch.device) -> dict[str, torch.nn.Module]:
    model_class = _load_punet_class(repo_root)
    out = {}
    for run in model_runs:
        checkpoint = torch.load(run.checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        model = _instantiate_model_from_state(model_class, state_dict)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        out[run.label] = model
    return out


def _evaluate_sample(
    sample: Any,
    models: dict[str, torch.nn.Module],
    model_runs: list[ModelRun],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    x = sample.x.to(device)
    edge_index = sample.edge_index.to(device)
    edge_features = sample.edge_features.to(device)
    y_true = (sample.y.to(device) > 0).bool()
    truth_components = _connected_components(edge_index[y_true], x.shape[0])

    out = {}
    for run in model_runs:
        _, logits = models[run.label].run(x, edge_features, edge_index)
        scores = models[run.label].scale(logits).squeeze(-1)
        y_pred = scores > run.threshold
        pred_components = _connected_components(edge_index[y_pred], x.shape[0])
        records = _matched_component_records(0, x, truth_components, pred_components, NODE_FEATURE)
        out[run.label] = {
            "scores": scores.detach().cpu().numpy(),
            "y_pred": y_pred.detach().cpu().numpy(),
            "components": pred_components,
            "summary": _summarize_records(records, truth_components, pred_components),
        }
    return out


def _summarize_records(
    records: list[dict[str, Any]],
    truth_components: list[list[int]],
    pred_components: list[list[int]],
) -> dict[str, float]:
    truth = [record for record in records if record["object_type"] == "truth"]
    reco = [record for record in records if record["object_type"] == "reco"]
    return {
        "n_truth_components": float(len(truth_components)),
        "n_reco_components": float(len(pred_components)),
        "fragmentation": _weighted_mean(truth, "fragmentation"),
        "split_rate": _weighted_mean(truth, "split_rate"),
        "duplicate_rate": _weighted_mean(reco, "duplicate_rate"),
        "merge_rate": _weighted_mean(reco, "merge_rate"),
        "fake_rate": _weighted_mean(reco, "fake_rate"),
        "mean_reco_size": float(np.mean([len(component) for component in pred_components])) if pred_components else 0.0,
        "max_reco_size": float(max((len(component) for component in pred_components), default=0)),
    }


def _weighted_mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    values = np.asarray([float(record[key]) for record in records], dtype=float)
    weights = np.asarray([max(float(record.get("weight", 1.0)), 0.0) for record in records], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(values)
    return float(np.average(values, weights=weights))


def _difference_score(left: dict[str, float], right: dict[str, float]) -> float:
    return float(
        abs(left["fragmentation"] - right["fragmentation"])
        + abs(left["duplicate_rate"] - right["duplicate_rate"])
        + abs(left["merge_rate"] - right["merge_rate"])
        + 0.02 * abs(left["n_reco_components"] - right["n_reco_components"])
    )


def _flat_summary(label: str, summary: dict[str, float]) -> dict[str, float]:
    prefix = label.lower().replace(" + ", "_").replace(" ", "_")
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def _write_event_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["difference_score"], reverse=True))


def _plot_event(
    sample: Any,
    result: dict[str, dict[str, Any]],
    model_runs: list[ModelRun],
    event_id: int,
    difference_score: float,
    path: Path,
    *,
    signal_only: bool,
    positive_z_only: bool,
) -> dict[str, int]:
    x = sample.x.detach().cpu().numpy()
    draw_mask = _draw_mask(sample, signal_only, positive_z_only)
    coords = {
        "x": x[:, NODE_FEATURE["barycenter_x"]],
        "y": x[:, NODE_FEATURE["barycenter_y"]],
        "z": x[:, NODE_FEATURE["barycenter_z"]],
        "energy": np.maximum(np.abs(x[:, NODE_FEATURE["raw_energy"]]), 0.0),
    }
    sizes = _marker_sizes(coords["energy"])
    fig = plt.figure(figsize=(15.6, 7.2), dpi=180)
    plot_counts: dict[str, int] = {}
    for index, run in enumerate(model_runs, start=1):
        ax = fig.add_subplot(1, len(model_runs), index, projection="3d")
        components = result[run.label]["components"]
        labels = _component_labels(components, x.shape[0])
        colors = _component_colors(labels, components, coords["energy"])
        active_components = int(np.unique(labels[draw_mask & (labels >= 0)]).size)
        plot_counts[run.label] = int(draw_mask.sum())
        ax.scatter(
            coords["z"][draw_mask],
            coords["x"][draw_mask],
            coords["y"][draw_mask],
            c=colors[draw_mask],
            s=sizes[draw_mask],
            alpha=0.86,
            linewidths=0.2,
            edgecolors="black",
            depthshade=False,
        )
        summary = result[run.label]["summary"]
        ax.set_title(
            "\n".join(
                [
                    run.label,
                    f"components={int(summary['n_reco_components'])}, merge={summary['merge_rate']:.3f}, duplicate={summary['duplicate_rate']:.3f}",
                    f"fragmentation={summary['fragmentation']:.3f}, shown components={active_components}",
                ]
            ),
            fontsize=11,
        )
        ax.set_xlabel("z [cm]")
        ax.set_ylabel("x [cm]")
        ax.set_zlabel("y [cm]")
        ax.view_init(elev=20, azim=-58)
        _set_equal_3d_axes(ax, coords["z"][draw_mask], coords["x"][draw_mask], coords["y"][draw_mask])
    scope_parts = []
    if signal_only:
        scope_parts.append("signal nodes")
    if positive_z_only:
        scope_parts.append("z > 0")
    scope = " and ".join(scope_parts) + " only" if scope_parts else "all nodes"
    fig.suptitle(
        f"Combined validation event {event_id}: largest reconstruction difference score {difference_score:.3f}\n"
        f"{scope}; each color is one reconstructed connected component; marker area scales with raw energy.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)
    return plot_counts


def _draw_mask(sample: Any, signal_only: bool, positive_z_only: bool) -> np.ndarray:
    num_nodes = int(sample.x.shape[0])
    mask = np.ones(num_nodes, dtype=bool)
    if signal_only:
        if not hasattr(sample, "isPU"):
            raise ValueError("Cannot use --signal-only because this sample has no isPU field")
        mask &= ~(sample.isPU.detach().cpu().bool().numpy().reshape(-1))
    if positive_z_only:
        z = sample.x[:, NODE_FEATURE["barycenter_z"]].detach().cpu().numpy()
        mask &= z > 0
    if not mask.any():
        raise ValueError("The selected draw filters removed every node")
    return mask


def _component_labels(components: list[list[int]], num_nodes: int) -> np.ndarray:
    order = sorted(range(len(components)), key=lambda idx: (-len(components[idx]), idx))
    labels = np.full(num_nodes, -1, dtype=int)
    for color_id, comp_idx in enumerate(order):
        labels[np.asarray(components[comp_idx], dtype=int)] = color_id
    return labels


def _component_colors(labels: np.ndarray, components: list[list[int]], energy: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("turbo", max(32, min(256, len(components) + 1)))
    colors = cmap((labels % cmap.N) / max(cmap.N - 1, 1))
    colors[labels < 0] = (0.65, 0.65, 0.65, 0.8)
    return colors


def _marker_sizes(energy: np.ndarray) -> np.ndarray:
    if energy.size == 0:
        return np.asarray([])
    finite = energy[np.isfinite(energy)]
    if finite.size == 0 or finite.max() <= 0:
        return np.full_like(energy, 18.0, dtype=float)
    scaled = np.sqrt(np.clip(energy, 0.0, np.percentile(finite, 98)) / max(np.percentile(finite, 98), 1e-9))
    return 10.0 + 95.0 * scaled


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
