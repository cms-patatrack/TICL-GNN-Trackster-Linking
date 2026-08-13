#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import types
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


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
NODE_FEATURE = {name: index for index, name in enumerate(NODE_FEATURE_KEYS)}
AXIS_LABELS = {
    "eta": r"$\eta$",
    "abs_eta": r"$|\eta|$",
    "phi": r"$\phi$",
    "z": "z [cm]",
    "abs_z": r"$|z|$ [cm]",
    "energy": "raw energy [GeV]",
}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: Path
    threshold: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved GNN checkpoints on processed dummy graphs and write binned performance curves."
    )
    parser.add_argument(
        "--dataset",
        default="/Users/chrisizeh/Documents/PhD/data/gnn_dataset/closeby_multi_0pu_test",
        help="Processed dummy graph dataset directory containing processed/data_*.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default="GNN_Paper/images/gnn_dummy_binned_performance",
        help="Directory for CSV/JSON/PNG outputs.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of graphs for a quick check.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument(
        "--focal-checkpoint",
        default=None,
        help="Override focal checkpoint path. Defaults to the archived dummy_reco_20sig_rootlike checkpoint.",
    )
    parser.add_argument(
        "--contrastive-checkpoint",
        default=None,
        help="Override focal+contrastive checkpoint path. Defaults to the archived dummy_reco_20sig_rootlike checkpoint.",
    )
    parser.add_argument("--focal-threshold", type=float, default=None, help="Override focal checkpoint threshold.")
    parser.add_argument("--contrastive-threshold", type=float, default=None, help="Override focal+contrastive checkpoint threshold.")
    parser.add_argument("--auto-threshold", action="store_true", help="Scan thresholds on this dataset and maximize weighted edge F1.")
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--piecewise-threshold-axis",
        choices=["energy", "eta", "abs_eta", "phi", "z", "abs_z"],
        default=None,
        help="Optimize multiple thresholds in bins of this representative edge coordinate.",
    )
    parser.add_argument("--piecewise-threshold-bins", type=int, default=2, help="Number of coordinate bins for piecewise thresholds.")
    parser.add_argument(
        "--piecewise-threshold-edges",
        default=None,
        help="Comma-separated explicit bin edges for the piecewise axis. Supports -inf/inf.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = Path(args.dataset).expanduser()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = [
        ModelSpec(
            "Focal",
            Path(args.focal_checkpoint).expanduser()
            if args.focal_checkpoint
            else repo_root / "data/dummy_reco_20sig_rootlike/focal/model_2026-07-09_epoch_50_dict.pt",
            args.focal_threshold if args.focal_threshold is not None else 0.800000011920929,
        ),
        ModelSpec(
            "Focal + contrastive",
            Path(args.contrastive_checkpoint).expanduser()
            if args.contrastive_checkpoint
            else repo_root
            / "data/dummy_reco_20sig_rootlike/focal_contrastive/model_2026-07-09_final_loss_-0.0006_epoch_48_dict.pt",
            args.contrastive_threshold if args.contrastive_threshold is not None else 0.699999988079071,
        ),
    ]
    data_paths = _data_paths(dataset_dir, args.limit)
    if not data_paths:
        raise SystemExit(f"No processed graphs found under {dataset_dir / 'processed'}")

    device = torch.device(args.device)
    model_class = _load_punet_class(repo_root)

    all_payloads: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    threshold_sweeps: dict[str, list[dict[str, Any]]] = {}
    piecewise_thresholds: dict[str, dict[str, Any]] = {}
    for spec in model_specs:
        checkpoint = torch.load(spec.checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        model = _instantiate_model_from_state(model_class, state_dict)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        threshold = spec.threshold
        threshold_map = None
        if args.auto_threshold or args.piecewise_threshold_axis:
            scores, labels, weights, coords = _collect_edge_scores(model, data_paths, device)
            thresholds = np.arange(
                args.threshold_min,
                args.threshold_max + 0.5 * args.threshold_step,
                args.threshold_step,
            )
        if args.auto_threshold:
            rows = _threshold_sweep(scores, labels, weights, thresholds)
            threshold_sweeps[spec.label] = rows
            threshold = float(max(rows, key=lambda row: row["edge_f1"])["threshold"])
            print(f"{spec.label}: selected threshold={threshold:.3f}", file=sys.stderr, flush=True)
        if args.piecewise_threshold_axis:
            threshold_map, rows = _piecewise_threshold_sweep(
                scores,
                labels,
                weights,
                coords[args.piecewise_threshold_axis],
                thresholds,
                axis=args.piecewise_threshold_axis,
                bins=args.piecewise_threshold_bins,
                explicit_edges=args.piecewise_threshold_edges,
            )
            piecewise_thresholds[spec.label] = threshold_map
            threshold_sweeps[f"{spec.label} piecewise {args.piecewise_threshold_axis}"] = rows
            pretty = ", ".join(f"[{row['low']:.3g}, {row['high']:.3g}): {row['threshold']:.3f}" for row in threshold_map["bins"])
            print(f"{spec.label}: selected piecewise {args.piecewise_threshold_axis} thresholds {pretty}", file=sys.stderr, flush=True)

        records = _evaluate_model(model, data_paths, threshold, device, threshold_map=threshold_map)
        payload = _make_payload(records, label=spec.label)
        all_payloads[spec.label] = payload
        summary_rows.extend(_summary_rows(spec.label, payload))

    if threshold_sweeps:
        _write_threshold_sweeps(output_dir, threshold_sweeps)

    _write_curves_csv(output_dir / "gnn_dummy_binned_curves.csv", all_payloads)
    _write_summary_csv(output_dir / "gnn_dummy_binned_summary.csv", summary_rows)
    _write_json(
        output_dir / "gnn_dummy_binned_payloads.json",
        {
            "dataset": str(dataset_dir),
            "graphs": len(data_paths),
            "axis_definition": "Edges are assigned to the higher-raw-energy endpoint. Components use raw-energy-weighted barycenters.",
            "models": {
                label: {
                    "checkpoint": str(next(spec.checkpoint for spec in model_specs if spec.label == label)),
                    "threshold": _selected_threshold(label, model_specs, threshold_sweeps),
                    "piecewise_threshold": piecewise_thresholds.get(label),
                    "archived_threshold": next(spec.threshold for spec in model_specs if spec.label == label),
                }
                for label in all_payloads
            },
            "threshold_sweeps": threshold_sweeps,
            "piecewise_thresholds": piecewise_thresholds,
            "payloads": all_payloads,
        },
    )
    plot_paths = _plot_payloads(output_dir, all_payloads)
    contact_path = _plot_contact_sheet(output_dir, plot_paths)
    print(
        json.dumps(
            {
                "dataset": str(dataset_dir),
                "graphs": len(data_paths),
                "output_dir": str(output_dir),
                "curves_csv": str(output_dir / "gnn_dummy_binned_curves.csv"),
                "summary_csv": str(output_dir / "gnn_dummy_binned_summary.csv"),
                "payload_json": str(output_dir / "gnn_dummy_binned_payloads.json"),
                "contact_sheet": str(contact_path),
                "plots": [str(path) for path in plot_paths],
            },
            indent=2,
        )
    )


def _load_punet_class(repo_root: Path):
    sys.modules.setdefault("cupy", np)
    for name in ("awkward", "uproot"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.path.insert(0, str(repo_root / "tracksterLinker"))
    from tracksterLinker.multiGNN.PUNet import PUNet

    return PUNet


def _instantiate_model_from_state(model_class, state_dict: dict[str, torch.Tensor]):
    hidden_dim = int(state_dict["inputnetwork.0.weight"].shape[0])
    input_dim = int(state_dict["inputnetwork.0.weight"].shape[1])
    edge_hidden_dim = int(state_dict["edge_inputnetwork.0.weight"].shape[0])
    edge_feature_dim = int(state_dict["edge_inputnetwork.0.weight"].shape[1])
    niters = 1 + max(
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("graphconvs.") and key.split(".")[1].isdigit()
    )
    num_layers = 1 + max(
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("encoder_layers.") and key.split(".")[1].isdigit()
    )
    num_heads = 8 if hidden_dim % 8 == 0 else 4 if hidden_dim % 4 == 0 else 2 if hidden_dim % 2 == 0 else 1
    return model_class(
        input_dim=input_dim,
        edge_feature_dim=edge_feature_dim,
        niters=niters,
        edge_hidden_dim=edge_hidden_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        weighted_aggr=True,
        dropout=0.3,
    )


def _data_paths(dataset_dir: Path, limit: int | None) -> list[Path]:
    def key(path: Path) -> int:
        match = re.search(r"data_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    paths = sorted((dataset_dir / "processed").glob("data_*.pt"), key=key)
    return paths if limit is None else paths[:limit]


def _evaluate_model(model, data_paths: list[Path], threshold: float, device: torch.device, threshold_map: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for event_id, path in enumerate(data_paths):
            if event_id and event_id % 25 == 0:
                print(f"  processed {event_id}/{len(data_paths)} graphs", file=sys.stderr, flush=True)
            sample = torch.load(path, weights_only=False, map_location=device)
            x = sample.x.to(device)
            edge_index = sample.edge_index.to(device)
            edge_features = sample.edge_features.to(device)
            y_true = (sample.y.to(device) > 0).bool()
            _, logits = model.run(x, edge_features, edge_index)
            scores = model.scale(logits).squeeze(-1)
            if threshold_map is None:
                y_pred = scores > threshold
            else:
                y_pred = scores > _edge_thresholds(x, edge_index, threshold_map, device)

            edge_records = _edge_records(event_id, x, edge_index, y_true, y_pred, scores)
            records.extend(edge_records)

            truth_components = _connected_components(edge_index[y_true], x.shape[0])
            pred_components = _connected_components(edge_index[y_pred], x.shape[0])
            records.extend(_component_records(event_id, x, truth_components, pred_components, object_type="truth"))
            records.extend(_component_records(event_id, x, pred_components, truth_components, object_type="reco"))
    return records


def _collect_edge_scores(model, data_paths: list[Path], device: torch.device):
    scores_out = []
    labels_out = []
    weights_out = []
    coords_out = defaultdict(list)
    with torch.inference_mode():
        for event_id, path in enumerate(data_paths):
            if event_id and event_id % 25 == 0:
                print(f"  scanned {event_id}/{len(data_paths)} graphs", file=sys.stderr, flush=True)
            sample = torch.load(path, weights_only=False, map_location=device)
            x = sample.x.to(device)
            edge_index = sample.edge_index.to(device)
            edge_features = sample.edge_features.to(device)
            _, logits = model.run(x, edge_features, edge_index)
            scores = model.scale(logits).squeeze(-1).detach().cpu().numpy()
            labels = (sample.y.detach().cpu().numpy() > 0)
            src = edge_index[:, 0]
            dst = edge_index[:, 1]
            energy = x[:, NODE_FEATURE["raw_energy"]].abs()
            weights_t = torch.maximum(energy[src], energy[dst]).clamp_min(0.0)
            rep = torch.where(energy[src] >= energy[dst], src, dst)
            eta = x[rep, NODE_FEATURE["barycenter_eta"]].detach().cpu().numpy()
            z = x[rep, NODE_FEATURE["barycenter_z"]].detach().cpu().numpy()
            weights = weights_t.detach().cpu().numpy()
            scores_out.append(scores)
            labels_out.append(labels)
            weights_out.append(weights)
            coords_out["energy"].append(weights)
            coords_out["eta"].append(eta)
            coords_out["abs_eta"].append(np.abs(eta))
            coords_out["phi"].append(x[rep, NODE_FEATURE["barycenter_phi"]].detach().cpu().numpy())
            coords_out["z"].append(z)
            coords_out["abs_z"].append(np.abs(z))
    return (
        np.concatenate(scores_out),
        np.concatenate(labels_out),
        np.concatenate(weights_out),
        {key: np.concatenate(values) for key, values in coords_out.items()},
    )


def _threshold_sweep(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, thresholds: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    weights = np.maximum(weights.astype(float), 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    labels = labels.astype(bool)
    for threshold in thresholds:
        pred = scores >= threshold
        tp = float(weights[labels & pred].sum())
        fp = float(weights[~labels & pred].sum())
        fn = float(weights[labels & ~pred].sum())
        tn = float(weights[~labels & ~pred].sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        rows.append(
            {
                "threshold": float(threshold),
                "edge_accuracy": _safe_div(tp + tn, tp + fp + fn + tn),
                "edge_precision": precision,
                "edge_recall": recall,
                "edge_f1": _safe_div(2 * precision * recall, precision + recall),
            }
        )
    return rows


def _piecewise_threshold_sweep(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    coord: np.ndarray,
    thresholds: np.ndarray,
    axis: str,
    bins: int,
    explicit_edges: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    edges = _piecewise_edges(coord, bins, explicit_edges)
    threshold_bins = []
    rows_out = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = _in_piecewise_bin(coord, low, high, is_last=index == len(edges) - 2)
        if not selected.any():
            threshold = float(thresholds[0])
            best = {
                "threshold": threshold,
                "edge_accuracy": 0.0,
                "edge_precision": 0.0,
                "edge_recall": 0.0,
                "edge_f1": 0.0,
            }
            rows = [best]
        else:
            rows = _threshold_sweep(scores[selected], labels[selected], weights[selected], thresholds)
            best = max(rows, key=lambda row: row["edge_f1"])
            threshold = float(best["threshold"])
        threshold_bins.append(
            {
                "bin": index,
                "axis": axis,
                "low": float(low),
                "high": float(high),
                "threshold": threshold,
                "entries": int(selected.sum()),
                "edge_f1": float(best["edge_f1"]),
            }
        )
        for row in rows:
            rows_out.append(
                {
                    "bin": index,
                    "axis": axis,
                    "low": float(low),
                    "high": float(high),
                    "entries": int(selected.sum()),
                    **row,
                }
            )
    return {"axis": axis, "bins": threshold_bins}, rows_out


def _piecewise_edges(coord: np.ndarray, bins: int, explicit_edges: str | None) -> np.ndarray:
    if bins < 1:
        raise ValueError("--piecewise-threshold-bins must be at least 1")
    if explicit_edges:
        values = []
        for item in explicit_edges.split(","):
            item = item.strip().lower()
            if item in {"inf", "+inf", "infinity", "+infinity"}:
                values.append(float("inf"))
            elif item in {"-inf", "-infinity"}:
                values.append(float("-inf"))
            else:
                values.append(float(item))
        edges = np.asarray(values, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError("--piecewise-threshold-edges must be strictly increasing")
        return edges
    finite = coord[np.isfinite(coord)]
    if finite.size == 0:
        return np.asarray([float("-inf"), float("inf")], dtype=float)
    internal = np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    internal = np.unique(internal)
    return np.asarray([float("-inf"), *internal.tolist(), float("inf")], dtype=float)


def _in_piecewise_bin(coord: np.ndarray, low: float, high: float, is_last: bool) -> np.ndarray:
    if is_last:
        return (coord >= low) & (coord <= high)
    return (coord >= low) & (coord < high)


def _edge_thresholds(x: torch.Tensor, edge_index: torch.Tensor, threshold_map: dict[str, Any], device: torch.device) -> torch.Tensor:
    axis = threshold_map["axis"]
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    energy = x[:, NODE_FEATURE["raw_energy"]].abs()
    rep = torch.where(energy[src] >= energy[dst], src, dst)
    if axis == "energy":
        coord = torch.maximum(energy[src], energy[dst]).clamp_min(0.0)
    elif axis == "eta":
        coord = x[rep, NODE_FEATURE["barycenter_eta"]]
    elif axis == "abs_eta":
        coord = x[rep, NODE_FEATURE["barycenter_eta"]].abs()
    elif axis == "phi":
        coord = x[rep, NODE_FEATURE["barycenter_phi"]]
    elif axis == "z":
        coord = x[rep, NODE_FEATURE["barycenter_z"]]
    elif axis == "abs_z":
        coord = x[rep, NODE_FEATURE["barycenter_z"]].abs()
    else:
        raise ValueError(axis)

    thresholds = torch.full_like(coord, float(threshold_map["bins"][-1]["threshold"]), dtype=torch.float32, device=device)
    for index, bin_spec in enumerate(threshold_map["bins"]):
        low = float(bin_spec["low"])
        high = float(bin_spec["high"])
        if index == len(threshold_map["bins"]) - 1:
            mask = (coord >= low) & (coord <= high)
        else:
            mask = (coord >= low) & (coord < high)
        thresholds[mask] = float(bin_spec["threshold"])
    return thresholds


def _edge_records(
    event_id: int,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    scores: torch.Tensor,
) -> list[dict[str, Any]]:
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    energy = x[:, NODE_FEATURE["raw_energy"]].abs()
    src_e = energy[src]
    dst_e = energy[dst]
    use_src = src_e >= dst_e
    representative = torch.where(use_src, src, dst)
    weights = torch.maximum(src_e, dst_e).clamp_min(0.0)
    out = []
    for index in range(edge_index.shape[0]):
        node = int(representative[index])
        out.append(
            {
                "event_id": event_id,
                "object_type": "edge",
                "eta": float(x[node, NODE_FEATURE["barycenter_eta"]].cpu()),
                "abs_eta": abs(float(x[node, NODE_FEATURE["barycenter_eta"]].cpu())),
                "phi": float(x[node, NODE_FEATURE["barycenter_phi"]].cpu()),
                "z": float(x[node, NODE_FEATURE["barycenter_z"]].cpu()),
                "abs_z": abs(float(x[node, NODE_FEATURE["barycenter_z"]].cpu())),
                "energy": float(weights[index].cpu()),
                "weight": float(weights[index].cpu()),
                "truth": bool(y_true[index].cpu()),
                "pred": bool(y_pred[index].cpu()),
                "score": float(scores[index].cpu()),
            }
        )
    return out


def _component_records(
    event_id: int,
    x: torch.Tensor,
    primary_components: list[list[int]],
    reference_components: list[list[int]],
    object_type: str,
) -> list[dict[str, Any]]:
    energy = x[:, NODE_FEATURE["raw_energy"]].abs()
    reference_sets = [set(component) for component in reference_components]
    out = []
    for component in primary_components:
        total_energy = float(energy[component].sum().cpu())
        if total_energy <= 0:
            continue
        best_overlap_energy = 0.0
        for ref in reference_sets:
            overlap = [node for node in component if node in ref]
            if overlap:
                best_overlap_energy = max(best_overlap_energy, float(energy[overlap].sum().cpu()))
        fraction = min(best_overlap_energy / total_energy, 1.0)
        features = _component_features(x, component)
        if object_type == "truth":
            out.append(
                {
                    "event_id": event_id,
                    "object_type": "truth",
                    **features,
                    "efficiency": float(fraction >= 0.40),
                    "sim_completeness": fraction,
                    "missing_energy_fraction": 1.0 - fraction,
                }
            )
        else:
            out.append(
                {
                    "event_id": event_id,
                    "object_type": "reco",
                    **features,
                    "reco_purity": fraction,
                    "fake_rate": float(fraction < 0.20),
                }
            )
    return out


def _component_features(x: torch.Tensor, component: list[int]) -> dict[str, float]:
    idx = torch.as_tensor(component, dtype=torch.long, device=x.device)
    energy = x[idx, NODE_FEATURE["raw_energy"]].abs().clamp_min(0.0)
    total = energy.sum()
    weights = energy / total if float(total.cpu()) > 0 else torch.full_like(energy, 1.0 / len(component))
    phi = x[idx, NODE_FEATURE["barycenter_phi"]]
    sin_phi = torch.sum(torch.sin(phi) * weights)
    cos_phi = torch.sum(torch.cos(phi) * weights)
    return {
        "eta": float(torch.sum(x[idx, NODE_FEATURE["barycenter_eta"]] * weights).cpu()),
        "abs_eta": abs(float(torch.sum(x[idx, NODE_FEATURE["barycenter_eta"]] * weights).cpu())),
        "phi": float(torch.atan2(sin_phi, cos_phi).cpu()),
        "z": float(torch.sum(x[idx, NODE_FEATURE["barycenter_z"]] * weights).cpu()),
        "abs_z": abs(float(torch.sum(x[idx, NODE_FEATURE["barycenter_z"]] * weights).cpu())),
        "energy": float(total.cpu()),
        "weight": float(total.cpu()),
        "n_nodes": len(component),
    }


def _connected_components(edges: torch.Tensor, num_nodes: int) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in edges.detach().cpu().tolist():
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    seen = [False] * num_nodes
    components: list[list[int]] = []
    for node in range(num_nodes):
        if seen[node]:
            continue
        seen[node] = True
        queue: deque[int] = deque([node])
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        components.append(component)
    return components


def _make_payload(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    bins = _default_bins(records)
    arrays = _records_to_arrays(records)
    curves: dict[str, dict[str, Any]] = {}
    curve_specs = {
        "accuracy": ("edge", "edge"),
        "precision": ("edge", "edge"),
        "recall": ("edge", "edge"),
        "f1": ("edge", "edge"),
        "efficiency": ("truth", "truth"),
        "sim_completeness": ("truth", "truth"),
        "missing_energy_fraction": ("truth", "truth"),
        "reco_purity": ("reco", "reco"),
        "fake_rate": ("reco", "reco"),
    }
    for metric, (record_type, curve_type) in curve_specs.items():
        curves[metric] = {}
        for axis, axis_bins in bins.items():
            curves[metric][axis] = _bin_metric_arrays(arrays[record_type], metric, axis, np.asarray(axis_bins), curve_type)
    return {"label": label, "axes": {axis: values.tolist() for axis, values in bins.items()}, "curves": curves}


def _records_to_arrays(records: list[dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    out = {
        "edge": defaultdict(list),
        "truth": defaultdict(list),
        "reco": defaultdict(list),
    }
    for record in records:
        target = out[record["object_type"]]
        for key, value in record.items():
            if key in {"event_id", "object_type"}:
                continue
            target[key].append(value)
    return {
        record_type: {key: np.asarray(values) for key, values in values_by_key.items()}
        for record_type, values_by_key in out.items()
    }


def _bin_metric_arrays(
    records: dict[str, np.ndarray],
    metric: str,
    axis: str,
    bins: np.ndarray,
    curve_type: str,
) -> dict[str, list[Any]]:
    centers = 0.5 * (bins[:-1] + bins[1:])
    axis_values = np.asarray(records.get(axis, []), dtype=float)
    if axis_values.size == 0:
        empty = [None for _ in centers]
        return {
            "bin_edges": bins.tolist(),
            "bin_centers": centers.tolist(),
            "values": empty,
            "errors": empty,
            "counts": [0 for _ in centers],
        }

    bin_indices = np.digitize(axis_values, bins) - 1
    values: list[float | None] = []
    errors: list[float | None] = []
    counts: list[int] = []
    for index in range(len(centers)):
        selected = bin_indices == index
        count = int(selected.sum())
        counts.append(count)
        if count == 0:
            values.append(None)
            errors.append(None)
            continue
        if curve_type == "edge":
            value = _edge_metric_arrays(records, metric, selected)
            error = _bootstrap_edge_error_arrays(records, metric, selected, value)
        else:
            weights = np.asarray(records.get("weight", np.ones_like(axis_values)), dtype=float)[selected]
            metric_values = np.asarray(records[metric], dtype=float)[selected]
            if weights.sum() <= 0:
                weights = np.ones_like(metric_values)
            value = float(np.average(metric_values, weights=weights))
            variance = float(np.average((metric_values - value) ** 2, weights=weights))
            error = math.sqrt(max(variance, 0.0) / count)
        values.append(value)
        errors.append(error)
    return {
        "bin_edges": bins.tolist(),
        "bin_centers": centers.tolist(),
        "values": values,
        "errors": errors,
        "counts": counts,
    }


def _edge_metric_arrays(records: dict[str, np.ndarray], metric: str, selected: np.ndarray) -> float:
    truth = np.asarray(records["truth"], dtype=bool)[selected]
    pred = np.asarray(records["pred"], dtype=bool)[selected]
    weights = np.asarray(records["weight"], dtype=float)[selected]
    weights = np.maximum(weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    tp = float(weights[truth & pred].sum())
    fp = float(weights[~truth & pred].sum())
    fn = float(weights[truth & ~pred].sum())
    tn = float(weights[~truth & ~pred].sum())
    if metric == "accuracy":
        return _safe_div(tp + tn, tp + fp + fn + tn)
    if metric == "precision":
        return _safe_div(tp, tp + fp)
    if metric == "recall":
        return _safe_div(tp, tp + fn)
    if metric == "f1":
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        return _safe_div(2 * precision * recall, precision + recall)
    raise ValueError(metric)


def _bootstrap_edge_error_arrays(records: dict[str, np.ndarray], metric: str, selected: np.ndarray, point: float) -> float:
    values = np.asarray(records["weight"], dtype=float)[selected]
    if len(values) <= 1:
        return 0.0
    n_eff = float(values.sum() ** 2 / np.square(values).sum()) if np.square(values).sum() > 0 else float(len(values))
    return math.sqrt(max(point * (1.0 - point), 0.0) / max(n_eff, 1.0))


def _bin_metric(
    records: list[dict[str, Any]],
    metric: str,
    axis: str,
    bins: np.ndarray,
    record_type: str,
    curve_type: str,
) -> dict[str, list[Any]]:
    centers = 0.5 * (bins[:-1] + bins[1:])
    values: list[float | None] = []
    errors: list[float | None] = []
    counts: list[int] = []
    for low, high in zip(bins[:-1], bins[1:]):
        selected = [
            record
            for record in records
            if record["object_type"] == record_type and low <= float(record[axis]) < high
        ]
        counts.append(len(selected))
        if not selected:
            values.append(None)
            errors.append(None)
            continue
        if curve_type == "edge":
            value = _edge_metric(selected, metric)
            error = _bootstrap_edge_error(selected, metric)
        else:
            weights = np.asarray([max(float(record.get("weight", 1.0)), 0.0) for record in selected], dtype=float)
            metric_values = np.asarray([float(record[metric]) for record in selected], dtype=float)
            if weights.sum() <= 0:
                weights = np.ones_like(metric_values)
            value = float(np.average(metric_values, weights=weights))
            variance = float(np.average((metric_values - value) ** 2, weights=weights))
            error = math.sqrt(max(variance, 0.0) / len(selected))
        values.append(value)
        errors.append(error)
    return {
        "bin_edges": bins.tolist(),
        "bin_centers": centers.tolist(),
        "values": values,
        "errors": errors,
        "counts": counts,
    }


def _edge_metric(records: list[dict[str, Any]], metric: str) -> float:
    truth = np.asarray([record["truth"] for record in records], dtype=bool)
    pred = np.asarray([record["pred"] for record in records], dtype=bool)
    weights = np.asarray([max(float(record["weight"]), 0.0) for record in records], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    tp = float(weights[truth & pred].sum())
    fp = float(weights[~truth & pred].sum())
    fn = float(weights[truth & ~pred].sum())
    tn = float(weights[~truth & ~pred].sum())
    if metric == "accuracy":
        return _safe_div(tp + tn, tp + fp + fn + tn)
    if metric == "precision":
        return _safe_div(tp, tp + fp)
    if metric == "recall":
        return _safe_div(tp, tp + fn)
    if metric == "f1":
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        return _safe_div(2 * precision * recall, precision + recall)
    raise ValueError(metric)


def _bootstrap_edge_error(records: list[dict[str, Any]], metric: str) -> float:
    values = np.asarray([float(record["weight"]) for record in records], dtype=float)
    if len(values) <= 1:
        return 0.0
    truth = np.asarray([record["truth"] for record in records], dtype=bool)
    pred = np.asarray([record["pred"] for record in records], dtype=bool)
    point = _edge_metric(records, metric)
    # Delta-method proxy: binomial-like uncertainty using effective weighted count.
    n_eff = float(values.sum() ** 2 / np.square(values).sum()) if np.square(values).sum() > 0 else float(len(values))
    return math.sqrt(max(point * (1.0 - point), 0.0) / max(n_eff, 1.0))


def _default_bins(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    z_values = np.asarray([float(record["z"]) for record in records if math.isfinite(float(record["z"]))], dtype=float)
    abs_z_values = np.asarray([float(record["abs_z"]) for record in records if math.isfinite(float(record["abs_z"]))], dtype=float)
    e_values = np.asarray(
        [float(record["energy"]) for record in records if math.isfinite(float(record["energy"])) and float(record["energy"]) > 0],
        dtype=float,
    )
    z_low = math.floor(float(z_values.min()) / 25.0) * 25.0 if z_values.size else -450.0
    z_high = math.ceil(float(z_values.max()) / 25.0) * 25.0 if z_values.size else 450.0
    if z_low == z_high:
        z_low -= 25.0
        z_high += 25.0
    abs_z_low = math.floor(float(abs_z_values.min()) / 25.0) * 25.0 if abs_z_values.size else 300.0
    abs_z_high = math.ceil(float(abs_z_values.max()) / 25.0) * 25.0 if abs_z_values.size else 550.0
    if abs_z_low == abs_z_high:
        abs_z_low -= 25.0
        abs_z_high += 25.0
    if e_values.size:
        e_low = 10.0 ** math.floor(math.log10(float(e_values.min())))
        e_high = 10.0 ** math.ceil(math.log10(float(e_values.max())))
    else:
        e_low, e_high = 0.1, 1.0
    return {
        "eta": np.linspace(-4.0, 4.0, 17),
        "abs_eta": np.linspace(1.5, 3.0, 13),
        "phi": np.linspace(-math.pi, math.pi, 17),
        "z": np.linspace(z_low, z_high, 19),
        "abs_z": np.linspace(abs_z_low, abs_z_high, 13),
        "energy": np.geomspace(e_low, e_high, 17),
    }


def _plot_payloads(output_dir: Path, payloads: dict[str, dict[str, Any]]) -> list[Path]:
    metric_labels = {
        "accuracy": "Edge accuracy",
        "precision": "Edge precision",
        "recall": "Edge recall",
        "f1": "Edge F1",
        "efficiency": "Reco efficiency",
        "sim_completeness": "Sim completeness",
        "missing_energy_fraction": "Missing energy fraction",
        "reco_purity": "Reco purity",
        "fake_rate": "Fake reco rate",
    }
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    paths: list[Path] = []
    for metric, ylabel in metric_labels.items():
        for axis, xlabel in AXIS_LABELS.items():
            fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
            for idx, (label, payload) in enumerate(payloads.items()):
                curve = payload["curves"][metric][axis]
                centers = np.asarray(curve["bin_centers"], dtype=float)
                values = _float_array(curve["values"])
                errors = _float_array(curve["errors"])
                counts = np.asarray(curve["counts"], dtype=int)
                valid = np.isfinite(values) & (counts > 0)
                if valid.any():
                    _plot_broken_errorbar(
                        ax,
                        centers,
                        values,
                        errors,
                        valid,
                        color=colors[idx % len(colors)],
                        label=label,
                        markersize=4.5,
                        linewidth=1.9,
                        capsize=2.8,
                    )
            ax.set_title(f"{ylabel} vs {axis}", fontsize=13)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_ylim(-0.03, 1.05)
            if axis == "energy":
                ax.set_xscale("log")
            ax.grid(True, alpha=0.28)
            ax.legend(frameon=False)
            fig.tight_layout()
            path = output_dir / f"gnn_dummy_{metric}_vs_{axis}.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)

    selected = [
        ("efficiency", "eta"),
        ("missing_energy_fraction", "abs_eta"),
        ("efficiency", "energy"),
        ("missing_energy_fraction", "energy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), dpi=180)
    for ax, (metric, axis) in zip(axes.ravel(), selected):
        for idx, (label, payload) in enumerate(payloads.items()):
            curve = payload["curves"][metric][axis]
            centers = np.asarray(curve["bin_centers"], dtype=float)
            values = _float_array(curve["values"])
            errors = _float_array(curve["errors"])
            counts = np.asarray(curve["counts"], dtype=int)
            valid = np.isfinite(values) & (counts > 0)
            _plot_broken_errorbar(
                ax,
                centers,
                values,
                errors,
                valid,
                color=colors[idx % len(colors)],
                label=label,
                markersize=4.2,
                linewidth=1.8,
                capsize=2.4,
            )
        ax.set_title(f"{metric.replace('_', ' ').title()} vs {axis}", fontsize=12)
        ax.set_xlabel(AXIS_LABELS[axis], fontsize=11)
        ax.set_ylabel(metric.replace("_", " "), fontsize=11)
        ax.set_ylim(-0.03, 1.05)
        if axis == "energy":
            ax.set_xscale("log")
        ax.grid(True, alpha=0.28)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    summary_path = output_dir / "gnn_dummy_eta_energy_story_panel.png"
    fig.savefig(summary_path)
    plt.close(fig)
    paths.append(summary_path)
    return paths


def _plot_broken_errorbar(
    ax,
    centers: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    valid: np.ndarray,
    *,
    color: str,
    label: str,
    markersize: float,
    linewidth: float,
    capsize: float,
) -> None:
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return
    start = 0
    first_segment = True
    for offset in range(1, valid_indices.size + 1):
        if offset == valid_indices.size or valid_indices[offset] != valid_indices[offset - 1] + 1:
            segment = valid_indices[start:offset]
            ax.errorbar(
                centers[segment],
                values[segment],
                yerr=errors[segment],
                marker="o",
                markersize=markersize,
                linewidth=linewidth,
                capsize=capsize,
                color=color,
                label=label if first_segment else None,
            )
            first_segment = False
            start = offset


def _plot_contact_sheet(output_dir: Path, plot_paths: list[Path]) -> Path:
    from PIL import Image, ImageDraw

    chosen = [
        path
        for path in plot_paths
        if path.name
        in {
            "gnn_dummy_efficiency_vs_eta.png",
            "gnn_dummy_efficiency_vs_abs_eta.png",
            "gnn_dummy_efficiency_vs_phi.png",
            "gnn_dummy_efficiency_vs_z.png",
            "gnn_dummy_efficiency_vs_abs_z.png",
            "gnn_dummy_efficiency_vs_energy.png",
            "gnn_dummy_missing_energy_fraction_vs_eta.png",
            "gnn_dummy_missing_energy_fraction_vs_abs_eta.png",
            "gnn_dummy_missing_energy_fraction_vs_phi.png",
            "gnn_dummy_missing_energy_fraction_vs_z.png",
            "gnn_dummy_missing_energy_fraction_vs_abs_z.png",
            "gnn_dummy_missing_energy_fraction_vs_energy.png",
            "gnn_dummy_f1_vs_eta.png",
            "gnn_dummy_f1_vs_abs_eta.png",
            "gnn_dummy_f1_vs_phi.png",
            "gnn_dummy_f1_vs_z.png",
            "gnn_dummy_f1_vs_abs_z.png",
            "gnn_dummy_f1_vs_energy.png",
        }
    ]
    tiles = []
    for path in chosen:
        img = Image.open(path).convert("RGB")
        img.thumbnail((360, 250))
        tile = Image.new("RGB", (380, 290), "white")
        tile.paste(img, ((380 - img.width) // 2, 6))
        draw = ImageDraw.Draw(tile)
        draw.text((12, 266), path.name.replace("gnn_dummy_", "").replace(".png", ""), fill=(20, 20, 20))
        tiles.append(tile)
    width, height = 4 * 380, math.ceil(len(tiles) / 4) * 290
    sheet = Image.new("RGB", (width, height), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 4) * 380, (index // 4) * 290))
    path = output_dir / "gnn_dummy_binned_contact_sheet.png"
    sheet.save(path)
    return path


def _summary_rows(label: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric, axes in payload["curves"].items():
        for axis, curve in axes.items():
            values = _float_array(curve["values"])
            counts = np.asarray(curve["counts"], dtype=int)
            valid = np.isfinite(values) & (counts > 0)
            if not valid.any():
                continue
            rows.append(
                {
                    "model": label,
                    "metric": metric,
                    "axis": axis,
                    "mean": float(np.average(values[valid], weights=counts[valid])),
                    "min": float(np.min(values[valid])),
                    "max": float(np.max(values[valid])),
                    "span": float(np.max(values[valid]) - np.min(values[valid])),
                    "nonempty_bins": int(valid.sum()),
                    "entries": int(counts[valid].sum()),
                }
            )
    return rows


def _write_curves_csv(path: Path, payloads: dict[str, dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "metric", "axis", "bin_low", "bin_high", "bin_center", "count", "value", "error"],
        )
        writer.writeheader()
        for label, payload in payloads.items():
            for metric, axes in payload["curves"].items():
                for axis, curve in axes.items():
                    edges = curve["bin_edges"]
                    for index, center in enumerate(curve["bin_centers"]):
                        writer.writerow(
                            {
                                "model": label,
                                "metric": metric,
                                "axis": axis,
                                "bin_low": edges[index],
                                "bin_high": edges[index + 1],
                                "bin_center": center,
                                "count": curve["counts"][index],
                                "value": curve["values"][index],
                                "error": curve["errors"][index],
                            }
                        )


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "axis", "mean", "min", "max", "span", "nonempty_bins", "entries"])
        writer.writeheader()
        writer.writerows(rows)


def _selected_threshold(label: str, model_specs: list[ModelSpec], threshold_sweeps: dict[str, list[dict[str, Any]]]) -> float:
    if label in threshold_sweeps:
        return float(max(threshold_sweeps[label], key=lambda row: row["edge_f1"])["threshold"])
    return float(next(spec.threshold for spec in model_specs if spec.label == label))


def _write_threshold_sweeps(output_dir: Path, threshold_sweeps: dict[str, list[dict[str, Any]]]) -> None:
    json_payload = {}
    csv_path = output_dir / "gnn_dummy_threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "bin",
                "axis",
                "low",
                "high",
                "entries",
                "threshold",
                "edge_accuracy",
                "edge_precision",
                "edge_recall",
                "edge_f1",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for label, rows in threshold_sweeps.items():
            json_payload[label] = rows
            for row in rows:
                writer.writerow({"model": label, **row})
    _write_json(output_dir / "gnn_dummy_threshold_sweep.json", json_payload)
    _plot_threshold_sweeps(output_dir / "gnn_dummy_threshold_sweep.png", threshold_sweeps)
    _plot_piecewise_thresholds(output_dir / "gnn_dummy_piecewise_thresholds.png", threshold_sweeps)


def _plot_threshold_sweeps(path: Path, threshold_sweeps: dict[str, list[dict[str, Any]]]) -> None:
    global_sweeps = {label: rows for label, rows in threshold_sweeps.items() if rows and "bin" not in rows[0]}
    if not global_sweeps:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=180, sharex=True)
    for label, rows in global_sweeps.items():
        thresholds = np.asarray([row["threshold"] for row in rows], dtype=float)
        for ax, metrics in zip(axes, (["edge_precision", "edge_recall", "edge_f1"], ["edge_accuracy", "edge_f1"])):
            for metric in metrics:
                values = np.asarray([row[metric] for row in rows], dtype=float)
                line_label = f"{label} {metric.replace('edge_', '')}"
                ax.plot(thresholds, values, linewidth=1.6, label=line_label)
        selected = max(rows, key=lambda row: row["edge_f1"])
        for ax in axes:
            ax.axvline(float(selected["threshold"]), linestyle="--", linewidth=1.2, alpha=0.5)
    for ax in axes:
        ax.set_xlabel("edge threshold")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(True, alpha=0.28)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("metric")
    axes[0].set_title("Precision/recall/F1")
    axes[1].set_title("Accuracy/F1")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_piecewise_thresholds(path: Path, threshold_sweeps: dict[str, list[dict[str, Any]]]) -> None:
    piecewise = {label: rows for label, rows in threshold_sweeps.items() if rows and "bin" in rows[0]}
    if not piecewise:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=180)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for index, (label, rows) in enumerate(piecewise.items()):
        best_by_bin = {}
        for row in rows:
            bin_index = int(row["bin"])
            if bin_index not in best_by_bin or row["edge_f1"] > best_by_bin[bin_index]["edge_f1"]:
                best_by_bin[bin_index] = row
        best_rows = [best_by_bin[key] for key in sorted(best_by_bin)]
        centers = np.asarray([_finite_bin_center(row["low"], row["high"], row["axis"]) for row in best_rows], dtype=float)
        thresholds = np.asarray([row["threshold"] for row in best_rows], dtype=float)
        f1 = np.asarray([row["edge_f1"] for row in best_rows], dtype=float)
        names = [f"{row['low']:.3g} to {row['high']:.3g}" for row in best_rows]
        color = colors[index % len(colors)]
        x = np.arange(len(best_rows))
        axes[0].plot(x, thresholds, marker="o", linewidth=1.8, color=color, label=label)
        axes[1].plot(x, f1, marker="o", linewidth=1.8, color=color, label=label)
        axes[0].set_xticks(x)
        axes[1].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=25, ha="right")
        axes[1].set_xticklabels(names, rotation=25, ha="right")
        axes[0].set_xlabel(best_rows[0]["axis"])
        axes[1].set_xlabel(best_rows[0]["axis"])
    axes[0].set_ylabel("selected threshold")
    axes[1].set_ylabel("weighted edge F1")
    axes[0].set_title("Piecewise thresholds")
    axes[1].set_title("Best F1 per threshold region")
    for ax in axes:
        ax.grid(True, alpha=0.28)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _finite_bin_center(low: float, high: float, axis: str) -> float:
    if math.isfinite(low) and math.isfinite(high):
        if axis == "energy" and low > 0 and high > 0:
            return math.sqrt(low * high)
        return 0.5 * (low + high)
    if math.isfinite(low):
        return low
    if math.isfinite(high):
        return high
    return 0.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _float_array(values: list[Any]) -> np.ndarray:
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


if __name__ == "__main__":
    main()
