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
PIECEWISE_AXES = tuple(AXIS_LABELS)
EFFICIENCY_MATCH_FRACTION = 0.40
FAKE_RATE_PURITY_THRESHOLD = 0.20
ASSOCIATION_SCORE_THRESHOLD = 0.20


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
    parser.add_argument(
        "--auto-piecewise-threshold",
        action="store_true",
        help="Scan axes and bin counts, then use the smallest useful piecewise threshold map.",
    )
    parser.add_argument(
        "--piecewise-threshold-axes",
        default="energy,abs_eta,eta,phi,abs_z,z",
        help="Comma-separated axes to scan for --auto-piecewise-threshold, or 'all'.",
    )
    parser.add_argument(
        "--piecewise-threshold-bin-candidates",
        default="2,3,4",
        help="Comma-separated bin counts to scan for --auto-piecewise-threshold.",
    )
    parser.add_argument(
        "--piecewise-min-improvement",
        type=float,
        default=0.002,
        help="Minimum absolute weighted edge-F1 gain over the global threshold before using bins.",
    )
    parser.add_argument(
        "--piecewise-parsimony-tolerance",
        type=float,
        default=0.003,
        help="Prefer fewer bins if their weighted edge F1 is within this value of the best scan candidate.",
    )
    parser.add_argument(
        "--piecewise-min-bin-entries",
        type=int,
        default=100,
        help="Ignore automatic piecewise candidates with fewer edge entries in any bin.",
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
    model_classes = _load_model_classes(repo_root)

    all_payloads: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    threshold_sweeps: dict[str, list[dict[str, Any]]] = {}
    piecewise_thresholds: dict[str, dict[str, Any]] = {}
    piecewise_selections: dict[str, dict[str, Any]] = {}
    piecewise_selection_rows: list[dict[str, Any]] = []
    for spec in model_specs:
        checkpoint = torch.load(spec.checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        model = _instantiate_model_from_state(model_classes, state_dict)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        threshold = spec.threshold
        threshold_map = None
        if args.auto_threshold or args.piecewise_threshold_axis or args.auto_piecewise_threshold:
            scores, labels, weights, coords = _collect_edge_scores(model, data_paths, device)
            thresholds = np.arange(
                args.threshold_min,
                args.threshold_max + 0.5 * args.threshold_step,
                args.threshold_step,
            )
        if args.auto_threshold or args.auto_piecewise_threshold:
            rows = _threshold_sweep(scores, labels, weights, thresholds)
            threshold_sweeps[spec.label] = rows
            threshold = float(max(rows, key=lambda row: row["edge_f1"])["threshold"])
            print(f"{spec.label}: selected threshold={threshold:.3f}", file=sys.stderr, flush=True)
        if args.auto_piecewise_threshold:
            threshold_map, selection, rows_by_name = _select_piecewise_thresholds(
                scores,
                labels,
                weights,
                coords,
                thresholds,
                global_threshold=threshold,
                axes=_parse_piecewise_axes(args.piecewise_threshold_axis, args.piecewise_threshold_axes),
                bin_candidates=_parse_bin_candidates(args.piecewise_threshold_bin_candidates),
                min_improvement=args.piecewise_min_improvement,
                parsimony_tolerance=args.piecewise_parsimony_tolerance,
                min_bin_entries=args.piecewise_min_bin_entries,
            )
            piecewise_selections[spec.label] = selection
            piecewise_selection_rows.extend({"model": spec.label, **row} for row in selection["candidates"])
            threshold_sweeps.update({f"{spec.label} {name}": rows for name, rows in rows_by_name.items()})
            if threshold_map is not None:
                piecewise_thresholds[spec.label] = threshold_map
                pretty = ", ".join(
                    f"[{row['low']:.3g}, {row['high']:.3g}): {row['threshold']:.3f}"
                    for row in threshold_map["bins"]
                )
                print(
                    f"{spec.label}: selected auto piecewise {threshold_map['axis']} "
                    f"with {len(threshold_map['bins'])} bins ({pretty})",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"{spec.label}: kept global threshold; best piecewise gain "
                    f"{selection['best_improvement']:.4f} < required {args.piecewise_min_improvement:.4f}",
                    file=sys.stderr,
                    flush=True,
                )
        elif args.piecewise_threshold_axis:
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
    if piecewise_selection_rows:
        _write_piecewise_selection_csv(output_dir / "gnn_dummy_piecewise_selection.csv", piecewise_selection_rows)

    _write_curves_csv(output_dir / "gnn_dummy_binned_curves.csv", all_payloads)
    _write_summary_csv(output_dir / "gnn_dummy_binned_summary.csv", summary_rows)
    _write_json(
        output_dir / "gnn_dummy_binned_payloads.json",
        {
            "dataset": str(dataset_dir),
            "graphs": len(data_paths),
            "axis_definition": "Edges are assigned to the higher-raw-energy endpoint. Components use raw-energy-weighted barycenters.",
            "component_metric_definition": {
                "efficiency": f"truth component has best reco overlap >= {EFFICIENCY_MATCH_FRACTION:.2f} of truth energy",
                "fake_rate": f"reco component has best truth purity < {FAKE_RATE_PURITY_THRESHOLD:.2f} of reco energy",
                "association_efficiency": (
                    "truth component has at least one associated reco component using the TICL association score "
                    f"threshold {ASSOCIATION_SCORE_THRESHOLD:.2f}"
                ),
                "split_rate": "truth component has more than one associated reco component",
                "duplicate_rate": "truth component has more than one associated reco component, matching reconstruction_metrics duplicate_rate",
                "merge_rate": "reco component has more than one associated truth component",
                "fragmentation": "truth-side 1 - sum(associated containment^2), matching associated_fragmentation_* in TICL-Pipeline",
            },
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
            "piecewise_selections": piecewise_selections,
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


def _load_model_classes(repo_root: Path):
    sys.modules.setdefault("cupy", np)
    for name in ("awkward", "uproot"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.path.insert(0, str(repo_root / "tracksterLinker"))
    from tracksterLinker.GNN.TrackLinkingNet import GNN_TrackLinkingNet
    from tracksterLinker.multiGNN.PUNet import PUNet

    return {"gnn": GNN_TrackLinkingNet, "punet": PUNet}


def _instantiate_model_from_state(model_classes: dict[str, Any], state_dict: dict[str, torch.Tensor]):
    hidden_dim = int(state_dict["inputnetwork.0.weight"].shape[0])
    input_dim = int(state_dict["inputnetwork.0.weight"].shape[1])
    edge_hidden_dim = int(state_dict["edge_inputnetwork.0.weight"].shape[0])
    edge_feature_dim = int(state_dict["edge_inputnetwork.0.weight"].shape[1])
    niters = 1 + max(
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("graphconvs.") and key.split(".")[1].isdigit()
    )
    kwargs = {
        "input_dim": input_dim,
        "edge_feature_dim": edge_feature_dim,
        "niters": niters,
        "edge_hidden_dim": edge_hidden_dim,
        "hidden_dim": hidden_dim,
        "weighted_aggr": True,
        "dropout": 0.3,
    }
    if not any(key.startswith("encoder_layers.") for key in state_dict):
        return model_classes["gnn"](**kwargs)

    num_layers = 1 + max(
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("encoder_layers.") and key.split(".")[1].isdigit()
    )
    num_heads = 8 if hidden_dim % 8 == 0 else 4 if hidden_dim % 4 == 0 else 2 if hidden_dim % 2 == 0 else 1
    return model_classes["punet"](
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
            records.extend(_matched_component_records(event_id, x, truth_components, pred_components))
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


def _select_piecewise_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    coords: dict[str, np.ndarray],
    thresholds: np.ndarray,
    *,
    global_threshold: float,
    axes: list[str],
    bin_candidates: list[int],
    min_improvement: float,
    parsimony_tolerance: float,
    min_bin_entries: int,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    global_metrics = _weighted_binary_metrics(scores, labels, weights, np.full_like(scores, global_threshold, dtype=float))
    global_f1 = float(global_metrics["edge_f1"])
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    candidates = []
    for axis in axes:
        coord = coords.get(axis)
        if coord is None:
            continue
        for bins in bin_candidates:
            threshold_map, rows = _piecewise_threshold_sweep(
                scores,
                labels,
                weights,
                coord,
                thresholds,
                axis=axis,
                bins=bins,
                explicit_edges=None,
            )
            actual_bins = len(threshold_map["bins"])
            min_entries = min((int(row["entries"]) for row in threshold_map["bins"]), default=0)
            threshold_values = _piecewise_threshold_array(coord, threshold_map, fallback=global_threshold)
            metrics = _weighted_binary_metrics(scores, labels, weights, threshold_values)
            edge_f1 = float(metrics["edge_f1"])
            improvement = edge_f1 - global_f1
            valid = min_entries >= min_bin_entries and actual_bins > 1
            name = f"auto piecewise {axis} {actual_bins} bins"
            rows_by_name[name] = rows
            candidates.append(
                {
                    "axis": axis,
                    "requested_bins": int(bins),
                    "bins": int(actual_bins),
                    "min_entries": int(min_entries),
                    "valid": bool(valid),
                    "selected": False,
                    "thresholds": [float(row["threshold"]) for row in threshold_map["bins"]],
                    "edges": [float(threshold_map["bins"][0]["low"])]
                    + [float(row["high"]) for row in threshold_map["bins"]],
                    "global_threshold": float(global_threshold),
                    "global_edge_f1": global_f1,
                    "edge_accuracy": float(metrics["edge_accuracy"]),
                    "edge_precision": float(metrics["edge_precision"]),
                    "edge_recall": float(metrics["edge_recall"]),
                    "edge_f1": edge_f1,
                    "improvement": float(improvement),
                    "threshold_map": threshold_map,
                }
            )

    valid_candidates = [row for row in candidates if row["valid"] and row["improvement"] >= min_improvement]
    best_candidate = max(candidates, key=lambda row: row["edge_f1"], default=None)
    if not valid_candidates:
        selection = {
            "selected": False,
            "reason": "no_valid_improving_candidate",
            "global_threshold": float(global_threshold),
            "global_edge_f1": global_f1,
            "best_edge_f1": float(best_candidate["edge_f1"]) if best_candidate else global_f1,
            "best_improvement": float(best_candidate["improvement"]) if best_candidate else 0.0,
            "min_improvement": float(min_improvement),
            "parsimony_tolerance": float(parsimony_tolerance),
            "min_bin_entries": int(min_bin_entries),
            "candidates": _candidate_rows_for_output(candidates),
        }
        return None, selection, rows_by_name

    best_valid_f1 = max(float(row["edge_f1"]) for row in valid_candidates)
    close_enough = [
        row for row in valid_candidates if float(row["edge_f1"]) >= best_valid_f1 - max(parsimony_tolerance, 0.0)
    ]
    selected = min(
        close_enough,
        key=lambda row: (
            int(row["bins"]),
            -float(row["improvement"]),
            PIECEWISE_AXES.index(row["axis"]) if row["axis"] in PIECEWISE_AXES else len(PIECEWISE_AXES),
        ),
    )
    selected["selected"] = True
    selection = {
        "selected": True,
        "reason": "smallest_candidate_within_tolerance_of_best",
        "axis": selected["axis"],
        "bins": int(selected["bins"]),
        "global_threshold": float(global_threshold),
        "global_edge_f1": global_f1,
        "best_edge_f1": best_valid_f1,
        "selected_edge_f1": float(selected["edge_f1"]),
        "selected_improvement": float(selected["improvement"]),
        "min_improvement": float(min_improvement),
        "parsimony_tolerance": float(parsimony_tolerance),
        "min_bin_entries": int(min_bin_entries),
        "candidates": _candidate_rows_for_output(candidates),
    }
    return selected["threshold_map"], selection, rows_by_name


def _candidate_rows_for_output(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in candidates:
        out.append({key: value for key, value in row.items() if key != "threshold_map"})
    return out


def _weighted_binary_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float]:
    weights = np.maximum(weights.astype(float), 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    labels = labels.astype(bool)
    pred = scores >= thresholds
    tp = float(weights[labels & pred].sum())
    fp = float(weights[~labels & pred].sum())
    fn = float(weights[labels & ~pred].sum())
    tn = float(weights[~labels & ~pred].sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "edge_accuracy": _safe_div(tp + tn, tp + fp + fn + tn),
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": _safe_div(2 * precision * recall, precision + recall),
    }


def _piecewise_threshold_array(
    coord: np.ndarray,
    threshold_map: dict[str, Any],
    *,
    fallback: float,
) -> np.ndarray:
    values = np.full(coord.shape, float(fallback), dtype=float)
    for index, bin_spec in enumerate(threshold_map["bins"]):
        mask = _in_piecewise_bin(
            coord,
            float(bin_spec["low"]),
            float(bin_spec["high"]),
            is_last=index == len(threshold_map["bins"]) - 1,
        )
        values[mask] = float(bin_spec["threshold"])
    return values


def _parse_piecewise_axes(forced_axis: str | None, axes_text: str) -> list[str]:
    if forced_axis:
        return [forced_axis]
    if axes_text.strip().lower() == "all":
        return list(PIECEWISE_AXES)
    axes = []
    for item in axes_text.split(","):
        axis = item.strip()
        if not axis:
            continue
        if axis not in PIECEWISE_AXES:
            raise ValueError(f"Unknown piecewise threshold axis {axis!r}. Choices: {', '.join(PIECEWISE_AXES)}")
        axes.append(axis)
    if not axes:
        raise ValueError("--piecewise-threshold-axes did not contain any axes")
    return axes


def _parse_bin_candidates(text: str) -> list[int]:
    bins = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not bins:
        raise ValueError("--piecewise-threshold-bin-candidates did not contain any bin counts")
    if bins[0] < 2:
        raise ValueError("--piecewise-threshold-bin-candidates must be >= 2; the global threshold is the one-bin baseline")
    return bins


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


def _matched_component_records(
    event_id: int,
    x: torch.Tensor,
    truth_components: list[list[int]],
    reco_components: list[list[int]],
) -> list[dict[str, Any]]:
    energy = x[:, NODE_FEATURE["raw_energy"]].abs().detach().cpu().numpy()
    truth_index = _component_index(truth_components, len(energy))
    reco_index = _component_index(reco_components, len(energy))
    truth_totals = _component_totals(energy, truth_components)
    reco_totals = _component_totals(energy, reco_components)
    overlap = np.zeros((len(truth_components), len(reco_components)), dtype=float)
    valid = (truth_index >= 0) & (reco_index >= 0) & (energy > 0)
    np.add.at(overlap, (truth_index[valid], reco_index[valid]), energy[valid])
    association_match = _association_match_matrix(truth_components, reco_components, energy)

    if overlap.size:
        truth_best_overlap = overlap.max(axis=1)
        truth_best_reco = overlap.argmax(axis=1)
        reco_best_overlap = overlap.max(axis=0)
        reco_best_truth = overlap.argmax(axis=0)
    else:
        truth_best_overlap = np.zeros(len(truth_components), dtype=float)
        truth_best_reco = np.full(len(truth_components), -1, dtype=int)
        reco_best_overlap = np.zeros(len(reco_components), dtype=float)
        reco_best_truth = np.full(len(reco_components), -1, dtype=int)

    containment = np.divide(
        overlap,
        truth_totals[:, None],
        out=np.zeros_like(overlap),
        where=truth_totals[:, None] > 0,
    )
    associated_containment = np.where(association_match, containment, np.zeros_like(containment))
    truth_coverage = associated_containment.sum(axis=1) if associated_containment.size else np.zeros(len(truth_components), dtype=float)
    fragmentation = 1.0 - np.minimum(np.square(associated_containment).sum(axis=1), 1.0)
    split_counts = association_match.sum(axis=1) if association_match.size else np.zeros(len(truth_components), dtype=int)
    merge_counts = association_match.sum(axis=0) if association_match.size else np.zeros(len(reco_components), dtype=int)

    purity = np.divide(
        overlap,
        reco_totals[None, :],
        out=np.zeros_like(overlap),
        where=reco_totals[None, :] > 0,
    )
    reco_purity = purity.max(axis=0) if purity.size else np.zeros(len(reco_components), dtype=float)

    out: list[dict[str, Any]] = []
    for truth_id, component in enumerate(truth_components):
        total_energy = float(truth_totals[truth_id])
        if total_energy <= 0:
            continue
        fraction = min(float(truth_best_overlap[truth_id] / total_energy), 1.0)
        out.append(
            {
                "event_id": event_id,
                "object_type": "truth",
                **_component_features(x, component),
                "efficiency": float(fraction >= EFFICIENCY_MATCH_FRACTION),
                "association_efficiency": float(split_counts[truth_id] > 0),
                "sim_completeness": fraction,
                "missing_energy_fraction": float(1.0 - min(truth_coverage[truth_id], 1.0)),
                "split_rate": float(split_counts[truth_id] > 1),
                "duplicate_rate": float(split_counts[truth_id] > 1),
                "fragmentation": float(fragmentation[truth_id]),
                "associated_reco_count": int(split_counts[truth_id]),
                "matched_reco_id": int(truth_best_reco[truth_id]),
            }
        )

    for reco_id, component in enumerate(reco_components):
        total_energy = float(reco_totals[reco_id])
        if total_energy <= 0:
            continue
        truth_id = int(reco_best_truth[reco_id])
        is_merged = int(merge_counts[reco_id]) > 1
        out.append(
            {
                "event_id": event_id,
                "object_type": "reco",
                **_component_features(x, component),
                "reco_purity": float(min(reco_purity[reco_id], 1.0)),
                "fake_rate": float(merge_counts[reco_id] == 0),
                "merge_rate": float(is_merged),
                "associated_truth_count": int(merge_counts[reco_id]),
                "matched_truth_id": truth_id,
            }
        )
    return out


def _association_match_matrix(
    truth_components: list[list[int]],
    reco_components: list[list[int]],
    energy: np.ndarray,
) -> np.ndarray:
    if not truth_components or not reco_components:
        return np.zeros((len(truth_components), len(reco_components)), dtype=bool)

    energy2 = np.square(np.maximum(energy, 0.0))
    truth_totals2 = _component_totals(energy2, truth_components)
    reco_totals2 = _component_totals(energy2, reco_components)
    overlap2 = np.zeros((len(truth_components), len(reco_components)), dtype=float)
    reco_sets = [set(component) for component in reco_components]
    for truth_id, truth_component in enumerate(truth_components):
        truth_set = set(truth_component)
        for reco_id, reco_set in enumerate(reco_sets):
            shared = list(truth_set & reco_set)
            if shared:
                overlap2[truth_id, reco_id] = float(energy2[np.asarray(shared, dtype=int)].sum())

    sim_to_reco_score = 1.0 - np.divide(
        overlap2,
        truth_totals2[:, None],
        out=np.zeros_like(overlap2),
        where=truth_totals2[:, None] > 0,
    )
    reco_to_sim_score = 1.0 - np.divide(
        overlap2,
        reco_totals2[None, :],
        out=np.zeros_like(overlap2),
        where=reco_totals2[None, :] > 0,
    )
    return (sim_to_reco_score < ASSOCIATION_SCORE_THRESHOLD) | (reco_to_sim_score < ASSOCIATION_SCORE_THRESHOLD)


def _component_index(components: list[list[int]], num_nodes: int) -> np.ndarray:
    out = np.full(num_nodes, -1, dtype=int)
    for index, component in enumerate(components):
        out[np.asarray(component, dtype=int)] = index
    return out


def _component_totals(energy: np.ndarray, components: list[list[int]]) -> np.ndarray:
    totals = np.zeros(len(components), dtype=float)
    for index, component in enumerate(components):
        totals[index] = float(energy[np.asarray(component, dtype=int)].sum())
    return totals


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
        "split_rate": ("truth", "truth"),
        "fragmentation": ("truth", "truth"),
        "duplicate_rate": ("truth", "truth"),
        "reco_purity": ("reco", "reco"),
        "fake_rate": ("reco", "reco"),
        "merge_rate": ("reco", "reco"),
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
        "split_rate": "Split rate",
        "fragmentation": "Fragmentation",
        "reco_purity": "Reco purity",
        "fake_rate": "Fake reco rate",
        "merge_rate": "Merge rate",
        "duplicate_rate": "Duplicate rate",
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
        ("fragmentation", "abs_eta"),
        ("split_rate", "abs_eta"),
        ("merge_rate", "abs_eta"),
        ("duplicate_rate", "abs_eta"),
        ("merge_rate", "energy"),
        ("duplicate_rate", "energy"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(24.0, 7.6), dpi=180)
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

    paths.append(_plot_metric_means(output_dir, payloads, metric_labels, colors))
    return paths


def _plot_metric_means(
    output_dir: Path,
    payloads: dict[str, dict[str, Any]],
    metric_labels: dict[str, str],
    colors: list[str],
) -> Path:
    metric_groups = [
        ("Edge metrics", ["accuracy", "precision", "recall", "f1"]),
        (
            "Reco metrics",
            [
                "efficiency",
                "sim_completeness",
                "missing_energy_fraction",
                "split_rate",
                "fragmentation",
                "reco_purity",
                "fake_rate",
                "merge_rate",
                "duplicate_rate",
            ],
        ),
    ]
    labels = list(payloads)
    width = min(0.8 / max(len(labels), 1), 0.36)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), dpi=180)
    for ax, (title, metrics) in zip(axes, metric_groups):
        x = np.arange(len(metrics), dtype=float)
        for model_index, label in enumerate(labels):
            means = [_mean_metric_over_bins(payloads[label], metric) for metric in metrics]
            offset = (model_index - (len(labels) - 1) / 2.0) * width
            bars = ax.bar(
                x + offset,
                means,
                width=width,
                color=colors[model_index % len(colors)],
                alpha=0.86,
                label=label,
            )
            for bar, value in zip(bars, means):
                if not math.isfinite(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    min(value + 0.018, 1.03),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    rotation=90,
                )
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([metric_labels[metric] for metric in metrics], rotation=24, ha="right")
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Mean over non-empty bins")
        ax.grid(True, axis="y", alpha=0.28)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=max(len(labels), 1), loc="upper center")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = output_dir / "gnn_dummy_metric_means.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _mean_metric_over_bins(payload: dict[str, Any], metric: str) -> float:
    values_out = []
    weights_out = []
    for curve in payload["curves"][metric].values():
        values = _float_array(curve["values"])
        counts = np.asarray(curve["counts"], dtype=float)
        valid = np.isfinite(values) & (counts > 0)
        if valid.any():
            values_out.append(values[valid])
            weights_out.append(counts[valid])
    if not values_out:
        return float("nan")
    values = np.concatenate(values_out)
    weights = np.concatenate(weights_out)
    return float(np.average(values, weights=weights))


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
            "gnn_dummy_split_rate_vs_eta.png",
            "gnn_dummy_split_rate_vs_abs_eta.png",
            "gnn_dummy_split_rate_vs_phi.png",
            "gnn_dummy_split_rate_vs_z.png",
            "gnn_dummy_split_rate_vs_abs_z.png",
            "gnn_dummy_split_rate_vs_energy.png",
            "gnn_dummy_fragmentation_vs_eta.png",
            "gnn_dummy_fragmentation_vs_abs_eta.png",
            "gnn_dummy_fragmentation_vs_phi.png",
            "gnn_dummy_fragmentation_vs_z.png",
            "gnn_dummy_fragmentation_vs_abs_z.png",
            "gnn_dummy_fragmentation_vs_energy.png",
            "gnn_dummy_merge_rate_vs_eta.png",
            "gnn_dummy_merge_rate_vs_abs_eta.png",
            "gnn_dummy_merge_rate_vs_phi.png",
            "gnn_dummy_merge_rate_vs_z.png",
            "gnn_dummy_merge_rate_vs_abs_z.png",
            "gnn_dummy_merge_rate_vs_energy.png",
            "gnn_dummy_duplicate_rate_vs_eta.png",
            "gnn_dummy_duplicate_rate_vs_abs_eta.png",
            "gnn_dummy_duplicate_rate_vs_phi.png",
            "gnn_dummy_duplicate_rate_vs_z.png",
            "gnn_dummy_duplicate_rate_vs_abs_z.png",
            "gnn_dummy_duplicate_rate_vs_energy.png",
            "gnn_dummy_metric_means.png",
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


def _write_piecewise_selection_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "selected",
                "valid",
                "axis",
                "requested_bins",
                "bins",
                "min_entries",
                "global_threshold",
                "global_edge_f1",
                "edge_accuracy",
                "edge_precision",
                "edge_recall",
                "edge_f1",
                "improvement",
                "edges",
                "thresholds",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["edges"] = ",".join(_format_float(value) for value in row.get("edges", []))
            serializable["thresholds"] = ",".join(_format_float(value) for value in row.get("thresholds", []))
            writer.writerow(serializable)


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


def _format_float(value: Any) -> str:
    value = float(value)
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.8g}"


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


if __name__ == "__main__":
    main()
