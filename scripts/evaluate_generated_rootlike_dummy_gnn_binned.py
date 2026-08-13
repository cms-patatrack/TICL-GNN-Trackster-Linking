#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from evaluate_dummy_gnn_binned import (
    NODE_FEATURE,
    NODE_FEATURE_KEYS,
    _component_records,
    _connected_components,
    _edge_records,
    _make_payload,
    _plot_contact_sheet,
    _plot_payloads,
    _summary_rows,
    _write_curves_csv,
    _write_json,
    _write_summary_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ROOT-like dummy events in memory and evaluate GNN binned performance curves."
    )
    parser.add_argument("--events", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--signal-mean", type=float, default=20.0)
    parser.add_argument("--pu-mean", type=float, default=200.0)
    parser.add_argument("--close-pair-fraction", type=float, default=0.85)
    parser.add_argument("--output-dir", default="GNN_Paper/images/gnn_dummy_rootlike_binned_performance")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _generate_samples(
        events=args.events,
        seed=args.seed,
        signal_mean=args.signal_mean,
        pu_mean=args.pu_mean,
        close_pair_fraction=args.close_pair_fraction,
    )
    models = {
        "Focal": {
            "path": repo_root / "data/dummy_reco_20sig_rootlike/focal/model_2026-07-09_traced.pt",
            "threshold": 0.800000011920929,
        },
        "Focal + contrastive": {
            "path": repo_root
            / "data/dummy_reco_20sig_rootlike/focal_contrastive/model_2026-07-09_final_loss_-0.0006_traced.pt",
            "threshold": 0.699999988079071,
        },
    }

    device = torch.device(args.device)
    payloads = {}
    summary_rows = []
    scalar_summary = {}
    for label, spec in models.items():
        model = torch.jit.load(str(spec["path"]), map_location=device)
        model.eval()
        records, scalar = _evaluate(model, samples, threshold=float(spec["threshold"]), device=device)
        payload = _make_payload(records, label=label)
        payloads[label] = payload
        summary_rows.extend(_summary_rows(label, payload))
        scalar_summary[label] = scalar

    _write_curves_csv(output_dir / "gnn_dummy_rootlike_binned_curves.csv", payloads)
    _write_summary_csv(output_dir / "gnn_dummy_rootlike_binned_summary.csv", summary_rows)
    _write_json(
        output_dir / "gnn_dummy_rootlike_binned_payloads.json",
        {
            "generator": {
                "events": args.events,
                "seed": args.seed,
                "signal_mean": args.signal_mean,
                "pu_mean": args.pu_mean,
                "close_pair_fraction": args.close_pair_fraction,
                "graph": "legacy eta/phi TICL window graph, delta=0.1",
            },
            "models": {label: {"checkpoint": str(spec["path"]), "threshold": spec["threshold"]} for label, spec in models.items()},
            "scalar_summary": scalar_summary,
            "payloads": payloads,
        },
    )
    plot_paths = _plot_payloads(output_dir, payloads)
    contact_path = _plot_contact_sheet(output_dir, plot_paths)
    print(
        json.dumps(
            {
                "events": args.events,
                "output_dir": str(output_dir),
                "curves_csv": str(output_dir / "gnn_dummy_rootlike_binned_curves.csv"),
                "summary_csv": str(output_dir / "gnn_dummy_rootlike_binned_summary.csv"),
                "payload_json": str(output_dir / "gnn_dummy_rootlike_binned_payloads.json"),
                "contact_sheet": str(contact_path),
                "scalar_summary": scalar_summary,
            },
            indent=2,
        )
    )


def _generate_samples(events: int, seed: int, signal_mean: float, pu_mean: float, close_pair_fraction: float) -> list[Data]:
    sys.path.insert(0, "/Users/chrisizeh/Documents/CERN/TICL-HGCAL-Dummy-Data")
    sys.modules.setdefault("awkward", types.ModuleType("awkward"))
    from ticl_hgcal_dummy.hgcal_like import generate_event

    rng = np.random.default_rng(seed)
    samples = []
    for event_index in range(events):
        event = generate_event(
            rng,
            signal_mean=signal_mean,
            pu_mean=pu_mean,
            close_pair_fraction=close_pair_fraction,
        )
        samples.append(_sample_from_event(event))
        if (event_index + 1) % 25 == 0:
            print(f"generated {event_index + 1}/{events} events", flush=True)
    return samples


def _sample_from_event(event: dict[str, object]) -> Data:
    x = np.stack([np.asarray(event[key], dtype=np.float32) for key in NODE_FEATURE_KEYS], axis=1)
    edge_index = _legacy_ticl_edges(x)
    edge_features = _edge_features(x, edge_index)
    labels = np.asarray(event["y"], dtype=np.int64)
    is_pu = np.asarray(event["isPU"], dtype=bool)
    y = (labels[edge_index[:, 0]] == labels[edge_index[:, 1]]).astype(np.float32)
    y[labels[edge_index[:, 0]] == -1] = 0.0
    y[labels[edge_index[:, 1]] == -1] = 0.0
    y[is_pu[edge_index[:, 0]] != is_pu[edge_index[:, 1]]] = 0.0
    return Data(
        x=torch.as_tensor(x).float(),
        edge_index=torch.as_tensor(edge_index).long(),
        edge_features=torch.as_tensor(edge_features).float(),
        y=torch.as_tensor(y).float(),
        isPU=torch.as_tensor(is_pu),
        num_nodes=x.shape[0],
    )


def _legacy_ticl_edges(x: np.ndarray) -> np.ndarray:
    min_eta = -math.pi
    max_eta = math.pi
    phi_bins = 72
    delta = 0.1
    positive_tiles: dict[tuple[int, int], list[int]] = defaultdict(list)
    negative_tiles: dict[tuple[int, int], list[int]] = defaultdict(list)

    def eta_bin(eta: float) -> int:
        return int((eta - min_eta) * 10)

    def phi_bin(phi: float) -> int:
        return int((phi + math.pi) / (2.0 * math.pi) * phi_bins)

    for idx in range(x.shape[0]):
        eta = float(x[idx, NODE_FEATURE["barycenter_eta"]])
        phi = float(x[idx, NODE_FEATURE["barycenter_phi"]])
        if eta > 0.0:
            positive_tiles[(eta_bin(eta), phi_bin(phi))].append(idx)
        elif eta < 0.0:
            negative_tiles[(eta_bin(eta), phi_bin(phi))].append(idx)

    edges: list[tuple[int, int]] = []
    for idx in range(x.shape[0]):
        eta = float(x[idx, NODE_FEATURE["barycenter_eta"]])
        phi = float(x[idx, NODE_FEATURE["barycenter_phi"]])
        eta_low = max(abs(eta) - delta, min_eta)
        eta_high = min(abs(eta) + delta, max_eta)
        phi_low = phi - delta
        phi_high = phi + delta
        eta_low_bin = eta_bin(eta_low)
        eta_high_bin = eta_bin(eta_high)
        phi_low_bin = phi_bin(phi_low)
        phi_high_bin = phi_bin(phi_high)
        if phi_low_bin > phi_high_bin:
            phi_high_bin += phi_bins

        if eta > 0.0:
            tiles = positive_tiles
            z = float(x[idx, NODE_FEATURE["barycenter_z"]])
            for eta_i in range(eta_low_bin, eta_high_bin + 1):
                for phi_i in range(phi_low_bin, phi_high_bin + 1):
                    for neighbor in tiles.get((eta_i, phi_i % phi_bins), []):
                        if float(x[neighbor, NODE_FEATURE["barycenter_z"]]) > z:
                            edges.append((neighbor, idx))
        elif eta < 0.0:
            tiles = negative_tiles
            abs_z = abs(float(x[idx, NODE_FEATURE["barycenter_z"]]))
            for eta_i in range(eta_low_bin, eta_high_bin + 1):
                for phi_i in range(phi_low_bin, phi_high_bin + 1):
                    for neighbor in tiles.get((eta_i, phi_i % phi_bins), []):
                        if abs(float(x[neighbor, NODE_FEATURE["barycenter_z"]])) > abs_z:
                            edges.append((neighbor, idx))
    return np.asarray(edges, dtype=np.int64)


def _edge_features(x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    out = np.zeros((edge_index.shape[0], 5), dtype=np.float32)
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    out[:, 0] = np.abs(x[dst, NODE_FEATURE["raw_energy"]] - x[src, NODE_FEATURE["raw_energy"]])
    out[:, 1] = np.abs(x[dst, NODE_FEATURE["barycenter_z"]] - x[src, NODE_FEATURE["barycenter_z"]])
    out[:, 2] = np.linalg.norm(
        x[dst][:, [NODE_FEATURE["barycenter_x"], NODE_FEATURE["barycenter_y"]]]
        - x[src][:, [NODE_FEATURE["barycenter_x"], NODE_FEATURE["barycenter_y"]]],
        axis=1,
    )
    dst_vec = x[dst][:, [NODE_FEATURE["eVector0_x"], NODE_FEATURE["eVector0_y"], NODE_FEATURE["eVector0_z"]]]
    src_vec = x[src][:, [NODE_FEATURE["eVector0_x"], NODE_FEATURE["eVector0_y"], NODE_FEATURE["eVector0_z"]]]
    out[:, 3] = np.arccos(np.clip(np.sum(dst_vec * src_vec, axis=1), -1.0, 1.0))
    out[:, 4] = np.abs(x[dst, NODE_FEATURE["time"]] - x[src, NODE_FEATURE["time"]])
    return out


def _evaluate(model: torch.jit.ScriptModule, samples: list[Data], threshold: float, device: torch.device):
    records = []
    tp = fp = fn = tn = 0.0
    with torch.inference_mode():
        for event_id, sample in enumerate(samples):
            sample = sample.to(device)
            scores = model.forward(sample.x, sample.edge_features, sample.edge_index).squeeze(-1)
            y_true = sample.y > 0.0
            y_pred = scores > threshold
            weights = torch.maximum(
                sample.x[sample.edge_index[:, 0], NODE_FEATURE["raw_energy"]].abs(),
                sample.x[sample.edge_index[:, 1], NODE_FEATURE["raw_energy"]].abs(),
            )
            tp += float(weights[y_true & y_pred].sum().cpu())
            fp += float(weights[(~y_true) & y_pred].sum().cpu())
            fn += float(weights[y_true & (~y_pred)].sum().cpu())
            tn += float(weights[(~y_true) & (~y_pred)].sum().cpu())
            records.extend(_edge_records(event_id, sample.x, sample.edge_index, y_true, y_pred, scores))
            truth_components = _connected_components(sample.edge_index[y_true], sample.x.shape[0])
            pred_components = _connected_components(sample.edge_index[y_pred], sample.x.shape[0])
            records.extend(_component_records(event_id, sample.x, truth_components, pred_components, object_type="truth"))
            records.extend(_component_records(event_id, sample.x, pred_components, truth_components, object_type="reco"))
            if (event_id + 1) % 25 == 0:
                print(f"evaluated {event_id + 1}/{len(samples)} events", flush=True)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + fp + fn + tn)
    return records, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


if __name__ == "__main__":
    main()
