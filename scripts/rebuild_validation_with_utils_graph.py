#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import torch
from torch_geometric.data import Data


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild processed validation graphs from the raw cache using the graphUtils eta/phi endcap graph convention."
    )
    parser.add_argument("source", help="Dataset directory containing raw/data_id_*.pt.")
    parser.add_argument("destination", help="New dataset directory to write.")
    parser.add_argument("--delta", type=float, default=0.2, help="Eta/phi tile search half-width.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of graphs to write.")
    parser.add_argument(
        "--reference-helper",
        action="store_true",
        help="Call tracksterLinker.utils.graphUtils.build_ticl_graph directly instead of the local vectorized equivalent.",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()
    if not (source / "raw").is_dir():
        raise SystemExit(f"Missing raw directory: {source / 'raw'}")
    if destination.exists():
        raise SystemExit(f"Destination already exists: {destination}")

    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".DS_Store", "processed"))
    processed_dir = destination / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    reference_builder = None
    if args.reference_helper:
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root / "tracksterLinker"))
        from tracksterLinker.utils.graphUtils import build_ticl_graph

        reference_builder = build_ticl_graph

    summary = rebuild_processed(source / "raw", processed_dir, args.delta, limit=args.limit, reference_builder=reference_builder)
    summary["source"] = str(source)
    summary["destination"] = str(destination)
    summary["delta"] = args.delta
    summary["graph_builder"] = "tracksterLinker.utils.graphUtils.build_ticl_graph" if args.reference_helper else "local_equivalent"
    with (destination / "utils_graph_rebuild_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    torch.save({"split": "val", "test": True, "events": summary["graphs"]}, processed_dir / "DONE")
    print(json.dumps(summary, indent=2))


def rebuild_processed(raw_dir: Path, processed_dir: Path, delta: float, limit: int | None = None, reference_builder=None) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "graphs": 0,
        "edges": 0,
        "positive_edges": 0,
        "cross_endcap_edges": 0,
        "negative_wrong_depth_edges": 0,
        "positive_wrong_depth_edges": 0,
    }
    graph_index = 0
    for raw_path in sorted(raw_dir.glob("data_id_*.pt"), key=_data_sort_key):
        run = torch.load(raw_path, weights_only=False)
        for event in run:
            if limit is not None and graph_index >= limit:
                return totals
            data = build_data(event, delta, reference_builder=reference_builder)
            if data is None:
                continue
            torch.save(data, processed_dir / f"data_{graph_index}.pt")
            graph_index += 1
            edge_index = data.edge_index
            z = data.x[:, NODE_FEATURE["barycenter_z"]]
            z0 = z[edge_index[:, 0]]
            z1 = z[edge_index[:, 1]]
            negative = (z0 < 0) & (z1 < 0)
            positive = (z0 > 0) & (z1 > 0)
            totals["graphs"] += 1
            totals["edges"] += int(edge_index.shape[0])
            totals["positive_edges"] += int((data.y > 0).sum().item())
            totals["cross_endcap_edges"] += int(((z0 * z1) < 0).sum().item())
            totals["negative_wrong_depth_edges"] += int((negative & (z0.abs() < z1.abs())).sum().item())
            totals["positive_wrong_depth_edges"] += int((positive & (z0.abs() < z1.abs())).sum().item())
    return totals


def build_data(event: ak.Record, delta: float, reference_builder=None) -> Data | None:
    n_tracksters = len(event["barycenter_x"])
    if n_tracksters <= 1:
        return None

    features = np.stack([ak.to_numpy(event[field]) for field in NODE_FEATURE_KEYS], axis=1).astype(np.float32)
    if reference_builder is None:
        edge_index = build_edges_like_graph_utils(features, delta=delta)
    else:
        graph = reference_builder(n_tracksters, event, delta=delta)
        targets = ak.to_numpy(ak.ravel(graph.outer))
        sources = ak.to_numpy(ak.ravel(ak.broadcast_arrays(ak.local_index(graph.outer, axis=0), graph.outer)[0]))
        edge_index = np.transpose(np.stack([targets, sources])).astype(np.int64)
    if edge_index.shape[0] < 2:
        return None

    edge_features = np.zeros((edge_index.shape[0], 5), dtype=np.float32)
    edge_features[:, 0] = np.abs(features[edge_index[:, 1], NODE_FEATURE["raw_energy"]] - features[edge_index[:, 0], NODE_FEATURE["raw_energy"]])
    edge_features[:, 1] = np.abs(features[edge_index[:, 1], NODE_FEATURE["barycenter_z"]] - features[edge_index[:, 0], NODE_FEATURE["barycenter_z"]])
    edge_features[:, 2] = np.linalg.norm(
        features[edge_index[:, 1]][:, [NODE_FEATURE["barycenter_x"], NODE_FEATURE["barycenter_y"]]]
        - features[edge_index[:, 0]][:, [NODE_FEATURE["barycenter_x"], NODE_FEATURE["barycenter_y"]]],
        axis=1,
    )
    edge_features[:, 3] = np.arccos(
        np.clip(
            np.sum(
                features[edge_index[:, 1]][:, [NODE_FEATURE["eVector0_x"], NODE_FEATURE["eVector0_y"], NODE_FEATURE["eVector0_z"]]]
                * features[edge_index[:, 0]][:, [NODE_FEATURE["eVector0_x"], NODE_FEATURE["eVector0_y"], NODE_FEATURE["eVector0_z"]]],
                axis=1,
            ),
            -1,
            1,
        )
    )
    edge_features[:, 4] = np.abs(features[edge_index[:, 1], NODE_FEATURE["time"]] - features[edge_index[:, 0], NODE_FEATURE["time"]])

    labels = calc_group_score(edge_index, event)
    cluster = ak.to_numpy(event.y).astype(np.int64)
    return Data(
        x=torch.as_tensor(features).float(),
        num_nodes=n_tracksters,
        edge_index=torch.as_tensor(edge_index).long(),
        edge_features=torch.as_tensor(edge_features).float(),
        y=torch.as_tensor(labels).float(),
        cluster=torch.as_tensor(cluster).long(),
        roots=roots_from_edges(edge_index, n_tracksters),
    )


def build_edges_like_graph_utils(features: np.ndarray, delta: float) -> np.ndarray:
    eta = features[:, NODE_FEATURE["barycenter_eta"]]
    phi = features[:, NODE_FEATURE["barycenter_phi"]]
    z = features[:, NODE_FEATURE["barycenter_z"]]
    abs_eta = np.abs(eta)
    abs_z = np.abs(z)
    min_eta = -math.pi
    max_eta = math.pi
    phi_bins = 72

    def eta_bin(value: float) -> int:
        return int((value - min_eta) * 10)

    def phi_bin(value: float) -> int:
        return int((value + math.pi) / (2 * math.pi) * phi_bins)

    tiles: dict[str, dict[tuple[int, int], list[int]]] = {"pos": {}, "neg": {}}
    for idx in range(features.shape[0]):
        side = "pos" if eta[idx] > 0.0 else "neg" if eta[idx] < 0.0 else None
        if side is None:
            continue
        key = (eta_bin(float(abs_eta[idx])), phi_bin(float(phi[idx])))
        tiles[side].setdefault(key, []).append(idx)

    edges: list[tuple[int, int]] = []
    for src in range(features.shape[0]):
        side = "pos" if eta[src] > 0.0 else "neg" if eta[src] < 0.0 else None
        if side is None:
            continue
        eta_low = max(float(abs_eta[src]) - delta, min_eta)
        eta_high = min(float(abs_eta[src]) + delta, max_eta)
        eta_low_bin = eta_bin(eta_low)
        eta_high_bin = eta_bin(eta_high)
        phi_low_bin = phi_bin(float(phi[src]) - delta)
        phi_high_bin = phi_bin(float(phi[src]) + delta)
        if phi_low_bin > phi_high_bin:
            phi_high_bin += phi_bins

        for eta_i in range(eta_low_bin, eta_high_bin + 1):
            for phi_i in range(phi_low_bin, phi_high_bin + 1):
                for dst in tiles[side].get((eta_i, phi_i % phi_bins), []):
                    if side == "pos":
                        if z[dst] > z[src]:
                            edges.append((dst, src))
                    elif abs_z[dst] > abs_z[src]:
                        edges.append((dst, src))

    return np.asarray(edges, dtype=np.int64)


def calc_group_score(edges: np.ndarray, event: ak.Record) -> np.ndarray:
    y = ak.to_numpy(event.y)
    score = ak.to_numpy(event.score)
    shared_energy = ak.to_numpy(event.shared_e)
    raw_energy = np.clip(ak.to_numpy(event.raw_energy), 1e-12, None)
    term_src = (1.0 - score[edges[:, 0]]) * shared_energy[edges[:, 0]] / raw_energy[edges[:, 0]]
    term_dst = (1.0 - score[edges[:, 1]]) * shared_energy[edges[:, 1]] / raw_energy[edges[:, 1]]
    weight = ((term_src + term_dst) / 2.0).astype(np.float32)
    weight[y[edges[:, 0]] != y[edges[:, 1]]] = 0.0
    weight[y[edges[:, 0]] == -1] = 0.0
    weight[y[edges[:, 1]] == -1] = 0.0
    return weight


def roots_from_edges(edge_index: np.ndarray, num_nodes: int) -> torch.Tensor:
    has_inner = np.zeros(num_nodes, dtype=bool)
    has_inner[edge_index[:, 0]] = True
    return torch.as_tensor(np.flatnonzero(~has_inner), dtype=torch.long)


def _data_sort_key(path: Path) -> int:
    match = re.search(r"data_(?:id_)?(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


if __name__ == "__main__":
    main()
