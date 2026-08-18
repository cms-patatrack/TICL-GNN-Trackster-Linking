#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tracksterLinker"))
os.environ.setdefault("MPLCONFIGDIR", osp.join(tempfile.gettempdir(), "matplotlib"))

import awkward as ak
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from tracksterLinker.datasets.ProcessedGraphDataset import ProcessedGraphDataset
from tracksterLinker.utils.dataUtils import calc_group_score, cross_PU, mask_PU
from tracksterLinker.utils.graphUtils import build_ticl_graph


NODE_FEATURE_KEYS = ProcessedGraphDataset.node_feature_keys
NODE_FEATURE = ProcessedGraphDataset.node_feature_dict
UNIFORM_ID_PROB = np.full(6, 1.0 / 6.0, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ColliderML calorimeter cells into TICL-linking-style PyG graphs. "
            "ColliderML cells are first grouped into pseudo-tracksters, then linked with graphUtils.build_ticl_graph."
        )
    )
    parser.add_argument("--dataset", default="ttbar_pu200", help="ColliderML shorthand, e.g. ttbar_pu200 or ttbar_pu0.")
    parser.add_argument(
        "--local-run-dir",
        default=None,
        help=(
            "Read a local ColliderML simulation run directory containing particles/*.parquet "
            "and calo_hits/*.parquet instead of using the Hugging Face-backed colliderml.load()."
        ),
    )
    parser.add_argument(
        "--local-sim-root",
        default=os.environ.get(
            "COLLIDERML_SIM_OUTPUT_ROOT",
            "~/.cache/colliderml/simulate/colliderml-production/colliderml_output",
        ),
        help="Root to search for local ColliderML simulation outputs matching --dataset.",
    )
    parser.add_argument(
        "--no-local-sim-search",
        action="store_true",
        help="Skip automatic local simulation output discovery and use colliderml.load() directly.",
    )
    parser.add_argument("--output-root", default="data/colliderml_gnn_dataset")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output split folders before writing.")
    parser.add_argument("--train-events", type=int, default=80)
    parser.add_argument("--val-events", type=int, default=10)
    parser.add_argument("--test-events", type=int, default=10)
    parser.add_argument("--event-start", type=int, default=0)
    parser.add_argument("--auto-download", action="store_true", help="Allow ColliderML to download missing cached shards.")
    parser.add_argument("--eta-bin-width", type=float, default=0.035)
    parser.add_argument("--phi-bin-width", type=float, default=0.035)
    parser.add_argument("--depth-bin-width", type=float, default=35.0)
    parser.add_argument(
        "--edge-delta-r",
        type=float,
        default=0.2,
        help="Eta/phi half-width passed directly to tracksterLinker.utils.graphUtils.build_ticl_graph.",
    )
    parser.add_argument(
        "--detectors",
        default=None,
        help="Comma-separated detector IDs/names to keep after exploding calo cells, e.g. '11,14'.",
    )
    parser.add_argument("--min-cell-energy", type=float, default=0.0)
    parser.add_argument("--min-node-energy", type=float, default=0.0)
    parser.add_argument("--min-cells-per-node", type=int, default=1)
    parser.add_argument(
        "--target-nodes",
        type=int,
        default=None,
        help="Merge nearby geometric fragments until each event has at most this many pseudo-tracksters.",
    )
    parser.add_argument(
        "--merge-max-distance",
        type=float,
        default=2.5,
        help="Maximum normalized eta/phi/depth distance for the optional --target-nodes merge pass.",
    )
    parser.add_argument(
        "--min-truth-purity",
        type=float,
        default=0.45,
        help="Nodes below this dominant-particle energy fraction get cluster=-1 and cannot form positive edges.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Optional debug cap on pseudo-tracksters. By default no nodes/hits are dropped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = _load_colliderml(args)
    cells_df, contribs_df = _explode_calo(frames["calo_hits"])
    cells_df = _filter_detectors(cells_df, args.detectors)
    particles_df = _explode_particles(frames["particles"])
    particle_lookup = _particle_lookup(particles_df)

    output_root = Path(args.output_root)
    split_counts = {
        "train": args.train_events,
        "val": args.val_events,
        "test": args.test_events,
    }
    split_dirs = _prepare_split_dirs(output_root, overwrite=args.overwrite)
    event_ids = sorted(cells_df["event_id"].dropna().unique().tolist())
    event_ids = [event_id for event_id in event_ids if event_id >= args.event_start]
    needed = sum(split_counts.values())
    if len(event_ids) < needed:
        raise SystemExit(f"ColliderML load returned {len(event_ids)} events, but {needed} were requested.")

    written: dict[str, int] = {}
    node_scaler = None
    edge_scaler = None
    cursor = 0
    for split, count in split_counts.items():
        split_event_ids = event_ids[cursor : cursor + count]
        cursor += count
        split_node_scalers = []
        split_edge_scalers = []
        out_idx = 0
        for event_id in split_event_ids:
            graph = _event_to_graph(
                cells_df[cells_df["event_id"] == event_id],
                contribs_df[contribs_df["event_id"] == event_id],
                particle_lookup,
                args,
            )
            if graph is None:
                continue
            torch.save(graph, split_dirs[split] / "processed" / f"data_{out_idx}.pt")
            split_node_scalers.append(torch.max(torch.abs(graph.x), axis=0).values.cpu())
            split_edge_scalers.append(torch.max(torch.abs(graph.edge_features), axis=0).values.cpu())
            out_idx += 1
        written[split] = out_idx
        torch.save({"split": split, "events": out_idx, "source": args.dataset}, split_dirs[split] / "processed" / "DONE")
        torch.save({"source": args.dataset}, split_dirs[split] / "raw" / "DONE")
        if split == "train":
            if not split_node_scalers or not split_edge_scalers:
                raise SystemExit("No train graphs were written; loosen energy/binning cuts.")
            node_scaler = torch.stack(split_node_scalers).amax(dim=0).clamp_min(1e-6)
            edge_scaler = torch.stack(split_edge_scalers).amax(dim=0).clamp_min(1e-6)
        if node_scaler is not None and edge_scaler is not None:
            torch.save(node_scaler, split_dirs[split] / "node_scaler.pt")
            torch.save(edge_scaler, split_dirs[split] / "edge_scaler.pt")

    metadata = {
        "dataset": args.dataset,
        "output_root": str(output_root),
        "written": written,
        "fragmentation": {
            "eta_bin_width": args.eta_bin_width,
            "phi_bin_width": args.phi_bin_width,
            "depth_bin_width": args.depth_bin_width,
            "graph_utils_delta": args.edge_delta_r,
            "detectors": args.detectors,
            "min_truth_purity": args.min_truth_purity,
            "max_nodes": args.max_nodes,
            "target_nodes": args.target_nodes,
            "merge_max_distance": args.merge_max_distance,
        },
        "graph_builder": "tracksterLinker.utils.graphUtils.build_ticl_graph",
        "paths": {split: str(path) for split, path in split_dirs.items()},
        "note": (
            "ColliderML calo cells are geometrically fragmented into pseudo-tracksters. "
            "Edges are built by graphUtils.build_ticl_graph, and truth contributions are used only for supervision."
        ),
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def _load_colliderml(args: argparse.Namespace) -> dict[str, Any]:
    if args.local_run_dir is not None:
        return _load_local_run_dir(Path(args.local_run_dir).expanduser(), args)

    if not args.no_local_sim_search:
        run_dir = _find_local_sim_run_dir(args.dataset, Path(args.local_sim_root).expanduser())
        if run_dir is not None:
            return _load_local_run_dir(run_dir, args)

    import colliderml

    max_events = args.event_start + args.train_events + args.val_events + args.test_events
    try:
        return colliderml.load(
            args.dataset,
            tables=["particles", "calo_hits"],
            max_events=max_events,
            auto_download=args.auto_download,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n\n"
            "No local ColliderML cache was found. Re-run this converter with --auto-download, "
            "or first download the same objects with:\n"
            f"  colliderml download --channels {args.dataset.rsplit('_pu', 1)[0]} "
            f"--pileup pu{args.dataset.rsplit('_pu', 1)[1]} "
            "--objects particles,calo_hits "
            f"--max-events {max_events}"
        ) from exc


def _load_local_run_dir(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    import polars as pl

    frames = {}
    for table in ("particles", "calo_hits"):
        files = sorted((run_dir / table).glob("*.parquet"))
        if not files:
            raise SystemExit(f"No {table} parquet files found under {run_dir / table}")
        frames[table] = pl.concat([pl.read_parquet(path) for path in files], how="vertical")
    max_events = args.event_start + args.train_events + args.val_events + args.test_events
    return {
        key: frame.filter((frame["event_id"] >= args.event_start) & (frame["event_id"] < max_events))
        for key, frame in frames.items()
    }


def _find_local_sim_run_dir(dataset: str, local_sim_root: Path) -> Path | None:
    if not local_sim_root.exists():
        return None
    roots = [local_sim_root]
    nested = local_sim_root / "colliderml_output"
    if nested.exists():
        roots.append(nested)
    candidates = []
    for root in roots:
        candidates.extend(root.glob(f"{dataset}*/runs/[0-9]*"))
        candidates.extend(root.glob(f"{dataset}*/runs/all/[0-9]*"))
    valid = [
        path
        for path in candidates
        if (path / "particles").is_dir()
        and (path / "calo_hits").is_dir()
        and any((path / "particles").glob("*.parquet"))
        and any((path / "calo_hits").glob("*.parquet"))
    ]
    if not valid:
        return None
    return sorted(valid, key=lambda path: (len(str(path)), str(path)))[0]


def _explode_calo(calo_hits):
    from colliderml.polars import explode_calo_cells_and_contribs

    cells_df, contribs_df = explode_calo_cells_and_contribs(calo_hits)
    required = {"event_id", "cell_index", "x", "y", "z", "total_energy"}
    missing = sorted(required - set(cells_df.columns))
    if missing:
        raise ValueError(f"ColliderML calo_hits missing required columns: {missing}")
    return cells_df, contribs_df


def _explode_particles(particles):
    from colliderml.polars import explode_particles

    return explode_particles(particles)


def _filter_detectors(cells_df, detectors: str | None):
    if detectors is None:
        return cells_df
    if "detector" not in cells_df.columns:
        raise ValueError("--detectors was set, but ColliderML calo cells do not contain a detector column.")
    wanted = {item.strip() for item in detectors.split(",") if item.strip()}
    if not wanted:
        return cells_df
    detector_values = cells_df["detector"].astype(str)
    return cells_df[detector_values.isin(wanted)]


def _prepare_split_dirs(output_root: Path, *, overwrite: bool) -> dict[str, Path]:
    split_dirs = {
        "train": output_root / "dataset_colliderml_reco",
        "val": output_root / "dataset_colliderml_reco_val",
        "test": output_root / "dataset_colliderml_reco_test",
    }
    if overwrite:
        for path in split_dirs.values():
            if path.exists():
                shutil.rmtree(path)
        metadata = output_root / "metadata.json"
        if metadata.exists():
            metadata.unlink()
    for path in split_dirs.values():
        (path / "processed").mkdir(parents=True, exist_ok=True)
        (path / "raw").mkdir(parents=True, exist_ok=True)
    return split_dirs


def _particle_lookup(particles_df) -> dict[tuple[int, int], dict[str, Any]]:
    id_col = _first_existing(particles_df, ["particle_id", "id", "barcode", "particle_index"])
    pdg_col = _first_existing(particles_df, ["pdg", "pdg_id", "pdgId", "PDG"])
    parent_col = _first_existing(particles_df, ["parent_id", "parent_particle_id", "mother_id"])
    vertex_col = _first_existing(particles_df, ["primary_vertex_index", "vertex_primary", "vertex_index", "production_vertex_index"])
    pileup_col = _first_existing(particles_df, ["is_pileup", "isPU", "pileup"])
    parent_by_event: dict[int, dict[int, int]] = defaultdict(dict)
    if parent_col:
        for row in particles_df.to_dict("records"):
            event_id = int(row["event_id"])
            particle_id = int(row[id_col])
            parent_id = int(row[parent_col]) if not _is_nan(row[parent_col]) else -1
            parent_by_event[event_id][particle_id] = parent_id
    hard_vertex_by_event = {}
    if vertex_col:
        for event_id, group in particles_df.groupby("event_id"):
            vertices = [int(value) for value in group[vertex_col].dropna().tolist()]
            hard_vertex_by_event[int(event_id)] = min(vertices) if vertices else None
    lookup = {}
    for row in particles_df.to_dict("records"):
        event_id = int(row["event_id"])
        particle_id = int(row[id_col])
        label_id = _root_particle_id(particle_id, parent_by_event.get(event_id, {}))
        if pileup_col:
            is_pu = bool(row[pileup_col])
        elif vertex_col:
            hard_vertex = hard_vertex_by_event.get(event_id)
            is_pu = hard_vertex is not None and int(row[vertex_col]) != hard_vertex
        else:
            is_pu = False
        lookup[(event_id, particle_id)] = {
            "label": label_id,
            "pdg": int(row[pdg_col]) if pdg_col and not _is_nan(row[pdg_col]) else 0,
            "isPU": is_pu,
        }
    return lookup


def _root_particle_id(particle_id: int, parent_by_id: dict[int, int]) -> int:
    current = particle_id
    seen = set()
    while current in parent_by_id and parent_by_id[current] >= 0 and current not in seen:
        seen.add(current)
        parent = parent_by_id[current]
        if parent == current:
            break
        current = parent
    return current


def _event_to_graph(cells, contribs, particle_lookup: dict[tuple[int, int], dict[str, Any]], args: argparse.Namespace) -> Data | None:
    cells = cells.copy()
    if args.min_cell_energy > 0:
        cells = cells[cells["total_energy"].astype(float) >= args.min_cell_energy]
    if len(cells) < 2:
        return None
    event_id = int(cells["event_id"].iloc[0])
    cells["eta"] = [_eta(x, y, z) for x, y, z in zip(cells["x"], cells["y"], cells["z"])]
    cells["phi"] = np.arctan2(cells["y"].astype(float), cells["x"].astype(float))
    cells["fragment_key"] = [
        _fragment_key(row, args)
        for row in cells[["eta", "phi", "z"] + (["detector"] if "detector" in cells.columns else [])].to_dict("records")
    ]
    contrib_by_cell = _cell_contributions(contribs)

    groups = [group for _, group in cells.groupby("fragment_key", sort=False)]
    if args.target_nodes is not None:
        groups = _merge_cell_groups(groups, args)

    nodes = []
    for group in groups:
        node = _build_node(group, contrib_by_cell, particle_lookup, event_id, args)
        if node is not None:
            nodes.append(node)
    if len(nodes) < 2:
        return None
    nodes = sorted(nodes, key=lambda node: node["raw_energy"], reverse=True)
    if args.max_nodes is not None:
        nodes = nodes[: args.max_nodes]
    trackster_density = len(nodes) / max(_eta_phi_area(nodes), 1e-6)
    for node in nodes:
        node["trackster_density"] = trackster_density
    event = _nodes_to_awkward_event(nodes, args.edge_delta_r)
    edges = _edges_from_event_outer(event)
    if len(edges) < 2:
        return None
    x = torch.as_tensor(np.stack([_node_feature_vector(node) for node in nodes]), dtype=torch.float32)
    edge_index = torch.as_tensor(edges, dtype=torch.long)
    edge_features = torch.as_tensor(_edge_features(nodes, edges), dtype=torch.float32)
    clusters = ak.to_numpy(event.y).astype(np.int64)
    is_pu = ak.to_numpy(event.isPU).astype(np.int64)
    y = calc_group_score(edges, event.y, event.score, event.shared_e, event.raw_energy)
    cross_pu = cross_PU(is_pu, edges)
    signal_edges = mask_PU(is_pu, edges, PU=False)
    pu_edges = mask_PU(is_pu, edges, PU=True)
    y[cross_pu | pu_edges] = 0
    return Data(
        x=x,
        num_nodes=len(nodes),
        edge_index=edge_index,
        edge_features=edge_features,
        y=torch.as_tensor(y, dtype=torch.float32),
        cluster=torch.as_tensor(clusters, dtype=torch.long),
        roots=ak.to_torch(event.roots).long(),
        isPU=torch.as_tensor(is_pu, dtype=torch.int32),
        PU_info=torch.as_tensor(np.stack([cross_pu, signal_edges, pu_edges], axis=1), dtype=torch.bool),
        node_feature_keys=list(NODE_FEATURE_KEYS),
        node_feature_dict=dict(NODE_FEATURE),
    )


def _nodes_to_awkward_event(nodes: list[dict[str, Any]], delta: float) -> ak.Record:
    event_data = {key: [_node_feature_value(node, key) for node in nodes] for key in NODE_FEATURE_KEYS}
    event_data["y"] = [int(node["cluster"]) for node in nodes]
    event_data["score"] = [0.0 if int(node["cluster"]) >= 0 else 1.0 for node in nodes]
    event_data["shared_e"] = [float(node["raw_energy"]) if int(node["cluster"]) >= 0 else 0.0 for node in nodes]
    event_data["isPU"] = [int(node["isPU"]) for node in nodes]
    trackster_event = ak.Array([event_data])[0]
    graph = build_ticl_graph(len(nodes), trackster_event, delta=delta)
    event_data["inner"] = graph["inner"]
    event_data["outer"] = graph["outer"]
    roots = ak.num(graph["inner"], axis=-1)
    event_data["roots"] = ak.local_index(roots)[roots == 0]
    event_data["idx"] = ak.local_index(ak.Array(event_data["barycenter_x"]))
    return ak.Array([event_data])[0]


def _edges_from_event_outer(event: ak.Record) -> np.ndarray:
    targets = ak.to_numpy(ak.ravel(event.outer))
    sources = ak.to_numpy(ak.ravel(ak.broadcast_arrays(ak.local_index(event.outer, axis=0), event.outer)[0]))
    if len(targets) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.transpose(np.stack([targets, sources])).astype(np.int64, copy=False)


def _fragment_key(row: dict[str, Any], args: argparse.Namespace) -> tuple[Any, ...]:
    eta = float(row["eta"])
    phi = float(row["phi"])
    z = float(row["z"])
    detector = row.get("detector", "calo")
    return (
        detector,
        1 if z >= 0 else -1,
        math.floor(eta / args.eta_bin_width),
        math.floor((phi + math.pi) / args.phi_bin_width),
        math.floor(abs(z) / args.depth_bin_width),
    )


def _merge_cell_groups(groups: list[Any], args: argparse.Namespace) -> list[Any]:
    target = max(int(args.target_nodes), 2)
    groups = list(groups)
    if len(groups) <= target:
        return groups

    stats = [_group_stats(group) for group in groups]
    while len(groups) > target:
        pair = _closest_merge_pair(stats, args.merge_max_distance)
        if pair is None:
            break
        left, right = pair
        groups[left] = pd.concat([groups[left], groups[right]], ignore_index=True)
        stats[left] = _group_stats(groups[left])
        del groups[right]
        del stats[right]
    return groups


def _closest_merge_pair(stats: list[dict[str, float]], max_distance: float) -> tuple[int, int] | None:
    best = None
    best_distance = float("inf")
    order = sorted(range(len(stats)), key=lambda index: stats[index]["energy"])
    for left in order:
        for right in range(len(stats)):
            if left == right:
                continue
            if stats[left]["detector"] != stats[right]["detector"] or stats[left]["sign_z"] != stats[right]["sign_z"]:
                continue
            distance = _group_distance(stats[left], stats[right])
            if distance < best_distance:
                best = (left, right)
                best_distance = distance
        if best is not None and best_distance <= max_distance:
            break
    if best is None or best_distance > max_distance:
        return None
    return tuple(sorted(best))


def _group_stats(group) -> dict[str, float]:
    weights = np.maximum(group["total_energy"].astype(float).to_numpy(), 0.0)
    if weights.sum() <= 0:
        weights = np.ones(len(group), dtype=float)
    eta = group["eta"].astype(float).to_numpy()
    phi = group["phi"].astype(float).to_numpy()
    z = group["z"].astype(float).to_numpy()
    sin_phi = float(np.average(np.sin(phi), weights=weights))
    cos_phi = float(np.average(np.cos(phi), weights=weights))
    detector = float(group["detector"].iloc[0]) if "detector" in group.columns else 0.0
    return {
        "energy": float(weights.sum()),
        "eta": float(np.average(eta, weights=weights)),
        "phi": math.atan2(sin_phi, cos_phi),
        "abs_z": float(np.average(np.abs(z), weights=weights)),
        "sign_z": 1.0 if float(np.average(z, weights=weights)) >= 0 else -1.0,
        "detector": detector,
    }


def _group_distance(left: dict[str, float], right: dict[str, float]) -> float:
    deta = left["eta"] - right["eta"]
    dphi = _delta_phi(left["phi"], right["phi"])
    dz = (left["abs_z"] - right["abs_z"]) / 120.0
    return math.sqrt(deta * deta + dphi * dphi + dz * dz)


def _cell_contributions(contribs) -> dict[int, list[tuple[int, float, float]]]:
    out: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for row in contribs.to_dict("records"):
        if _is_nan(row.get("particle_id")):
            continue
        out[int(row["cell_index"])].append(
            (
                int(row["particle_id"]),
                float(row.get("energy", 0.0) or 0.0),
                float(row.get("time", 0.0) or 0.0),
            )
        )
    return out


def _build_node(group, contrib_by_cell, particle_lookup, event_id: int, args: argparse.Namespace) -> dict[str, Any] | None:
    energy = group["total_energy"].astype(float).to_numpy()
    if float(energy.sum()) < args.min_node_energy or len(group) < args.min_cells_per_node:
        return None
    coords = group[["x", "y", "z"]].astype(float).to_numpy()
    raw_weights = np.maximum(energy, 0.0)
    geom_weights = raw_weights if float(raw_weights.sum()) > 0 else np.ones(len(group), dtype=float)
    bary = np.average(coords, axis=0, weights=geom_weights)
    eigvals, eigvec = _weighted_pca(coords, geom_weights)
    eta = _eta(*bary)
    phi = math.atan2(float(bary[1]), float(bary[0]))
    truth_energy: dict[int, float] = defaultdict(float)
    truth_time_num = 0.0
    truth_time_den = 0.0
    for cell_index in group["cell_index"].astype(int).tolist():
        for particle_id, contrib_energy, contrib_time in contrib_by_cell.get(cell_index, []):
            particle_info = particle_lookup.get((event_id, particle_id), {"label": particle_id})
            truth_energy[int(particle_info["label"])] += max(contrib_energy, 0.0)
            truth_time_num += max(contrib_energy, 0.0) * contrib_time
            truth_time_den += max(contrib_energy, 0.0)
    if truth_energy:
        dominant_particle, dominant_energy = max(truth_energy.items(), key=lambda item: item[1])
        purity = dominant_energy / max(sum(truth_energy.values()), 1e-12)
    else:
        dominant_particle, purity = -1, 0.0
    particle = particle_lookup.get((event_id, dominant_particle), {"pdg": 0, "isPU": False})
    return {
        "barycenter_x": float(bary[0]),
        "barycenter_y": float(bary[1]),
        "barycenter_z": float(bary[2]),
        "barycenter_eta": eta,
        "barycenter_phi": phi,
        "eVector0_x": float(eigvec[0]),
        "eVector0_y": float(eigvec[1]),
        "eVector0_z": float(eigvec[2]),
        "EV1": float(eigvals[0]),
        "EV2": float(eigvals[1]),
        "EV3": float(eigvals[2]),
        "sigmaPCA1": float(math.sqrt(max(eigvals[0], 0.0))),
        "sigmaPCA2": float(math.sqrt(max(eigvals[1], 0.0))),
        "sigmaPCA3": float(math.sqrt(max(eigvals[2], 0.0))),
        "num_LCs": float(len(group)),
        "num_hits": float(len(group)),
        "raw_energy": float(raw_weights.sum()),
        "raw_em_energy": float(_em_energy(group, raw_weights)),
        "z_min": float(np.min(coords[:, 2])),
        "z_max": float(np.max(coords[:, 2])),
        "LC_density": float(len(group) / max(np.ptp(coords[:, 2]), 1.0)),
        "trackster_density": 0.0,
        "time": float(truth_time_num / truth_time_den) if truth_time_den > 0 else 0.0,
        "cluster": int(dominant_particle) if purity >= args.min_truth_purity else -1,
        "isPU": int(bool(particle["isPU"])),
        "pdg": int(particle["pdg"]),
    }


def _weighted_pca(coords: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(coords) < 2 or float(weights.sum()) <= 0:
        return np.asarray([0.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0])
    mean = np.average(coords, axis=0, weights=weights)
    centered = coords - mean
    cov = (centered * weights[:, None]).T @ centered / max(float(weights.sum()), 1e-12)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    vec = eigvecs[:, order[0]]
    if vec[2] < 0:
        vec = -vec
    return eigvals.astype(float), vec.astype(float)


def _em_energy(group, weights: np.ndarray) -> float:
    if "detector" not in group.columns:
        return float(weights.sum())
    mask = group["detector"].astype(str).str.lower().str.contains("em|ecal|electro", regex=True).to_numpy()
    return float(weights[mask].sum())


def _edge_features(nodes: list[dict[str, Any]], edges: np.ndarray) -> np.ndarray:
    out = np.zeros((len(edges), 5), dtype=np.float32)
    for idx, (src, dst) in enumerate(edges):
        s = nodes[int(src)]
        d = nodes[int(dst)]
        out[idx, 0] = abs(d["raw_energy"] - s["raw_energy"])
        out[idx, 1] = abs(d["barycenter_z"] - s["barycenter_z"])
        out[idx, 2] = math.hypot(d["barycenter_x"] - s["barycenter_x"], d["barycenter_y"] - s["barycenter_y"])
        dot = s["eVector0_x"] * d["eVector0_x"] + s["eVector0_y"] * d["eVector0_y"] + s["eVector0_z"] * d["eVector0_z"]
        out[idx, 3] = math.acos(float(np.clip(dot, -1.0, 1.0)))
        out[idx, 4] = abs(d["time"] - s["time"])
    return out


def _node_feature_vector(node: dict[str, Any]) -> np.ndarray:
    return np.asarray([_node_feature_value(node, key) for key in NODE_FEATURE_KEYS], dtype=np.float32)


def _node_feature_value(node: dict[str, Any], key: str) -> float:
    if key.endswith("_prob"):
        prob_index = [
            "photon_prob",
            "electron_prob",
            "muon_prob",
            "neutral_pion_prob",
            "charged_hadron_prob",
            "neutral_hadron_prob",
        ].index(key)
        return float(UNIFORM_ID_PROB[prob_index])
    return float(node[key])


def _eta_phi_area(nodes: list[dict[str, Any]]) -> float:
    eta = np.asarray([node["barycenter_eta"] for node in nodes], dtype=float)
    phi = np.asarray([node["barycenter_phi"] for node in nodes], dtype=float)
    return max(float(np.ptp(eta)), 1e-3) * max(float(np.ptp(phi)), 1e-3)


def _eta(x: float, y: float, z: float) -> float:
    r = math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z))
    denom = max(r - float(z), 1e-9)
    return 0.5 * math.log(max((r + float(z)) / denom, 1e-12))


def _delta_phi(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _first_existing(df, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _is_nan(value: Any) -> bool:
    try:
        return bool(math.isnan(value))
    except Exception:
        return False


if __name__ == "__main__":
    main()
