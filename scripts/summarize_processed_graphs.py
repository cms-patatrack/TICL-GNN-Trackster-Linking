#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prebuilt PyG graph datasets.")
    parser.add_argument(
        "path",
        help=(
            "Processed split directory containing processed/data_*.pt, or a parent "
            "directory containing dataset_colliderml_reco* split folders."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Write machine-readable JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.path).expanduser()
    summaries = {}
    for split, path in _split_dirs(root).items():
        summaries[split] = _summarize_split(path)

    if args.json:
        print(json.dumps(summaries, indent=2))
        return

    for split, summary in summaries.items():
        print(f"\n{split}: {summary['path']}")
        if summary["graphs"] == 0:
            print("  graphs: 0")
            continue
        print(f"  graphs: {summary['graphs']}")
        print(
            "  nodes/event: "
            f"mean={summary['nodes_mean']:.2f} min={summary['nodes_min']} max={summary['nodes_max']} total={summary['nodes_total']}"
        )
        print(
            "  edges/event: "
            f"mean={summary['edges_mean']:.2f} min={summary['edges_min']} max={summary['edges_max']} total={summary['edges_total']}"
        )
        print(
            "  positives: "
            f"{summary['positive_edges_total']} / {summary['edges_total']} "
            f"({summary['positive_edge_fraction']:.6f})"
        )
        print(
            "  node classes: "
            f"signal={summary['signal_nodes_total']} PU={summary['pu_nodes_total']}"
        )
        print(
            "  sums: "
            f"num_hits={summary['num_hits_total']:.1f} raw_energy={summary['raw_energy_total']:.6g}"
        )


def _split_dirs(root: Path) -> dict[str, Path]:
    if (root / "processed").is_dir():
        return {"input": root}

    return {
        "train": _first_existing(root, ["dataset_colliderml_reco", "dataset_dummy_reco"]),
        "val": _first_existing(root, ["dataset_colliderml_reco_val", "dataset_dummy_reco_val"]),
        "test": _first_existing(root, ["dataset_colliderml_reco_test", "dataset_dummy_reco_test"]),
    }


def _first_existing(root: Path, names: list[str]) -> Path:
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    return root / names[0]


def _summarize_split(path: Path) -> dict[str, Any]:
    data_paths = sorted((path / "processed").glob("data_*.pt"), key=_data_index)
    if not data_paths:
        return {"path": str(path), "graphs": 0}

    nodes = []
    edges = []
    positives = []
    signal_nodes = []
    pu_nodes = []
    num_hits = []
    raw_energy = []
    for data_path in data_paths:
        graph = torch.load(data_path, weights_only=False, map_location=torch.device("cpu"))
        nodes.append(int(graph.num_nodes))
        edges.append(int(graph.edge_index.shape[0]))
        positives.append(int((graph.y > 0).sum()))
        if hasattr(graph, "isPU"):
            signal_nodes.append(int((graph.isPU == 0).sum()))
            pu_nodes.append(int((graph.isPU != 0).sum()))
        else:
            signal_nodes.append(int(graph.num_nodes))
            pu_nodes.append(0)
        num_hits.append(float(graph.x[:, 15].sum()))
        raw_energy.append(float(graph.x[:, 16].sum()))

    return {
        "path": str(path),
        "graphs": len(data_paths),
        "nodes_total": int(sum(nodes)),
        "nodes_mean": _mean(nodes),
        "nodes_min": int(min(nodes)),
        "nodes_max": int(max(nodes)),
        "edges_total": int(sum(edges)),
        "edges_mean": _mean(edges),
        "edges_min": int(min(edges)),
        "edges_max": int(max(edges)),
        "positive_edges_total": int(sum(positives)),
        "negative_edges_total": int(sum(edges) - sum(positives)),
        "positive_edge_fraction": float(sum(positives) / sum(edges)) if sum(edges) else 0.0,
        "signal_nodes_total": int(sum(signal_nodes)),
        "pu_nodes_total": int(sum(pu_nodes)),
        "num_hits_total": float(sum(num_hits)),
        "raw_energy_total": float(sum(raw_energy)),
    }


def _data_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except Exception:
        return 0


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
