#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch


BARYCENTER_Z_INDEX = 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a validation dataset copy with impossible cross-endcap candidate edges removed."
    )
    parser.add_argument("source", help="Dataset directory containing processed/data_*.pt.")
    parser.add_argument("destination", help="Output dataset directory. Use the source path together with --in-place to modify it.")
    parser.add_argument("--in-place", action="store_true", help="Allow source and destination to be identical.")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()
    if source == destination and not args.in_place:
        raise SystemExit("Refusing in-place repair without --in-place.")
    if not (source / "processed").is_dir():
        raise SystemExit(f"Missing processed directory: {source / 'processed'}")

    if source != destination:
        if destination.exists():
            raise SystemExit(f"Destination already exists: {destination}")
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".DS_Store"))

    summary = repair_processed_dataset(destination / "processed")
    summary["source"] = str(source)
    summary["destination"] = str(destination)
    summary_path = destination / "endcap_edge_repair_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def repair_processed_dataset(processed_dir: Path) -> dict[str, Any]:
    totals = {
        "graphs": 0,
        "edges_before": 0,
        "edges_after": 0,
        "removed_cross_endcap_edges": 0,
        "removed_positive_edges": 0,
        "graphs_with_removed_edges": 0,
    }
    per_graph: list[dict[str, Any]] = []

    for path in sorted(processed_dir.glob("data_*.pt"), key=_data_sort_key):
        data = torch.load(path, weights_only=False, map_location="cpu")
        edge_index = data.edge_index
        if edge_index.numel() == 0:
            continue

        z = data.x[:, BARYCENTER_Z_INDEX]
        z0 = z[edge_index[:, 0]]
        z1 = z[edge_index[:, 1]]
        keep = (z0 * z1) >= 0
        removed = ~keep
        removed_count = int(removed.sum().item())
        positive_removed = int((data.y[removed] > 0).sum().item()) if hasattr(data, "y") else 0

        totals["graphs"] += 1
        totals["edges_before"] += int(edge_index.shape[0])
        totals["edges_after"] += int(keep.sum().item())
        totals["removed_cross_endcap_edges"] += removed_count
        totals["removed_positive_edges"] += positive_removed
        if removed_count:
            totals["graphs_with_removed_edges"] += 1
            per_graph.append(
                {
                    "file": path.name,
                    "edges_before": int(edge_index.shape[0]),
                    "edges_after": int(keep.sum().item()),
                    "removed_cross_endcap_edges": removed_count,
                    "removed_positive_edges": positive_removed,
                }
            )

        if removed_count:
            data.edge_index = edge_index[keep]
            if hasattr(data, "edge_features"):
                data.edge_features = data.edge_features[keep]
            if hasattr(data, "y"):
                data.y = data.y[keep]
            if hasattr(data, "PU_info"):
                data.PU_info = data.PU_info[keep]
            data.roots = _roots_from_edges(data.edge_index, int(data.num_nodes))
            torch.save(data, path)

    totals["per_graph"] = per_graph
    return totals


def _roots_from_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    has_inner = torch.zeros(num_nodes, dtype=torch.bool)
    if edge_index.numel():
        has_inner[edge_index[:, 0].long()] = True
    return torch.nonzero(~has_inner, as_tuple=False).flatten().long()


def _data_sort_key(path: Path) -> int:
    match = re.search(r"data_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


if __name__ == "__main__":
    main()
