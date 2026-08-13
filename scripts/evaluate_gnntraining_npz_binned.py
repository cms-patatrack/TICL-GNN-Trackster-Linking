#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from evaluate_dummy_gnn_binned import (
    _make_payload,
    _plot_contact_sheet,
    _plot_payloads,
    _summary_rows,
    _write_curves_csv,
    _write_json,
    _write_summary_csv,
)
from evaluate_generated_rootlike_dummy_gnn_binned import _evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate traced GNNs on exported GNNTraining npz events.")
    parser.add_argument("--input-dir", default="data/cms_gnntraining_20pions_200pu_npz")
    parser.add_argument("--output-dir", default="GNN_Paper/images/gnn_cms_gnntraining_binned_performance")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute():
        input_dir = repo_root / input_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(input_dir)
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

    _write_curves_csv(output_dir / "gnn_cms_gnntraining_binned_curves.csv", payloads)
    _write_summary_csv(output_dir / "gnn_cms_gnntraining_binned_summary.csv", summary_rows)
    _write_json(
        output_dir / "gnn_cms_gnntraining_binned_payloads.json",
        {
            "input_dir": str(input_dir),
            "events": len(samples),
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
                "events": len(samples),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "curves_csv": str(output_dir / "gnn_cms_gnntraining_binned_curves.csv"),
                "summary_csv": str(output_dir / "gnn_cms_gnntraining_binned_summary.csv"),
                "payload_json": str(output_dir / "gnn_cms_gnntraining_binned_payloads.json"),
                "contact_sheet": str(contact_path),
                "scalar_summary": scalar_summary,
            },
            indent=2,
        )
    )


def _load_samples(input_dir: Path) -> list[Data]:
    paths = sorted(input_dir.glob("event_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No event_*.npz files found in {input_dir}")
    samples = []
    for path in paths:
        with np.load(path) as event:
            samples.append(
                Data(
                    x=torch.as_tensor(event["x"]).float(),
                    edge_index=torch.as_tensor(event["edge_index"]).long(),
                    edge_features=torch.as_tensor(event["edge_features"]).float(),
                    y=torch.as_tensor(event["y"]).float(),
                    num_nodes=int(event["x"].shape[0]),
                )
            )
    return samples


if __name__ == "__main__":
    main()
