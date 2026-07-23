import argparse
import csv
import json
import os
import os.path as osp
import sys
import tempfile
from glob import glob

os.environ.setdefault("MPLCONFIGDIR", osp.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, osp.join(REPO_ROOT, "tracksterLinker"))


OBJECTIVES = {
    "reco_balance": {
        "metric": "association_iou_efficiency",
        "penalties": ["duplicate_rate", "merge_rate"],
        "description": "association_iou_efficiency - duplicate_rate - merge_rate",
    },
    "edge_f1": {"metric": "edge_f1", "penalties": [], "description": "edge_f1"},
    "b3_f1": {"metric": "b3_f1", "penalties": [], "description": "b3_f1"},
    "energy_weighted_iou": {
        "metric": "energy_weighted_iou",
        "penalties": [],
        "description": "energy_weighted_iou",
    },
    "association_iou_efficiency": {
        "metric": "association_iou_efficiency",
        "penalties": [],
        "description": "association_iou_efficiency",
    },
}

SWEEP_PLOT_METRICS = [
    "edge_f1",
    "b3_f1",
    "association_iou_efficiency",
    "mean_best_iou",
    "energy_weighted_iou",
    "duplicate_rate",
    "merge_rate",
]


def plot_metric_bars(metrics, output_dir, filename="reconstruction_metric_bars.png", title="Held-out reconstruction metrics"):
    keys = [
        "edge_f1",
        "b3_f1",
        "containment_efficiency_40",
        "association_iou_efficiency",
        "fake_rate",
        "duplicate_rate",
        "merge_rate",
        "mean_best_iou",
    ]
    names = list(metrics.keys())
    x = np.arange(len(keys))
    width = 0.8 / max(1, len(names))
    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, name in enumerate(names):
        values = [metrics[name].get(key, np.nan) for key in keys]
        ax.bar(x + idx * width, values, width=width, label=name)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(keys, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, filename), dpi=180)
    plt.close(fig)


def write_metric_csv(metrics, output_dir, filename="metrics.csv"):
    rows = []
    for model_name, values in metrics.items():
        for key, value in sorted(values.items()):
            rows.append({"model": model_name, "metric": key, "value": value})

    with open(osp.join(output_dir, filename), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def default_base_folder():
    parent_data = osp.abspath(osp.join(REPO_ROOT, "..", "data"))
    repo_data = osp.abspath(osp.join(REPO_ROOT, "data"))
    if osp.isdir(parent_data):
        return parent_data
    if osp.isdir(repo_data):
        return repo_data
    return parent_data


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate dummy reconstruction metric plots from existing threshold-sweep tables. "
            "This does not rerun model inference."
        )
    )
    parser.add_argument("--base-folder", default=default_base_folder())
    parser.add_argument("--run-name", default="dummy_reco_experiment")
    parser.add_argument("--model-folder", default=None, help="Override training output folder.")
    parser.add_argument("--output-dir", default=None, help="Override where summary plots are written.")
    parser.add_argument("--model-names", nargs="+", default=["focal", "focal_contrastive"])
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), default="reco_balance")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not include unlinked_baseline from the existing metrics.json in bar plots.",
    )
    parser.add_argument(
        "--update-main",
        action="store_true",
        help="Also overwrite metrics.json, metrics.csv, and reconstruction_metric_bars.png in the run folder.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any requested model has no threshold sweep file.",
    )
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def coerce_value(value):
    if value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def read_sweep_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{key: coerce_value(value) for key, value in row.items()} for row in reader]


def read_sweep(path):
    if path.endswith(".json"):
        rows = read_json(path)
    elif path.endswith(".csv"):
        rows = read_sweep_csv(path)
    else:
        raise ValueError(f"Unsupported sweep file type: {path}")

    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Threshold sweep is empty or malformed: {path}")
    return [{key: coerce_value(value) for key, value in row.items()} for row in rows]


def candidate_sweep_paths(model_folder, model_name):
    sweep_dir = osp.join(model_folder, model_name, "threshold_sweep")
    exact = [
        osp.join(sweep_dir, f"{model_name}_threshold_sweep.json"),
        osp.join(sweep_dir, f"{model_name}_threshold_sweep.csv"),
    ]
    legacy = [
        osp.join(sweep_dir, "contrastive_threshold_sweep.json"),
        osp.join(sweep_dir, "contrastive_threshold_sweep.csv"),
    ]
    discovered = sorted(glob(osp.join(sweep_dir, "*threshold_sweep.json"))) + sorted(
        glob(osp.join(sweep_dir, "*threshold_sweep.csv"))
    )
    ordered = []
    for path in exact + legacy + discovered:
        if path not in ordered:
            ordered.append(path)
    return ordered


def find_sweep_path(model_folder, model_name):
    for path in candidate_sweep_paths(model_folder, model_name):
        if osp.isfile(path):
            return path
    return None


def objective_score(row, objective):
    spec = OBJECTIVES[objective]
    value = row.get(spec["metric"], float("nan"))
    if not np.isfinite(value):
        return float("-inf")
    for penalty in spec["penalties"]:
        penalty_value = row.get(penalty, 0.0)
        if np.isfinite(penalty_value):
            value -= penalty_value
    return float(value)


def best_row(rows, objective):
    best = max(rows, key=lambda row: objective_score(row, objective))
    best = dict(best)
    best["threshold_objective"] = OBJECTIVES[objective]["description"]
    best["threshold_objective_score"] = objective_score(best, objective)
    return best


def metric_value(row, key):
    value = row.get(key, np.nan)
    return float(value) if isinstance(value, (int, float)) else np.nan


def plot_model_sweep(rows, model_name, objective, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    thresholds = np.asarray([metric_value(row, "threshold") for row in rows], dtype=float)
    selected = best_row(rows, objective)
    selected_threshold = float(selected["threshold"])

    metrics = [metric for metric in SWEEP_PLOT_METRICS if any(np.isfinite(metric_value(row, metric)) for row in rows)]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for metric in metrics:
        ax.plot(thresholds, [metric_value(row, metric) for row in rows], marker="o", linewidth=1.8, label=metric)

    ax.axvline(selected_threshold, color="black", linestyle="--", linewidth=1.4, label=f"selected {selected_threshold:.3f}")
    ax.set_xlabel("edge threshold")
    ax.set_ylabel("metric")
    ax.set_title(f"{model_name} threshold sweep")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = osp.join(output_dir, f"{model_name}_threshold_sweep_replot.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_sweep_comparison(sweeps, objective, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metrics = ["association_iou_efficiency", "duplicate_rate", "merge_rate", "b3_f1", "edge_f1"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8.5, 2.35 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for model_name, rows in sweeps.items():
            thresholds = [metric_value(row, "threshold") for row in rows]
            values = [metric_value(row, metric) for row in rows]
            if not np.isfinite(values).any():
                continue
            selected = best_row(rows, objective)
            ax.plot(thresholds, values, marker="o", linewidth=1.6, label=model_name)
            ax.axvline(float(selected["threshold"]), color=ax.lines[-1].get_color(), linestyle="--", alpha=0.45)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("edge threshold")
    axes[0].legend(loc="best")
    fig.suptitle(f"Threshold-sweep metrics, selected by {OBJECTIVES[objective]['description']}")
    fig.tight_layout()
    path = osp.join(output_dir, "threshold_sweep_metric_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def numeric_metrics(row):
    return {
        key: float(value)
        for key, value in row.items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    }


def load_existing_metrics(model_folder):
    metrics_path = osp.join(model_folder, "metrics.json")
    if not osp.isfile(metrics_path):
        return {}
    metrics = read_json(metrics_path)
    return metrics if isinstance(metrics, dict) else {}


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def main():
    args = parse_args()
    model_folder = args.model_folder or osp.join(args.base_folder, "training_data", args.run_name)
    output_dir = args.output_dir or model_folder
    os.makedirs(output_dir, exist_ok=True)

    sweeps = {}
    sources = {}
    existing_metrics = load_existing_metrics(model_folder)
    missing = []
    for model_name in args.model_names:
        sweep_path = find_sweep_path(model_folder, model_name)
        if sweep_path is None:
            missing.append(model_name)
            if isinstance(existing_metrics.get(model_name), dict):
                print(f"[fallback] No threshold sweep found for {model_name}; using metrics.json")
            else:
                print(f"[skip] No threshold sweep found for {model_name} and no metrics.json fallback exists")
            continue
        rows = read_sweep(sweep_path)
        rows = sorted(rows, key=lambda row: metric_value(row, "threshold"))
        sweeps[model_name] = rows
        sources[model_name] = sweep_path

    if missing and args.require_all:
        raise FileNotFoundError(f"Missing threshold sweeps for: {', '.join(missing)}")

    selected_metrics = {}
    if not args.no_baseline:
        baseline = existing_metrics.get("unlinked_baseline")
        if baseline is not None:
            selected_metrics["unlinked_baseline"] = numeric_metrics(baseline)

    selection_summary = {
        "objective": args.objective,
        "objective_description": OBJECTIVES[args.objective]["description"],
        "sources": sources,
        "models": {},
    }

    for model_name in args.model_names:
        if model_name in sweeps:
            continue
        fallback = existing_metrics.get(model_name)
        if not isinstance(fallback, dict):
            continue
        selected_metrics[model_name] = numeric_metrics(fallback)
        selection_summary["models"][model_name] = {
            **numeric_metrics(fallback),
            "fallback_source": osp.join(model_folder, "metrics.json"),
            "threshold_objective": "existing metrics.json fallback; no threshold sweep found",
        }

    per_model_plots = {}
    for model_name, rows in sweeps.items():
        selected = best_row(rows, args.objective)
        selected_metrics[model_name] = numeric_metrics(selected)
        selection_summary["models"][model_name] = selected
        sweep_output_dir = osp.join(model_folder, model_name, "threshold_sweep")
        per_model_plots[model_name] = plot_model_sweep(rows, model_name, args.objective, sweep_output_dir)
        print(
            f"[{model_name}] selected threshold={selected['threshold']:.3f} "
            f"score={selected['threshold_objective_score']:.4f}"
        )

    if not selected_metrics:
        raise FileNotFoundError(f"No threshold sweeps or metrics.json fallbacks found under {model_folder}")

    comparison_path = None
    if sweeps:
        comparison_path = plot_sweep_comparison(sweeps, args.objective, output_dir)
    bars_path = osp.join(output_dir, "reconstruction_metric_bars_from_threshold_sweep.png")
    plot_metric_bars(
        selected_metrics,
        output_dir,
        filename=osp.basename(bars_path),
        title=f"Held-out reconstruction metrics from threshold sweep ({OBJECTIVES[args.objective]['description']})",
    )
    metrics_json = osp.join(output_dir, "metrics_from_threshold_sweep.json")
    metrics_csv = osp.join(output_dir, "metrics_from_threshold_sweep.csv")
    summary_json = osp.join(output_dir, "threshold_sweep_metric_selection.json")
    write_json(metrics_json, selected_metrics)
    write_metric_csv(selected_metrics, output_dir, filename=osp.basename(metrics_csv))
    write_json(summary_json, selection_summary)

    if args.update_main:
        write_json(osp.join(model_folder, "metrics.json"), selected_metrics)
        write_metric_csv(selected_metrics, model_folder)
        plot_metric_bars(
            selected_metrics,
            model_folder,
            title=f"Held-out reconstruction metrics from threshold sweep ({OBJECTIVES[args.objective]['description']})",
        )

    print(f"Wrote {metrics_json}")
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {bars_path}")
    if comparison_path is not None:
        print(f"Wrote {comparison_path}")
    for path in per_model_plots.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
