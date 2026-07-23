import argparse
import csv
import glob
import json
import os
import os.path as osp
import tempfile

os.environ.setdefault("MPLCONFIGDIR", osp.join(tempfile.gettempdir(), "matplotlib"))

import awkward as ak
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
BASE_FOLDER = osp.abspath(osp.join(REPO_ROOT, "..", "data"))
DEFAULT_REAL_SUMMARY = osp.join(BASE_FOLDER, "training_data", "root_dummy_20pions_200pu_analysis", "root_dummy_z_summary.json")
DEFAULT_DUMMY_DIR = osp.join(BASE_FOLDER, "linking_dataset", "dummy_reco_experiment", "histo")
DEFAULT_OUTPUT = osp.join(BASE_FOLDER, "training_data", "dummy_real_stat_comparison")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare generated dummy parquet statistics to the analyzed real ROOT statistics.")
    parser.add_argument("--real-summary", default=DEFAULT_REAL_SUMMARY, help="Path to root_dummy_z_summary.json from analyze_histo_root_z.py.")
    parser.add_argument("--real-z-profile", default=None, help="Optional z_binned_node_profile.csv from analyze_histo_root_z.py.")
    parser.add_argument("--dummy-dir", default=DEFAULT_DUMMY_DIR, help="Directory containing dummy train/val/test parquet folders.")
    parser.add_argument("--splits", nargs="+", default=["test"], help="Dummy splits to compare. Default: test.")
    parser.add_argument("--max-files-per-split", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def summary_values(values):
    values = finite(values)
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def percentiles(values, names=(("p5", 5), ("p50", 50), ("p95", 95))):
    values = finite(values)
    if values.size == 0:
        return {name: float("nan") for name, _ in names}
    return {name: float(np.percentile(values, pct)) for name, pct in names}


def flatten_event(event, key, dtype=float):
    return ak.to_numpy(ak.flatten(event[key], axis=None)).astype(dtype, copy=False)


def collect_dummy_files(dummy_dir, splits, max_files_per_split):
    files = []
    for split in splits:
        split_files = sorted(glob.glob(osp.join(dummy_dir, split, "*.parquet")))
        if max_files_per_split is not None:
            split_files = split_files[: max_files_per_split]
        files.extend(split_files)
    return files


def summarize_dummy(dummy_dir, splits, max_files_per_split=None, max_events=None):
    files = collect_dummy_files(dummy_dir, splits, max_files_per_split)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dummy_dir} for splits {splits}")

    event_counts = []
    pu_counts = []
    signal_trackster_counts = []
    signal_label_counts = []
    pu_label_counts = []
    node_z = []
    raw_energy = []
    num_lcs = []
    num_hits = []
    invalid_time = []
    layer_cluster_counts = []
    lc_z = []
    lc_energy = []
    lc_invalid_time = []
    used_files = []
    n_events = 0

    for file_name in files:
        if max_events is not None and n_events >= max_events:
            break
        data = ak.from_parquet(file_name)
        used_files.append(file_name)
        for event in data:
            if max_events is not None and n_events >= max_events:
                break
            is_pu = np.asarray(event["isPU"], dtype=bool)
            labels = np.asarray(event["y"], dtype=np.int64)
            event_counts.append(len(labels))
            pu_counts.append(int(np.count_nonzero(is_pu)))
            signal_trackster_counts.append(int(np.count_nonzero(~is_pu)))
            signal_label_counts.append(int(len(np.unique(labels[~is_pu]))))
            pu_label_counts.append(int(len(np.unique(labels[is_pu]))))

            node_z.append(np.abs(np.asarray(event["barycenter_z"], dtype=float)))
            raw_energy.append(np.asarray(event["raw_energy"], dtype=float))
            num_lcs.append(np.asarray(event["num_LCs"], dtype=float))
            num_hits.append(np.asarray(event["num_hits"], dtype=float))
            invalid_time.append(np.asarray(event["time"], dtype=float) < -90.0)

            if "vertices_x" in event.fields:
                layer_counts = ak.to_numpy(ak.num(event["vertices_x"], axis=1))
                layer_cluster_counts.append(int(np.sum(layer_counts)))
                lc_z.append(np.abs(flatten_event(event, "vertices_z")))
                lc_energy.append(flatten_event(event, "vertices_energy"))
                lc_invalid_time.append(flatten_event(event, "vertices_time") < -90.0)
            n_events += 1

    node_z = np.concatenate(node_z) if node_z else np.array([])
    raw_energy = np.concatenate(raw_energy) if raw_energy else np.array([])
    num_lcs = np.concatenate(num_lcs) if num_lcs else np.array([])
    num_hits = np.concatenate(num_hits) if num_hits else np.array([])
    invalid_time = np.concatenate(invalid_time) if invalid_time else np.array([], dtype=bool)
    lc_z = np.concatenate(lc_z) if lc_z else np.array([])
    lc_energy = np.concatenate(lc_energy) if lc_energy else np.array([])
    lc_invalid_time = np.concatenate(lc_invalid_time) if lc_invalid_time else np.array([], dtype=bool)

    return {
        "input": dummy_dir,
        "splits": splits,
        "files": used_files,
        "events": int(n_events),
        "nodes_per_event": summary_values(event_counts),
        "pu_tracksters_per_event": summary_values(pu_counts),
        "signal_tracksters_per_event": summary_values(signal_trackster_counts),
        "signal_simtracksters_per_event": summary_values(signal_label_counts),
        "pu_simtracksters_per_event": summary_values(pu_label_counts),
        "pu_trackster_fraction": float(np.sum(pu_counts) / max(1, np.sum(event_counts))),
        "node_abs_z": percentiles(node_z),
        "node_raw_energy": summary_values(raw_energy),
        "node_num_LCs": summary_values(num_lcs),
        "node_num_hits": summary_values(num_hits),
        "invalid_time_fraction": float(np.mean(invalid_time)) if invalid_time.size else float("nan"),
        "layer_clusters_per_event": summary_values(layer_cluster_counts),
        "layer_cluster_abs_z": percentiles(lc_z),
        "layer_cluster_energy": summary_values(lc_energy),
        "layer_cluster_invalid_time_fraction": float(np.mean(lc_invalid_time)) if lc_invalid_time.size else float("nan"),
    }


def get_path(mapping, keys):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return float("nan")
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return float("nan")


def metric_specs():
    return [
        ("nodes_per_event_mean", ("nodes_per_event", "mean"), ("nodes_per_event", "mean")),
        ("nodes_per_event_median", ("nodes_per_event", "median"), ("nodes_per_event", "median")),
        ("nodes_per_event_p95", ("nodes_per_event", "p95"), ("nodes_per_event", "p95")),
        ("node_abs_z_p5", ("node_abs_z", "p5"), ("node_abs_z", "p5")),
        ("node_abs_z_median", ("node_abs_z", "p50"), ("node_abs_z", "p50")),
        ("node_abs_z_p95", ("node_abs_z", "p95"), ("node_abs_z", "p95")),
        ("node_raw_energy_mean", ("node_raw_energy", "mean"), ("node_raw_energy", "mean")),
        ("node_raw_energy_median", ("node_raw_energy", "median"), ("node_raw_energy", "median")),
        ("node_raw_energy_p95", ("node_raw_energy", "p95"), ("node_raw_energy", "p95")),
        ("node_num_LCs_mean", ("node_num_LCs", "mean"), ("node_num_LCs", "mean")),
        ("node_num_LCs_median", ("node_num_LCs", "median"), ("node_num_LCs", "median")),
        ("node_num_LCs_p95", ("node_num_LCs", "p95"), ("node_num_LCs", "p95")),
        ("node_num_hits_mean", ("node_num_hits", "mean"), ("node_num_hits", "mean")),
        ("node_num_hits_median", ("node_num_hits", "median"), ("node_num_hits", "median")),
        ("node_num_hits_p95", ("node_num_hits", "p95"), ("node_num_hits", "p95")),
        ("invalid_time_fraction", ("invalid_time_fraction",), ("invalid_time_fraction",)),
        ("lc_abs_z_p5", ("trackster_vertex_z", "abs_z", "p5"), ("layer_cluster_abs_z", "p5")),
        ("lc_abs_z_median", ("trackster_vertex_z", "abs_z", "p50"), ("layer_cluster_abs_z", "p50")),
        ("lc_abs_z_p95", ("trackster_vertex_z", "abs_z", "p95"), ("layer_cluster_abs_z", "p95")),
        ("lc_energy_median", ("trackster_vertex_z", "energy", "median"), ("layer_cluster_energy", "median")),
        ("lc_energy_p95", ("trackster_vertex_z", "energy", "p95"), ("layer_cluster_energy", "p95")),
        ("signal_simtracksters_per_event_mean", ("signal_simtracksters_per_event", "mean"), ("signal_simtracksters_per_event", "mean")),
    ]


def comparison_rows(real_summary, dummy_summary):
    rows = []
    for name, real_keys, dummy_keys in metric_specs():
        real = get_path(real_summary, real_keys)
        dummy = get_path(dummy_summary, dummy_keys)
        diff = dummy - real
        rel = diff / real if np.isfinite(real) and abs(real) > 1e-12 else float("nan")
        rows.append(
            {
                "metric": name,
                "real": real,
                "dummy": dummy,
                "dummy_minus_real": diff,
                "relative_difference": rel,
                "relative_difference_percent": rel * 100.0 if np.isfinite(rel) else float("nan"),
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_differences(rows, output_dir):
    selected = [
        row
        for row in rows
        if np.isfinite(row["relative_difference_percent"])
        and row["metric"]
        not in {
            "signal_simtracksters_per_event_mean",
            "node_num_hits_mean",
            "node_raw_energy_mean",
            "node_num_LCs_mean",
        }
    ]
    if not selected:
        return None

    labels = [row["metric"].replace("_", " ") for row in selected]
    values = np.asarray([row["relative_difference_percent"] for row in selected], dtype=float)
    order = np.argsort(np.abs(values))
    labels = [labels[idx] for idx in order]
    values = values[order]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.34 * len(labels))))
    colors = np.where(values >= 0, "#4c78a8", "#f58518")
    ax.barh(labels, values, color=colors, alpha=0.82)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("dummy - real [% of real]")
    ax.set_title("Dummy vs ROOT summary differences")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    path = osp.join(output_dir, "dummy_vs_root_metric_difference.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def read_real_z_profile(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [{key: float(value) if value != "" else float("nan") for key, value in row.items()} for row in csv.DictReader(handle)]


def dummy_z_profile(dummy_dir, splits, real_rows, max_files_per_split=None, max_events=None):
    files = collect_dummy_files(dummy_dir, splits, max_files_per_split)
    bins = np.asarray([row["z_low"] for row in real_rows] + [real_rows[-1]["z_high"]], dtype=float)
    z = []
    energy = []
    num_lcs = []
    num_hits = []
    invalid = []
    n_events = 0
    for file_name in files:
        if max_events is not None and n_events >= max_events:
            break
        data = ak.from_parquet(file_name)
        for event in data:
            if max_events is not None and n_events >= max_events:
                break
            z.append(np.abs(np.asarray(event["barycenter_z"], dtype=float)))
            energy.append(np.asarray(event["raw_energy"], dtype=float))
            num_lcs.append(np.asarray(event["num_LCs"], dtype=float))
            num_hits.append(np.asarray(event["num_hits"], dtype=float))
            invalid.append((np.asarray(event["time"], dtype=float) < -90.0).astype(float))
            n_events += 1

    z = np.concatenate(z) if z else np.array([])
    energy = np.concatenate(energy) if energy else np.array([])
    num_lcs = np.concatenate(num_lcs) if num_lcs else np.array([])
    num_hits = np.concatenate(num_hits) if num_hits else np.array([])
    invalid = np.concatenate(invalid) if invalid else np.array([])

    rows = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (z >= low) & (z < high)
        row = {"z_low": float(low), "z_high": float(high), "z_center": float(0.5 * (low + high))}
        row["nodes_per_event"] = float(np.count_nonzero(mask) / max(n_events, 1))
        for key, values in [
            ("raw_energy_median", energy),
            ("num_LCs_median", num_lcs),
            ("num_hits_median", num_hits),
        ]:
            selected = values[mask]
            row[key] = float(np.median(selected)) if selected.size else float("nan")
        row["time_invalid_fraction"] = float(np.mean(invalid[mask])) if np.any(mask) else float("nan")
        rows.append(row)
    return rows


def write_z_profile_comparison(real_rows, dummy_rows, output_dir):
    rows = []
    for real, dummy in zip(real_rows, dummy_rows):
        row = {"z_low": real["z_low"], "z_high": real["z_high"], "z_center": real["z_center"]}
        for key in ["nodes_per_event", "raw_energy_median", "num_LCs_median", "num_hits_median", "time_invalid_fraction"]:
            real_value = real.get(key, float("nan"))
            dummy_value = dummy.get(key, float("nan"))
            row[f"real_{key}"] = real_value
            row[f"dummy_{key}"] = dummy_value
            row[f"diff_{key}"] = dummy_value - real_value
        rows.append(row)
    write_csv(osp.join(output_dir, "z_profile_comparison.csv"), rows)
    return rows


def plot_z_profile_comparison(real_rows, dummy_rows, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    axes = axes.ravel()
    specs = [
        ("nodes_per_event", "nodes / event", False),
        ("raw_energy_median", "median raw energy", True),
        ("num_LCs_median", "median LCs", True),
        ("time_invalid_fraction", "invalid time fraction", False),
    ]
    x = [row["z_center"] for row in real_rows]
    for ax, (key, ylabel, logy) in zip(axes, specs):
        ax.plot(x, [row.get(key, float("nan")) for row in real_rows], marker="o", label="ROOT")
        ax.plot(x, [row.get(key, float("nan")) for row in dummy_rows], marker="o", label="dummy")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if logy:
            ax.set_yscale("log")
        ax.legend()
    for ax in axes[-2:]:
        ax.set_xlabel("|z| [cm]")
    fig.tight_layout()
    path = osp.join(output_dir, "dummy_vs_root_z_profile.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    real_z_profile = args.real_z_profile or osp.join(osp.dirname(args.real_summary), "z_binned_node_profile.csv")

    with open(args.real_summary, encoding="utf-8") as handle:
        real_summary = json.load(handle)
    dummy_summary = summarize_dummy(
        args.dummy_dir,
        args.splits,
        max_files_per_split=args.max_files_per_split,
        max_events=args.max_events,
    )
    rows = comparison_rows(real_summary, dummy_summary)

    summary = {
        "real_summary": args.real_summary,
        "dummy_dir": args.dummy_dir,
        "splits": args.splits,
        "dummy_summary": dummy_summary,
        "comparison": rows,
    }
    with open(osp.join(args.output_dir, "dummy_vs_root_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(osp.join(args.output_dir, "dummy_vs_root_metric_comparison.csv"), rows)
    plots = [plot_metric_differences(rows, args.output_dir)]

    if osp.isfile(real_z_profile):
        real_rows = read_real_z_profile(real_z_profile)
        dummy_rows = dummy_z_profile(
            args.dummy_dir,
            args.splits,
            real_rows,
            max_files_per_split=args.max_files_per_split,
            max_events=args.max_events,
        )
        write_z_profile_comparison(real_rows, dummy_rows, args.output_dir)
        plots.append(plot_z_profile_comparison(real_rows, dummy_rows, args.output_dir))

    plots = [plot for plot in plots if plot is not None]
    print(json.dumps({"summary": summary["dummy_summary"], "outputs": args.output_dir, "plots": plots}, indent=2))


if __name__ == "__main__":
    main()
