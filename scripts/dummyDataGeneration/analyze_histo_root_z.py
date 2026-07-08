import argparse
import csv
import json
import os
import os.path as osp
import tempfile

os.environ.setdefault("MPLCONFIGDIR", osp.join(tempfile.gettempdir(), "matplotlib"))

import awkward as ak
import matplotlib
import numpy as np
import uproot


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
DEFAULT_INPUT = osp.expanduser("~/Documents/PhD/data/datasets/histo_20Pions_200PU.root")
DEFAULT_OUTPUT = osp.join(REPO_ROOT, "data", "training_data", "root_dummy_20pions_200pu_analysis")
GNN_TREE = "ticlDumperGNN/GNNTraining"
RAW_TREE_SPECS = [
    ("simtrackstersCP", "ticlDumper/simtrackstersCP"),
    ("simtrackstersSC", "ticlDumper/simtrackstersSC"),
    ("trackstersCLUE3DHigh", "ticlDumper/ticlTrackstersCLUE3DHigh"),
]


NODE_BRANCHES = [
    "node_raw_energy",
    "node_raw_em_energy",
    "node_barycenter_x",
    "node_barycenter_y",
    "node_barycenter_z",
    "node_barycenter_eta",
    "node_barycenter_phi",
    "node_EV1",
    "node_EV2",
    "node_EV3",
    "node_sigmaPCA1",
    "node_sigmaPCA2",
    "node_sigmaPCA3",
    "node_z_min",
    "node_z_max",
    "node_time",
    "node_LC_density",
    "node_trackster_density",
    "node_num_LCs",
    "node_num_hits",
    "node_match_idx",
    "node_match_score",
    "simTrackster_isPU",
    "simTrackster_pdgID",
    "simTrackster_raw_energy",
    "simTrackster_true_energy",
]

EDGE_BRANCHES = [
    "edge_barycenter_z",
    "edge_barycenter_xy",
    "edge_time",
    "edge_label",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze z-axis evolution in a GNN-prepared TICL dummy ROOT file.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--tree", default=GNN_TREE)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--z-bins", type=int, default=24)
    return parser.parse_args()


def flatten(array, dtype=float):
    return ak.to_numpy(ak.flatten(array, axis=None)).astype(dtype, copy=False)


def finite_percentiles(values, percentiles=(5, 50, 95)):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"p{p}": float("nan") for p in percentiles}
    return {f"p{p}": float(np.percentile(values, p)) for p in percentiles}


def summarize_values(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def summarize_abs_z(values):
    values = np.abs(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in ["min", "p1", "p5", "p50", "p95", "p99", "max"]}
    return {
        "min": float(np.min(values)),
        "p1": float(np.percentile(values, 1)),
        "p5": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def node_pu_labels(match_idx_events, sim_is_pu_events):
    labels = []
    for match_idx, sim_is_pu in zip(match_idx_events, sim_is_pu_events):
        match_idx = np.asarray(match_idx, dtype=int)
        sim_is_pu = np.asarray(sim_is_pu, dtype=int)
        out = np.full(match_idx.shape, -1, dtype=int)
        valid = (match_idx >= 0) & (match_idx < len(sim_is_pu))
        out[valid] = sim_is_pu[match_idx[valid]]
        labels.append(out)
    return labels


def binned_profile(z_abs, values, bins, reducer="median"):
    z_abs = np.asarray(z_abs, dtype=float)
    values = np.asarray(values, dtype=float)
    rows = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (z_abs >= low) & (z_abs < high) & np.isfinite(values)
        selected = values[mask]
        row = {
            "z_low": float(low),
            "z_high": float(high),
            "z_center": float(0.5 * (low + high)),
            "count": int(selected.size),
        }
        if selected.size == 0:
            row.update({"value": float("nan"), "p25": float("nan"), "p75": float("nan")})
        elif reducer == "mean":
            row.update({"value": float(np.mean(selected)), "p25": float(np.percentile(selected, 25)), "p75": float(np.percentile(selected, 75))})
        else:
            row.update({"value": float(np.median(selected)), "p25": float(np.percentile(selected, 25)), "p75": float(np.percentile(selected, 75))})
        rows.append(row)
    return rows


def fraction_profile(z_abs, mask_values, bins):
    z_abs = np.asarray(z_abs, dtype=float)
    mask_values = np.asarray(mask_values)
    rows = []
    for low, high in zip(bins[:-1], bins[1:]):
        in_bin = (z_abs >= low) & (z_abs < high) & (mask_values >= 0)
        denom = int(np.count_nonzero(in_bin))
        value = float(np.mean(mask_values[in_bin] > 0)) if denom else float("nan")
        rows.append({"z_low": float(low), "z_high": float(high), "z_center": float(0.5 * (low + high)), "count": denom, "value": value})
    return rows


def write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_node_z_summary(output_dir, z, z_abs, raw_energy, num_lcs, num_hits, time, node_is_pu, bins, n_events):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    axes[0].hist(z, bins=80, histtype="step", linewidth=1.8, label="signed z")
    axes[0].hist(z_abs, bins=80, histtype="step", linewidth=1.8, label="|z|")
    axes[0].set_xlabel("z [cm]")
    axes[0].set_ylabel("nodes")
    axes[0].set_title("Trackster barycenter z")
    axes[0].legend()

    counts, _ = np.histogram(z_abs, bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    axes[1].bar(centers, counts / max(n_events, 1), width=np.diff(bins), align="center", alpha=0.75)
    axes[1].set_xlabel("|z| [cm]")
    axes[1].set_ylabel("nodes / event")
    axes[1].set_title("Node multiplicity along z")

    for values, label in [(raw_energy, "raw energy"), (num_lcs, "num LCs"), (num_hits, "num hits")]:
        profile = binned_profile(z_abs, values, bins)
        centers = [row["z_center"] for row in profile]
        med = [row["value"] for row in profile]
        axes[2].plot(centers, med, marker="o", label=label)
    axes[2].set_xlabel("|z| [cm]")
    axes[2].set_ylabel("median value")
    axes[2].set_yscale("log")
    axes[2].set_title("Median node size/energy vs z")
    axes[2].legend()

    invalid_time = np.where(time < -90.0, 1, 0)
    for values, label in [(node_is_pu, "matched to PU"), (invalid_time, "invalid time")]:
        profile = fraction_profile(z_abs, values, bins)
        axes[3].plot([row["z_center"] for row in profile], [row["value"] for row in profile], marker="o", label=label)
    axes[3].set_xlabel("|z| [cm]")
    axes[3].set_ylabel("fraction")
    axes[3].set_ylim(0, 1.05)
    axes[3].set_title("PU/time fraction vs z")
    axes[3].legend()

    fig.tight_layout()
    path = osp.join(output_dir, "node_z_summary.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_feature_z_trends(output_dir, z_abs, arrays, bins):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.ravel()
    groups = [
        ("PCA eigenvalues", ["node_EV1", "node_EV2", "node_EV3"], True),
        ("PCA sigmas", ["node_sigmaPCA1", "node_sigmaPCA2", "node_sigmaPCA3"], False),
        ("z span", ["z_span"], False),
        ("match quality", ["node_match_score"], False),
    ]
    for ax, (title, keys, logy) in zip(axes, groups):
        for key in keys:
            profile = binned_profile(z_abs, arrays[key], bins)
            ax.plot([row["z_center"] for row in profile], [row["value"] for row in profile], marker="o", label=key.replace("node_", ""))
        ax.set_title(title)
        ax.set_ylabel("median")
        ax.grid(True, alpha=0.25)
        if logy:
            ax.set_yscale("log")
        ax.legend()
    for ax in axes[-2:]:
        ax.set_xlabel("|z| [cm]")
    fig.tight_layout()
    path = osp.join(output_dir, "node_feature_z_trends.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def per_sim_z_spans(arrays):
    rows = []
    for event_idx, (z, energy, match_idx, sim_is_pu, sim_pdg) in enumerate(
        zip(
            arrays["node_barycenter_z"],
            arrays["node_raw_energy"],
            arrays["node_match_idx"],
            arrays["simTrackster_isPU"],
            arrays["simTrackster_pdgID"],
        )
    ):
        z = np.asarray(z, dtype=float)
        energy = np.asarray(energy, dtype=float)
        match_idx = np.asarray(match_idx, dtype=int)
        sim_is_pu = np.asarray(sim_is_pu, dtype=int)
        sim_pdg = np.asarray(sim_pdg, dtype=int)
        for sim_idx in np.unique(match_idx[match_idx >= 0]):
            if sim_idx >= len(sim_is_pu):
                continue
            mask = match_idx == sim_idx
            abs_z = np.abs(z[mask])
            weights = np.clip(energy[mask], 0, None)
            rows.append(
                {
                    "event": int(event_idx),
                    "sim_idx": int(sim_idx),
                    "is_pu": int(sim_is_pu[sim_idx]),
                    "pdg_id": int(sim_pdg[sim_idx]) if sim_idx < len(sim_pdg) else 0,
                    "nodes": int(np.count_nonzero(mask)),
                    "z_min": float(np.min(abs_z)),
                    "z_max": float(np.max(abs_z)),
                    "z_span": float(np.max(abs_z) - np.min(abs_z)),
                    "energy_sum": float(np.sum(weights)),
                    "energy_weighted_z": float(np.average(abs_z, weights=weights)) if np.sum(weights) > 0 else float(np.mean(abs_z)),
                }
            )
    return rows


def plot_particle_z_spans(output_dir, span_rows):
    signal = np.asarray([row["z_span"] for row in span_rows if row["is_pu"] == 0], dtype=float)
    pu = np.asarray([row["z_span"] for row in span_rows if row["is_pu"] == 1], dtype=float)
    signal_nodes = np.asarray([row["nodes"] for row in span_rows if row["is_pu"] == 0], dtype=float)
    signal_span = np.asarray([row["z_span"] for row in span_rows if row["is_pu"] == 0], dtype=float)
    pu_nodes = np.asarray([row["nodes"] for row in span_rows if row["is_pu"] == 1], dtype=float)
    pu_span = np.asarray([row["z_span"] for row in span_rows if row["is_pu"] == 1], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(signal, bins=50, histtype="step", linewidth=1.8, label="signal")
    axes[0].hist(pu, bins=50, histtype="step", linewidth=1.8, label="PU")
    axes[0].set_xlabel("|z| span per simTrackster [cm]")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].set_title("Longitudinal shower span")

    axes[1].scatter(signal_span, signal_nodes, s=16, alpha=0.55, label="signal")
    axes[1].scatter(pu_span, pu_nodes, s=12, alpha=0.25, label="PU")
    axes[1].set_xlabel("|z| span [cm]")
    axes[1].set_ylabel("matched nodes")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].set_title("Fragments vs z span")
    fig.tight_layout()
    path = osp.join(output_dir, "particle_z_spans.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_edge_z_profile(output_dir, edge_dz, edge_label):
    edge_dz = np.abs(np.asarray(edge_dz, dtype=float))
    edge_label = np.asarray(edge_label, dtype=int) > 0
    finite = np.isfinite(edge_dz)
    edge_dz = edge_dz[finite]
    edge_label = edge_label[finite]
    if edge_dz.size == 0:
        return None

    bins = np.linspace(0, np.percentile(edge_dz, 99.5), 45)
    bins[0] = 0.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(edge_dz[~edge_label], bins=bins, histtype="step", density=True, linewidth=1.8, label="false edge")
    axes[0].hist(edge_dz[edge_label], bins=bins, histtype="step", density=True, linewidth=1.8, label="true edge")
    axes[0].set_xlabel("|delta z| [cm]")
    axes[0].set_ylabel("density")
    axes[0].set_title("Edge z separation")
    axes[0].legend()

    centers = 0.5 * (bins[:-1] + bins[1:])
    true_fraction = []
    counts = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (edge_dz >= low) & (edge_dz < high)
        counts.append(int(np.count_nonzero(mask)))
        true_fraction.append(float(np.mean(edge_label[mask])) if counts[-1] else float("nan"))
    axes[1].plot(centers, true_fraction, marker="o")
    axes[1].set_xlabel("|delta z| [cm]")
    axes[1].set_ylabel("true-edge fraction")
    axes[1].set_ylim(0, max(0.05, np.nanmax(true_fraction) * 1.2))
    axes[1].set_title("Truth rate vs z separation")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    path = osp.join(output_dir, "edge_z_profile.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def top_counts(values, limit=10):
    values = np.asarray(values)
    if values.size == 0:
        return []
    unique, counts = np.unique(values, return_counts=True)
    order = np.argsort(counts)[::-1][:limit]
    return [{"value": int(unique[idx]), "count": int(counts[idx])} for idx in order]


def raw_tree_z_summaries(root_file, entry_stop):
    summaries = {}
    flat_z = {}
    for label, tree_name in RAW_TREE_SPECS:
        if tree_name not in root_file:
            continue
        tree = root_file[tree_name]
        branches = ["NTracksters", "barycenter_z", "raw_energy"]
        if "pdgID" in tree.keys():
            branches.append("pdgID")
        arrays = tree.arrays(branches, library="ak", entry_stop=entry_stop)
        z_values = flatten(arrays["barycenter_z"])
        energy_values = flatten(arrays["raw_energy"])
        summary = {
            "tracksters_per_event": summarize_values(ak.to_numpy(arrays["NTracksters"])),
            "abs_z": summarize_abs_z(z_values),
            "raw_energy": summarize_values(energy_values),
        }
        if "pdgID" in arrays.fields:
            pdg_values = flatten(arrays["pdgID"], dtype=int)
            summary["top_pdg_id"] = top_counts(pdg_values)
        summaries[label] = summary
        flat_z[label] = np.abs(z_values[np.isfinite(z_values)])
    return summaries, flat_z


def reco_vertex_z_summary(root_file, entry_stop):
    tree_name = "ticlDumper/ticlTrackstersCLUE3DHigh"
    if tree_name not in root_file:
        return {}, np.array([])
    tree = root_file[tree_name]
    arrays = tree.arrays(["vertices_z", "vertices_energy"], library="ak", entry_stop=entry_stop)
    vertex_z = flatten(arrays["vertices_z"])
    vertex_energy = flatten(arrays["vertices_energy"])
    return {
        "abs_z": summarize_abs_z(vertex_z),
        "energy": summarize_values(vertex_energy),
    }, np.abs(vertex_z[np.isfinite(vertex_z)])


def plot_tree_z_comparison(output_dir, gnn_abs_z, raw_abs_z, vertex_abs_z):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(gnn_abs_z, bins=90, histtype="step", density=True, linewidth=1.8, label="GNN nodes")
    for label, values in raw_abs_z.items():
        axes[0].hist(values, bins=90, histtype="step", density=True, linewidth=1.5, label=label)
    axes[0].set_xlabel("|z| [cm]")
    axes[0].set_ylabel("density")
    axes[0].set_xlim(315, 520)
    axes[0].set_title("Barycenter z by tree")
    axes[0].legend()

    axes[1].hist(gnn_abs_z, bins=90, histtype="step", density=True, linewidth=1.8, label="GNN nodes")
    if vertex_abs_z.size:
        axes[1].hist(vertex_abs_z, bins=90, histtype="step", density=True, linewidth=1.5, label="reco vertices")
    axes[1].set_xlabel("|z| [cm]")
    axes[1].set_ylabel("density")
    axes[1].set_xlim(315, 520)
    axes[1].set_title("Node vs layer-cluster vertex z")
    axes[1].legend()
    fig.tight_layout()
    path = osp.join(output_dir, "tree_z_comparison.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    entry_stop = args.max_events
    with uproot.open(args.input) as root_file:
        tree = root_file[args.tree]
        n_events = tree.num_entries if entry_stop is None else min(entry_stop, tree.num_entries)
        arrays = tree.arrays(NODE_BRANCHES, library="ak", entry_stop=entry_stop)
        edge_arrays = tree.arrays(EDGE_BRANCHES, library="ak", entry_stop=entry_stop)
        raw_tree_summaries, raw_tree_z = raw_tree_z_summaries(root_file, entry_stop)
        vertex_summary, vertex_z = reco_vertex_z_summary(root_file, entry_stop)

    node_is_pu_events = node_pu_labels(arrays["node_match_idx"], arrays["simTrackster_isPU"])
    node_is_pu = np.concatenate(node_is_pu_events)

    flat = {key: flatten(arrays[key]) for key in NODE_BRANCHES if key.startswith("node_")}
    z = flat["node_barycenter_z"]
    z_abs = np.abs(z)
    z_span = np.abs(flat["node_z_max"] - flat["node_z_min"])
    flat["z_span"] = z_span

    z_min = max(300.0, float(np.nanpercentile(z_abs, 0.5)))
    z_max = min(560.0, float(np.nanpercentile(z_abs, 99.5)))
    bins = np.linspace(z_min, z_max, args.z_bins + 1)

    z_profile_rows = []
    profile_specs = [
        ("raw_energy_median", flat["node_raw_energy"], "median"),
        ("raw_em_energy_median", flat["node_raw_em_energy"], "median"),
        ("num_LCs_median", flat["node_num_LCs"], "median"),
        ("num_hits_median", flat["node_num_hits"], "median"),
        ("EV1_median", flat["node_EV1"], "median"),
        ("EV2_median", flat["node_EV2"], "median"),
        ("EV3_median", flat["node_EV3"], "median"),
        ("sigmaPCA1_median", flat["node_sigmaPCA1"], "median"),
        ("time_invalid_fraction", np.where(flat["node_time"] < -90.0, 1, 0), "mean"),
    ]
    for low, high in zip(bins[:-1], bins[1:]):
        row = {"z_low": float(low), "z_high": float(high), "z_center": float(0.5 * (low + high))}
        mask = (z_abs >= low) & (z_abs < high)
        row["nodes_per_event"] = float(np.count_nonzero(mask) / max(n_events, 1))
        for name, values, reducer in profile_specs:
            values = np.asarray(values)
            selected = values[mask]
            selected = selected[np.isfinite(selected)]
            if selected.size == 0:
                row[name] = float("nan")
            elif reducer == "mean":
                row[name] = float(np.mean(selected))
            else:
                row[name] = float(np.median(selected))
        matched = mask & (node_is_pu >= 0)
        row["matched_fraction"] = float(np.count_nonzero(matched) / max(np.count_nonzero(mask), 1))
        row["matched_pu_fraction"] = float(np.mean(node_is_pu[matched] > 0)) if np.any(matched) else float("nan")
        z_profile_rows.append(row)
    write_rows(osp.join(args.output_dir, "z_binned_node_profile.csv"), z_profile_rows)

    span_rows = per_sim_z_spans(arrays)
    write_rows(osp.join(args.output_dir, "simtrackster_z_spans.csv"), span_rows)

    edge_dz = flatten(edge_arrays["edge_barycenter_z"])
    edge_label = flatten(edge_arrays["edge_label"], dtype=int)

    plots = [
        plot_node_z_summary(args.output_dir, z, z_abs, flat["node_raw_energy"], flat["node_num_LCs"], flat["node_num_hits"], flat["node_time"], node_is_pu, bins, n_events),
        plot_feature_z_trends(args.output_dir, z_abs, flat, bins),
        plot_particle_z_spans(args.output_dir, span_rows),
        plot_edge_z_profile(args.output_dir, edge_dz, edge_label),
        plot_tree_z_comparison(args.output_dir, z_abs, raw_tree_z, vertex_z),
    ]

    event_node_counts = ak.to_numpy(ak.num(arrays["node_barycenter_z"], axis=1))
    sim_counts = ak.to_numpy(ak.num(arrays["simTrackster_isPU"], axis=1))
    pu_sim_counts = ak.to_numpy(ak.sum(arrays["simTrackster_isPU"], axis=1))
    signal_sim_counts = sim_counts - pu_sim_counts
    summary = {
        "input": args.input,
        "tree": args.tree,
        "events": int(n_events),
        "nodes_per_event": summarize_values(event_node_counts),
        "simtracksters_per_event": summarize_values(sim_counts),
        "signal_simtracksters_per_event": summarize_values(signal_sim_counts),
        "pu_simtracksters_per_event": summarize_values(pu_sim_counts),
        "node_abs_z": finite_percentiles(z_abs),
        "node_raw_energy": summarize_values(flat["node_raw_energy"]),
        "node_num_LCs": summarize_values(flat["node_num_LCs"]),
        "node_num_hits": summarize_values(flat["node_num_hits"]),
        "invalid_time_fraction": float(np.mean(flat["node_time"] < -90.0)),
        "matched_node_pu_fraction": float(np.mean(node_is_pu[node_is_pu >= 0] > 0)),
        "matched_node_unknown_fraction": float(np.mean(node_is_pu < 0)),
        "true_edge_fraction": float(np.mean(edge_label > 0)) if edge_label.size else float("nan"),
        "edge_abs_delta_z": finite_percentiles(np.abs(edge_dz)),
        "signal_z_span": summarize_values([row["z_span"] for row in span_rows if row["is_pu"] == 0]),
        "pu_z_span": summarize_values([row["z_span"] for row in span_rows if row["is_pu"] == 1]),
        "raw_trees": raw_tree_summaries,
        "trackster_vertex_z": vertex_summary,
        "plots": [path for path in plots if path is not None],
    }
    with open(osp.join(args.output_dir, "root_dummy_z_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
