import argparse
import importlib.util
import os
import os.path as osp
import tempfile

os.environ.setdefault("MPLCONFIGDIR", osp.join(tempfile.gettempdir(), "matplotlib"))

import awkward as ak
import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
DEFAULT_BASE_FOLDER = osp.abspath(osp.join(REPO_ROOT, "..", "data"))
DEFAULT_PLOT_ASSOCIATIONS = osp.expanduser(
    "~/Documents/PhD/complex/awkward_complex/datasets/cern/plot_associations.py"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot dummy trackster associations using awkward_complex's CERN association plotter."
    )
    parser.add_argument("--base-folder", default=DEFAULT_BASE_FOLDER)
    parser.add_argument("--run-name", default="dummy_reco_experiment")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--file-index", type=int, default=0)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--input-file", default=None, help="Override the dummy parquet file to plot.")
    parser.add_argument("--output-folder", default=None, help="Override where plots are written.")
    parser.add_argument("--file-suffix", default=None)
    parser.add_argument("--signal-only", action="store_true", help="Only plot non-PU tracksters.")
    parser.add_argument("--poster", action="store_true", help="Write cleaner PNG/PDF versions suitable for posters.")
    parser.add_argument(
        "--plot-associations",
        default=DEFAULT_PLOT_ASSOCIATIONS,
        help="Path to awkward_complex/datasets/cern/plot_associations.py.",
    )
    return parser.parse_args()


def load_plot_associations(path):
    if not osp.isfile(path):
        raise FileNotFoundError(f"Could not find plot_associations.py at {path}")

    spec = importlib.util.spec_from_file_location("awkward_complex_plot_associations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_input_file(args):
    return osp.join(
        args.base_folder,
        "linking_dataset",
        args.run_name,
        "histo",
        args.split,
        f"dummy_{args.split}_{args.file_index:05d}.parquet",
    )


def default_output_folder(args):
    return osp.join(args.base_folder, "training_data", args.run_name, "association_plots")


def event_to_features_and_labels(event, signal_only=False):
    y = np.asarray(event["y"], dtype=np.int64)
    is_pu = np.asarray(event["isPU"], dtype=bool)
    features = np.stack(
        [
            np.asarray(event["barycenter_eta"], dtype=np.float32),
            np.asarray(event["barycenter_phi"], dtype=np.float32),
            np.asarray(event["barycenter_z"], dtype=np.float32),
        ],
        axis=1,
    )

    if signal_only:
        mask = ~is_pu
        features = features[mask]
        y = y[mask]
        is_pu = is_pu[mask]

    signal_labels = y.copy()
    signal_labels[is_pu] = -1
    return torch.as_tensor(features), y, signal_labels, is_pu


def label_colors(associations, cmap_name="tab20"):
    valid_labels = sorted(set(np.concatenate([np.asarray(a).ravel() for a in associations])) - {-1})
    base_cmap = plt.get_cmap(cmap_name, max(len(valid_labels), 1))
    label_to_color = {lab: base_cmap(i) for i, lab in enumerate(valid_labels)}
    label_to_color[-1] = (0.58, 0.58, 0.58, 0.38)
    return label_to_color


def plot_poster_associations(plot_associations, features, associations, names, output_folder, file_suffix):
    os.makedirs(output_folder, exist_ok=True)
    x, y = plot_associations.eta_phi_to_xy(features[:, 0], features[:, 1], features[:, 2])
    x = x.numpy()
    y = y.numpy()
    z = features[:, 2].numpy()
    label_to_color = label_colors(associations)

    fig = plt.figure(figsize=(10.5, 4.7), constrained_layout=True)
    fig.patch.set_alpha(0.0)

    for idx, (assoc, name) in enumerate(zip(associations, names)):
        ax = fig.add_subplot(1, len(associations), idx + 1, projection="3d")
        labels = np.asarray(assoc)
        colors = np.asarray([label_to_color[int(label)] for label in labels])
        sizes = np.where(labels == -1, 12, 26)

        ax.scatter(z, y, x, c=colors, s=sizes, linewidth=0, depthshade=False)
        ax.view_init(elev=22, azim=-64)
        ax.set_xlabel("z [cm]", labelpad=8)
        ax.set_ylabel("y [cm]", labelpad=8)
        ax.set_zlabel("x [cm]", labelpad=8)
        ax.set_title(name, pad=12, fontsize=17)
        ax.grid(True, alpha=0.22)
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
        ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    png_path = osp.join(output_folder, file_suffix + "_poster_assoc.png")
    pdf_path = osp.join(output_folder, file_suffix + "_poster_assoc.pdf")
    fig.savefig(png_path, dpi=450, bbox_inches="tight", transparent=True)
    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    input_file = args.input_file or default_input_file(args)
    output_folder = args.output_folder or default_output_folder(args)
    file_suffix = args.file_suffix or f"{args.run_name}_{args.split}_{args.file_index:05d}_event_{args.event_index}"

    os.makedirs(output_folder, exist_ok=True)
    plot_associations = load_plot_associations(args.plot_associations)

    events = ak.from_parquet(input_file)
    event = events[args.event_index]
    features, all_labels, signal_labels, is_pu = event_to_features_and_labels(event, signal_only=args.signal_only)

    label_sets = [all_labels] if args.signal_only else [all_labels, signal_labels]
    names = ["all"] if args.signal_only else ["all", "signal, PU grey"]
    if args.poster:
        names = ["All tracksters"] if args.signal_only else ["All tracksters", "Signal truth, PU grey"]
        output_paths = plot_poster_associations(
            plot_associations,
            features,
            label_sets,
            names,
            output_folder,
            file_suffix,
        )
    else:
        plot_associations.compare_associations(
            features,
            label_sets,
            names,
            output_folder=output_folder,
            file_suffix=file_suffix,
        )
        output_paths = [osp.join(output_folder, file_suffix + "_scatter_assoc.png")]
        plt.close("all")

    n_tracksters = int(len(all_labels))
    n_signal = int((~is_pu).sum())
    n_labels = int(len(set(all_labels.tolist()) - {-1}))
    print(f"Input: {input_file}")
    for output_path in output_paths:
        print(f"Output: {output_path}")
    print(f"Tracksters plotted: {n_tracksters}; signal: {n_signal}; labels: {n_labels}")


if __name__ == "__main__":
    main()
