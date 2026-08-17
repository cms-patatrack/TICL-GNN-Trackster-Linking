import argparse
import json
import os
import os.path as osp
import random
import sys
import tempfile
import multiprocessing as mp
from glob import glob

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, osp.join(REPO_ROOT, "tracksterLinker"))
os.environ.setdefault("MPLCONFIGDIR", osp.join(tempfile.gettempdir(), "matplotlib"))

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader.dataloader import DataLoader

from tracksterLinker.datasets.DummyDataset import DummyDataset
from tracksterLinker.datasets.GNNDataset import GNNDataset
from tracksterLinker.datasets.ProcessedGraphDataset import ProcessedGraphDataset
from tracksterLinker.GNN.LossFunctions import CombinedLoss, FocalLossLogits
from tracksterLinker.GNN.TrackLinkingNet import GNN_TrackLinkingNet, weight_init
from tracksterLinker.GNN.train import run_gnn_training, validate
from tracksterLinker.multiGNN.PUNet import PUNet
from tracksterLinker.utils.graphUtils import negative_edge_imbalance, print_graph_statistics
from tracksterLinker.utils.hgcalDummy import HGCALLikeDummyConfig, write_dataset
from tracksterLinker.utils.plotResults import plot_metric_bars, plot_training_comparison
from tracksterLinker.utils.reco_metrics import (
    edge_classification_metrics,
    evaluate_model_reconstruction,
    evaluate_unlinked_reconstruction,
    find_best_edge_threshold,
    write_metric_csv,
)


# Dummy outputs default to ../data while keeping the existing training_data/linking_dataset layout.
base_folder = osp.abspath(osp.join(REPO_ROOT, "..", "data"))
run_name = "dummy_reco_experiment"
model_folder = osp.join(base_folder, "training_data", run_name)
data_folder = osp.join(base_folder, "linking_dataset", run_name)
raw_data_folder = osp.join(data_folder, "histo")
data_folder_training = osp.join(data_folder, "dataset_dummy_reco")
data_folder_val = osp.join(data_folder, "dataset_dummy_reco_val")
data_folder_test = osp.join(data_folder, "dataset_dummy_reco_test")
signal_mean = 20.0
pu_mean = 200.0
close_pair_fraction = 0.85


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the dummy reconstruction experiment with the same training, "
            "checkpointing, and plotting utilities used by the real GNN scripts."
        )
    )
    parser.add_argument("--base-folder", default=base_folder, help="Root folder for dummy data and outputs. Default: ../data.")
    parser.add_argument("--run-name", default=run_name)
    parser.add_argument("--model-folder", "--work-dir", dest="model_folder", default=None, help="Override model/plot output folder.")
    parser.add_argument(
        "--data-folder",
        "--data-dir",
        dest="data_folder",
        default=None,
        help="Override dataset root.",
    )
    parser.add_argument("--raw-data-dir", default=None, help="Override the raw train/val/test input directory.")
    parser.add_argument("--processed-data-dir", default=None, help="Override the processed dataset parent directory.")
    parser.add_argument("--generate-data", action="store_true", help="Generate HGCAL-like dummy parquet data before training.")
    parser.add_argument(
        "--dataset-kind",
        choices=["dummy", "gnn", "processed"],
        default="dummy",
        help=(
            "Use DummyDataset for parquet-like dummy data, GNNDataset for ROOT files from "
            "TICL-HGCAL-Dummy-Data, or ProcessedGraphDataset for prebuilt PyG graphs."
        ),
    )
    parser.add_argument("--train-files", type=int, default=0)
    parser.add_argument("--val-files", type=int, default=0)
    parser.add_argument("--test-files", type=int, default=10)
    parser.add_argument("--events-per-file", type=int, default=10)
    parser.add_argument("--architecture", default="gnn", choices=["gnn", "punet"], help="Use the same architecture for both loss variants.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1, help="Keep at 1; repository edge_index tensors are stored as [E, 2].")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-lr", type=float, default=None)
    parser.add_argument("--contrastive-weight", type=float, default=1e-3)
    parser.add_argument("--focal-weight", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--plot-every", type=int, default=5, help="Save validation plots every N epochs and at the final epoch.")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save standard model checkpoints every N epochs and at the final epoch.")
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, or cpu. Defaults to cuda when available.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_subset(dataset, limit):
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def dummy_data_paths(args):
    use_script_paths = args.base_folder == base_folder and args.run_name == run_name
    default_model_folder = model_folder if use_script_paths else osp.join(args.base_folder, "training_data", args.run_name)
    default_data_folder = data_folder if use_script_paths else osp.join(args.base_folder, "linking_dataset", args.run_name)
    selected_model_folder = args.model_folder or default_model_folder
    selected_data_folder = args.data_folder or default_data_folder
    raw_data_dir = args.raw_data_dir or (raw_data_folder if use_script_paths and args.data_folder is None else osp.join(selected_data_folder, "histo"))

    if args.processed_data_dir is not None:
        train_folder = _processed_split_folder(args.processed_data_dir, "train")
        val_folder = _processed_split_folder(args.processed_data_dir, "val")
        test_folder = _processed_split_folder(args.processed_data_dir, "test")
    elif args.dataset_kind == "processed":
        train_folder = _processed_split_folder(selected_data_folder, "train")
        val_folder = _processed_split_folder(selected_data_folder, "val")
        test_folder = _processed_split_folder(selected_data_folder, "test")
    elif use_script_paths and args.data_folder is None:
        train_folder = data_folder_training
        val_folder = data_folder_val
        test_folder = data_folder_test
    else:
        train_folder = osp.join(selected_data_folder, "dataset_dummy_reco")
        val_folder = osp.join(selected_data_folder, "dataset_dummy_reco_val")
        test_folder = osp.join(selected_data_folder, "dataset_dummy_reco_test")

    return {
        "base": args.base_folder,
        "model": selected_model_folder,
        "data": selected_data_folder,
        "raw": raw_data_dir,
        "train": train_folder,
        "val": val_folder,
        "test": test_folder,
    }


def _processed_split_folder(parent, split):
    suffix = {"train": "", "val": "_val", "test": "_test"}[split]
    candidates = [
        osp.join(parent, f"dataset_colliderml_reco{suffix}"),
        osp.join(parent, f"dataset_dummy_reco{suffix}"),
    ]
    for candidate in candidates:
        if osp.isdir(candidate):
            return candidate
    return candidates[0]


def build_model(architecture, dataset, device):
    sample = dataset[0]
    input_dim = len(dataset.model_feature_keys)
    edge_dim = sample.edge_features.shape[1]
    kwargs = {
        "input_dim": input_dim,
        "edge_feature_dim": edge_dim,
        "niters": 4,
        "weighted_aggr": True,
        "dropout": 0.3,
        "node_scaler": dataset.node_scaler,
        "edge_scaler": dataset.edge_scaler,
    }
    if architecture == "punet":
        model = PUNet(edge_hidden_dim=64, hidden_dim=128, num_heads=8, **kwargs)
    else:
        model = GNN_TrackLinkingNet(edge_hidden_dim=32, hidden_dim=64, **kwargs)
    model.apply(weight_init)
    return model.to(device)


def write_model_metadata(model_output_dir, args, history, threshold, val_f1, model_name):
    metadata = {
        "model": model_name,
        "architecture": args.architecture,
        "threshold": float(threshold),
        "validation_weighted_edge_f1": float(val_f1),
        "history": history,
        "checkpoint": history.get("checkpoint"),
    }
    with open(osp.join(model_output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("This repository stores edge_index as [E, 2], so the dummy experiment requires --batch-size 1.")

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paths = dummy_data_paths(args)
    os.makedirs(paths["model"], exist_ok=True)

    if args.generate_data and args.dataset_kind == "gnn":
        raise ValueError(
            "--dataset-kind gnn expects ROOT files generated by TICL-HGCAL-Dummy-Data. "
            "Run that generator first and pass its train/val/test parent via --raw-data-dir."
        )

    if args.generate_data and args.dataset_kind == "processed":
        raise ValueError("--dataset-kind processed expects existing processed/data_*.pt graphs.")

    if args.dataset_kind == "dummy" and (args.generate_data or not osp.isdir(paths["raw"])):
        config = HGCALLikeDummyConfig(
            train_files=args.train_files,
            val_files=args.val_files,
            test_files=args.test_files,
            events_per_file=args.events_per_file,
            signal_mean=signal_mean,
            pu_mean=pu_mean,
            close_pair_fraction=close_pair_fraction,
            seed=args.seed,
        )
        print(f"Generating dummy data in {paths['raw']}")
        write_dataset(paths["raw"], config)

    if args.dataset_kind == "gnn":
        missing_splits = [split for split in ("train", "val", "test") if not glob(osp.join(paths["raw"], split, "*.root"))]
        if missing_splits:
            raise FileNotFoundError(
                f"Missing ROOT files for split(s) {missing_splits} under {paths['raw']}. "
                "Generate them with TICL-HGCAL-Dummy-Data first."
            )

    dataset_cls = {"gnn": GNNDataset, "dummy": DummyDataset, "processed": ProcessedGraphDataset}[args.dataset_kind]
    train_dataset = dataset_cls(
        paths["train"],
        paths["raw"],
        split="train",
        num_workers=args.num_workers,
        device=device,
    )
    val_dataset = dataset_cls(
        paths["val"],
        paths["raw"],
        split="val",
        node_scaler=train_dataset.node_scaler,
        edge_scaler=train_dataset.edge_scaler,
        num_workers=args.num_workers,
        device=device,
    )
    test_dataset = dataset_cls(
        paths["test"],
        paths["raw"],
        split="test",
        node_scaler=train_dataset.node_scaler,
        edge_scaler=train_dataset.edge_scaler,
        num_workers=args.num_workers,
        device=device,
    )

    print("Training split statistics:")
    print_graph_statistics(train_dataset)
    print("Validation split statistics:")
    print_graph_statistics(val_dataset)
    print("Test split statistics:")
    print_graph_statistics(test_dataset)

    train_data = maybe_subset(train_dataset, args.limit_train)
    val_data = maybe_subset(val_dataset, args.limit_val)
    test_data = maybe_subset(test_dataset, args.limit_test)
    train_dl = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_dl = DataLoader(val_data, shuffle=False, batch_size=args.batch_size)
    test_dl = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)

    alpha = 0.5 + float(negative_edge_imbalance(train_data)) / 2
    print(f"Using focal alpha={alpha:.4f}, gamma=2")

    histories = {}
    all_metrics = {}
    node_feature_dict = dataset_cls.node_feature_dict
    baseline_metrics = evaluate_unlinked_reconstruction(test_dl, node_feature_dict, selection="signal")
    all_metrics["unlinked_baseline"] = baseline_metrics

    model_specs = {
        "focal": {
            "loss": FocalLossLogits(alpha=alpha, gamma=2),
            "scores": False,
            "lr": args.lr,
            "optimizer": "adam",
        },
        "focal_contrastive": {
            "loss": CombinedLoss(
                alpha=alpha,
                gamma=2,
                margin=args.margin,
                weightFocal=args.focal_weight,
                weightContrastive=args.contrastive_weight,
            ),
            "scores": True,
            "lr": args.contrastive_lr or args.lr,
            "optimizer": "adamw",
        },
    }

    for model_name, spec in model_specs.items():
        set_seed(args.seed)
        model_output_dir = osp.join(paths["model"], model_name)
        model = build_model(args.architecture, train_dataset, device)
        if spec["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=spec["lr"],
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.01,
                amsgrad=True,
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=spec["lr"])

        focal_loss = spec["loss"].focal if spec["scores"] else spec["loss"]
        history = run_gnn_training(
            model,
            optimizer,
            train_dl,
            val_dl,
            args.epochs,
            loss_obj=spec["loss"],
            validation_loss_obj=focal_loss,
            output_folder=model_output_dir,
            dummy_input=train_dataset[0],
            scores=spec["scores"],
            weighted="raw_energy",
            node_feature_dict=node_feature_dict,
            device=device,
            plot_every=args.plot_every,
            checkpoint_every=args.checkpoint_every,
            early_stopping_patience=args.early_stopping_patience,
            log_prefix=model_name,
        )
        histories[model_name] = history

        _, val_scores, val_labels, val_weights, _ = validate(
            model,
            val_dl,
            history["epochs"],
            loss_obj=focal_loss,
            weighted="raw_energy",
            node_feature_dict=node_feature_dict,
        )
        threshold, val_f1 = find_best_edge_threshold(val_scores, val_labels, val_weights)
        model.threshold = threshold
        print(f"[{model_name}] validation threshold={threshold:.3f} weighted_edge_f1={val_f1:.4f}")

        _, test_scores, test_labels, test_weights, _ = validate(
            model,
            test_dl,
            history["epochs"],
            loss_obj=focal_loss,
            weighted="raw_energy",
            node_feature_dict=node_feature_dict,
        )
        metrics = edge_classification_metrics(test_scores, test_labels, test_weights, threshold)
        metrics.update(
            evaluate_model_reconstruction(
                model,
                test_dl,
                node_feature_dict,
                threshold=threshold,
                selection="signal",
            )
        )
        all_metrics[model_name] = metrics
        write_model_metadata(model_output_dir, args, history, threshold, val_f1, model_name)

    plot_training_comparison(histories, paths["model"], title="Dummy-data training comparison")
    plot_metric_bars(all_metrics, paths["model"], title="Held-out reconstruction metrics")
    write_metric_csv(all_metrics, paths["model"])
    with open(osp.join(paths["model"], "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)
    with open(osp.join(paths["model"], "config.json"), "w", encoding="utf-8") as handle:
        config = vars(args).copy()
        config["paths"] = paths
        config["dummy_generation"] = {
            "signal_mean": signal_mean,
            "pu_mean": pu_mean,
            "close_pair_fraction": close_pair_fraction,
        }
        json.dump(config, handle, indent=2)

    print(json.dumps(all_metrics, indent=2))
    print(f"Saved experiment outputs to {paths['model']}")
    print(f"Dummy data is stored under {paths['data']}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
