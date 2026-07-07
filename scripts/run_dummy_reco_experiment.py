import argparse
import csv
import json
import os
import os.path as osp
import random
import sys
from datetime import datetime

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, osp.join(REPO_ROOT, "tracksterLinker"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader.dataloader import DataLoader

from tracksterLinker.datasets.DummyDataset import DummyDataset
from tracksterLinker.GNN.LossFunctions import CombinedLoss, FocalLossLogits
from tracksterLinker.GNN.TrackLinkingNet import GNN_TrackLinkingNet, weight_init
from tracksterLinker.GNN.train import test, train, validate
from tracksterLinker.multiGNN.PUNet import PUNet
from tracksterLinker.utils.graphUtils import negative_edge_imbalance, print_graph_statistics
from tracksterLinker.utils.hgcalDummy import HGCALLikeDummyConfig, write_dataset
from tracksterLinker.utils.reco_metrics import (
    evaluate_model_reconstruction,
    evaluate_unlinked_reconstruction,
    find_best_edge_threshold,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run focal-vs-contrastive reconstruction experiments on HGCAL-like dummy data.")
    parser.add_argument("--work-dir", default="outputs/dummy_reco_experiment")
    parser.add_argument("--raw-data-dir", default=None, help="Directory containing train/val/test parquet splits.")
    parser.add_argument("--generate-data", action="store_true", help="Generate HGCAL-like dummy parquet data before training.")
    parser.add_argument("--scenario", default="mixed", choices=["mixed", "closeby_pions", "multiparticle", "single_particle_pu"])
    parser.add_argument("--train-files", type=int, default=80)
    parser.add_argument("--val-files", type=int, default=20)
    parser.add_argument("--test-files", type=int, default=20)
    parser.add_argument("--events-per-file", type=int, default=10)
    parser.add_argument("--signal-mean", type=float, default=12.0)
    parser.add_argument("--pu-mean", type=float, default=35.0)
    parser.add_argument("--close-pair-fraction", type=float, default=0.35)
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


def edge_metrics(scores, labels, weights, threshold):
    scores = scores.detach().cpu().float().reshape(-1)
    labels = (labels.detach().cpu().float().reshape(-1) > 0)
    weights = weights.detach().cpu().float().reshape(-1).clamp_min(0)
    pred = scores >= threshold
    tp = weights[pred & labels].sum()
    fp = weights[pred & ~labels].sum()
    fn = weights[~pred & labels].sum()
    tn = weights[~pred & ~labels].sum()
    precision = tp / (tp + fp).clamp_min(1e-12)
    recall = tp / (tp + fn).clamp_min(1e-12)
    specificity = tn / (tn + fp).clamp_min(1e-12)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    accuracy = (tp + tn) / (tp + fp + fn + tn).clamp_min(1e-12)
    return {
        "edge_accuracy": float(accuracy.item()),
        "edge_precision": float(precision.item()),
        "edge_recall": float(recall.item()),
        "edge_specificity": float(specificity.item()),
        "edge_f1": float(f1.item()),
        "threshold": float(threshold),
    }


def plot_training_curves(histories, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, history in histories.items():
        ax.plot(history["train_loss"], label=f"{name} train")
        ax.plot(history["val_loss"], label=f"{name} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Dummy-data training comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "loss_comparison.png"), dpi=180)
    plt.close(fig)


def plot_metric_bars(metrics, output_dir):
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
    ax.set_title("Held-out reconstruction metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "reconstruction_metric_bars.png"), dpi=180)
    plt.close(fig)


def write_metric_csv(metrics, output_dir):
    rows = []
    for model_name, values in metrics.items():
        for key, value in sorted(values.items()):
            rows.append({"model": model_name, "metric": key, "value": value})
    with open(osp.join(output_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def train_one_model(name, model, optimizer, loss_obj, train_dl, val_dl, epochs, scores):
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(1, epochs + 1):
        print(f"[{name}] Epoch {epoch}/{epochs}")
        train_loss = train(
            model,
            optimizer,
            train_dl,
            epoch,
            scores=scores,
            loss_obj=loss_obj,
            node_feature_dict=DummyDataset.node_feature_dict,
        )
        focal_loss = loss_obj.focal if scores else loss_obj
        val_loss, _, _, _ = test(
            model,
            val_dl,
            epoch,
            loss_obj=focal_loss,
            weighted="raw_energy",
            node_feature_dict=DummyDataset.node_feature_dict,
        )
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        print(f"[{name}] train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
    return {"train_loss": train_loss_hist, "val_loss": val_loss_hist}


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("This repository stores edge_index as [E, 2], so the dummy experiment currently requires --batch-size 1.")
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.work_dir, exist_ok=True)

    raw_data_dir = args.raw_data_dir or osp.join(args.work_dir, "parquet")
    if args.generate_data or not osp.isdir(raw_data_dir):
        config = HGCALLikeDummyConfig(
            scenario=args.scenario,
            train_files=args.train_files,
            val_files=args.val_files,
            test_files=args.test_files,
            events_per_file=args.events_per_file,
            signal_mean=args.signal_mean,
            pu_mean=args.pu_mean,
            close_pair_fraction=args.close_pair_fraction,
            seed=args.seed,
        )
        print(f"Generating dummy data in {raw_data_dir}")
        write_dataset(raw_data_dir, config)

    dataset_root = osp.join(args.work_dir, "processed")
    train_dataset = DummyDataset(
        osp.join(dataset_root, "train"),
        raw_data_dir,
        split="train",
        num_workers=args.num_workers,
        device=device,
    )
    val_dataset = DummyDataset(
        osp.join(dataset_root, "val"),
        raw_data_dir,
        split="val",
        node_scaler=train_dataset.node_scaler,
        edge_scaler=train_dataset.edge_scaler,
        num_workers=args.num_workers,
        device=device,
    )
    test_dataset = DummyDataset(
        osp.join(dataset_root, "test"),
        raw_data_dir,
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
    baseline_metrics = evaluate_unlinked_reconstruction(test_dl, DummyDataset.node_feature_dict, selection="signal")
    all_metrics["unlinked_baseline"] = baseline_metrics

    model_specs = {
        "focal": {
            "loss": FocalLossLogits(alpha=alpha, gamma=2),
            "scores": False,
            "lr": args.lr,
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
        },
    }

    for model_name, spec in model_specs.items():
        set_seed(args.seed)
        model_output_dir = osp.join(args.work_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        model = build_model(args.architecture, train_dataset, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=spec["lr"])
        history = train_one_model(model_name, model, optimizer, spec["loss"], train_dl, val_dl, args.epochs, scores=spec["scores"])
        histories[model_name] = history

        focal_loss = spec["loss"].focal if spec["scores"] else spec["loss"]
        _, val_scores, val_labels, val_weights, _ = validate(
            model,
            val_dl,
            args.epochs,
            loss_obj=focal_loss,
            weighted="raw_energy",
            node_feature_dict=DummyDataset.node_feature_dict,
        )
        threshold, val_f1 = find_best_edge_threshold(val_scores, val_labels, val_weights)
        model.threshold = threshold
        print(f"[{model_name}] validation threshold={threshold:.3f} weighted_edge_f1={val_f1:.4f}")

        _, test_scores, test_labels, test_weights, _ = validate(
            model,
            test_dl,
            args.epochs,
            loss_obj=focal_loss,
            weighted="raw_energy",
            node_feature_dict=DummyDataset.node_feature_dict,
        )
        metrics = edge_metrics(test_scores, test_labels, test_weights, threshold)
        metrics.update(evaluate_model_reconstruction(model, test_dl, DummyDataset.node_feature_dict, threshold=threshold, selection="signal"))
        all_metrics[model_name] = metrics

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "threshold": threshold,
                "architecture": args.architecture,
                "history": history,
                "metrics": metrics,
            },
            osp.join(model_output_dir, f"{model_name}_{datetime.now():%Y%m%d_%H%M%S}.pt"),
        )

    plot_training_curves(histories, args.work_dir)
    plot_metric_bars(all_metrics, args.work_dir)
    write_metric_csv(all_metrics, args.work_dir)
    with open(osp.join(args.work_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)
    with open(osp.join(args.work_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    print(json.dumps(all_metrics, indent=2))
    print(f"Saved experiment outputs to {args.work_dir}")


if __name__ == "__main__":
    main()
