import argparse
import glob
import os
import os.path as osp
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader.dataloader import DataLoader

from tracksterLinker.datasets.GNNDataset import GNNDataset
from tracksterLinker.datasets.NeoGNNDataset import NeoGNNDataset
from tracksterLinker.GNN.TrackLinkingNet import EarlyStopping, weight_init
from tracksterLinker.multiGNN.PUNet import PUNet
from tracksterLinker.GNN.LossFunctions import CombinedLoss
from tracksterLinker.GNN.train import train, test, validate
from tracksterLinker.utils.dataStatistics import plot_loss, save_model
from tracksterLinker.utils.graphUtils import negative_edge_imbalance
from tracksterLinker.utils.plotResults import (
    get_best_threshold,
    plot_binned_validation_results,
    plot_validation_results,
    print_acc_scores_from_precalc,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train or resume the contrastive TICL GNN.")
    parser.add_argument("--base-folder", default="/data/czeh")
    parser.add_argument("--model-folder", default=None, help="Output folder for checkpoints and plots.")
    parser.add_argument("--load-model-folder", default=None, help="Folder used when resolving relative checkpoint names.")
    parser.add_argument("--dataset-kind", choices=["neo", "gnn"], default="neo", help="Use old GNNTraining-tree data or the newer GNNDataset ROOT structure.")
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--test-data", default=None)
    parser.add_argument("--raw-data-dir", default=None, help="ROOT train/test parent for --dataset-kind gnn.")
    parser.add_argument("--num-workers", type=int, default=1, help="GNNDataset processing workers.")
    parser.add_argument("--resume-checkpoint", default=None, help="Checkpoint path or basename to continue from.")
    parser.add_argument("--resume-latest", action="store_true", help="Continue from latest_resume_state.pt, or the newest *_dict.pt in --model-folder.")
    parser.add_argument("--weights-checkpoint", default=None, help="Load model weights only, starting a new optimizer/history.")
    parser.add_argument("--no-resume-optimizer", action="store_true", help="Do not restore optimizer state when resuming.")
    parser.add_argument("--no-resume-history", action="store_true", help="Do not carry checkpoint loss history forward.")
    parser.add_argument("--start-epoch", type=int, default=5, help="Epoch to start from when not resuming.")
    parser.add_argument("--epochs", type=int, default=60, help="Additional epochs to run.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save full model artifacts every N epochs.")
    parser.add_argument("--resume-state-every", type=int, default=1, help="Save lightweight resume state every N epochs.")
    parser.add_argument("--resume-state-name", default="latest_resume_state.pt")
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, or cpu. Defaults to cuda when available.")
    parser.add_argument("--filename-prefix", default=None, help="Checkpoint filename prefix. Defaults to model_<date>.")
    return parser.parse_args()


def resolve_paths(args):
    model_folder = args.model_folder or osp.join(args.base_folder, "training_data/9999_CHANGE_TO_NEW_MODEL")
    default_train = "dataset_gnn" if args.dataset_kind == "gnn" else "dataset_hardronics"
    default_test = "dataset_gnn_test" if args.dataset_kind == "gnn" else "dataset_hardronics_test"
    return {
        "model": model_folder,
        "load_model": args.load_model_folder or osp.join(args.base_folder, "model_results/0002_model_large_contr_att"),
        "train": args.train_data or osp.join(args.base_folder, "linking_dataset", default_train),
        "test": args.test_data or osp.join(args.base_folder, "linking_dataset", default_test),
        "raw": args.raw_data_dir or osp.join(args.base_folder, "linking_dataset/histo"),
    }


def latest_checkpoint(folder):
    candidates = sorted(glob.glob(osp.join(folder, "*_dict.pt")), key=osp.getmtime)
    return candidates[-1] if candidates else None


def resolve_checkpoint(name_or_path, search_folder):
    if name_or_path is None:
        return None
    candidates = [name_or_path]
    if not name_or_path.endswith(".pt"):
        candidates.append(f"{name_or_path}.pt")
    candidates.extend(osp.join(search_folder, candidate) for candidate in list(candidates))
    for candidate in candidates:
        if osp.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find checkpoint {name_or_path!r} in current directory or {search_folder}")


def load_checkpoint(path, model, optimizer=None, *, restore_optimizer=True, restore_history=True, strict=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer is not None and restore_optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    train_loss_hist = list(checkpoint.get("training_loss", [])) if restore_history else []
    val_loss_hist = list(checkpoint.get("validation_loss", [])) if restore_history else []
    return checkpoint, start_epoch, train_loss_hist, val_loss_hist


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_resume_state(path, model, optimizer, scheduler, epoch, train_loss_hist, val_loss_hist):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_loss": train_loss_hist,
            "validation_loss": val_loss_hist,
        },
        path,
    )


def build_datasets(args, paths, device):
    if args.dataset_kind == "gnn":
        dataset_training = GNNDataset(
            paths["train"],
            paths["raw"],
            split="train",
            num_workers=args.num_workers,
            device=device,
        )
        dataset_test = GNNDataset(
            paths["test"],
            paths["raw"],
            split="test",
            node_scaler=dataset_training.node_scaler,
            edge_scaler=dataset_training.edge_scaler,
            num_workers=args.num_workers,
            device=device,
        )
        return dataset_training, dataset_test, GNNDataset.node_feature_dict

    dataset_training = NeoGNNDataset(paths["train"], only_signal=False, device=device)
    dataset_test = NeoGNNDataset(paths["test"], test=True, only_signal=False, device=device)
    return dataset_training, dataset_test, NeoGNNDataset.node_feature_dict


def main():
    args = parse_args()
    paths = resolve_paths(args)
    os.makedirs(paths["model"], exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    dataset_training, dataset_test, node_feature_dict = build_datasets(args, paths, device)
    train_dl = DataLoader(dataset_training, shuffle=True, batch_size=args.batch_size)
    test_dl = DataLoader(dataset_test, shuffle=True, batch_size=args.batch_size)
    print(f"Training Dataset: {len(train_dl)}, Test Dataset: {len(test_dl)}")

    model = PUNet(
        input_dim=len(dataset_training.model_feature_keys),
        edge_feature_dim=dataset_training[0].edge_features.shape[1],
        niters=4,
        edge_hidden_dim=64,
        hidden_dim=128,
        num_heads=8,
        weighted_aggr=True,
        dropout=0.3,
        node_scaler=dataset_training.node_scaler,
        edge_scaler=dataset_training.edge_scaler,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=True,
    )

    alpha = 0.5 + negative_edge_imbalance(dataset_test) / 2
    print(f"Focal loss with alpha={alpha}")
    loss_obj = CombinedLoss(alpha=alpha, gamma=2, margin=2.0, weightFocal=100, weightContrastive=0.0001)
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, delta=0)

    start_epoch = args.start_epoch
    train_loss_hist = []
    val_loss_hist = []
    resume_state_path = osp.join(paths["model"], args.resume_state_name)
    if args.resume_latest and osp.isfile(resume_state_path):
        resume_path = resume_state_path
    elif args.resume_latest:
        resume_path = latest_checkpoint(paths["model"])
    else:
        resume_path = resolve_checkpoint(args.resume_checkpoint, paths["load_model"])
    scheduler_state = None

    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint, start_epoch, train_loss_hist, val_loss_hist = load_checkpoint(
            resume_path,
            model,
            optimizer,
            restore_optimizer=not args.no_resume_optimizer,
            restore_history=not args.no_resume_history,
            strict=True,
        )
        scheduler_state = checkpoint.get("scheduler_state_dict")
        move_optimizer_state(optimizer, device)
    else:
        model.apply(weight_init)
        weights_path = resolve_checkpoint(args.weights_checkpoint, paths["load_model"])
        if weights_path is not None:
            print(f"Loading model weights only from: {weights_path}")
            load_checkpoint(weights_path, model, optimizer=None, restore_optimizer=False, restore_history=False, strict=False)

    date = f"{datetime.now():%Y-%m-%d}"
    filename_prefix = args.filename_prefix or f"model_{date}"
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, start_epoch + args.epochs), eta_min=1e-6)

    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    last_epoch = start_epoch + args.epochs
    print(f"Training from epoch {start_epoch} through {last_epoch - 1}")
    print(scheduler.get_last_lr())

    for epoch in range(start_epoch, last_epoch):
        print(f"Epoch: {epoch}")
        loss = train(model, optimizer, train_dl, epoch, loss_obj=loss_obj, scores=True, node_feature_dict=node_feature_dict)
        train_loss_hist.append(loss)

        val_loss, cross_edges, signal_edges, pu_edges = test(
            model,
            test_dl,
            epoch,
            loss_obj=loss_obj.focal,
            device=device,
            weighted="raw_energy",
            node_feature_dict=node_feature_dict,
        )
        val_loss_hist.append(val_loss)
        print(f"Training loss: {loss}, Validation loss: {val_loss}, Learning Rate: {scheduler.get_last_lr()}")

        plot_loss(train_loss_hist, val_loss_hist, save=True, output_folder=paths["model"], filename=f"model_date_{date}_loss_epochs")

        print("Fast statistic on model threshold:")
        print("Only cross selected:")
        print_acc_scores_from_precalc(*cross_edges)
        print("Only signal trackster:")
        print_acc_scores_from_precalc(*signal_edges)
        print("Only PU trackster:")
        print_acc_scores_from_precalc(*pu_edges)

        is_final_epoch = epoch + 1 == last_epoch
        if args.checkpoint_every > 0 and (epoch % args.checkpoint_every == 0 or is_final_epoch):
            print("Store Model")
            save_model(model, epoch, optimizer, train_loss_hist, val_loss_hist, output_folder=paths["model"], filename=filename_prefix, dummy_input=dataset_training[0])

        if args.plot_every > 0 and (epoch % args.plot_every == 0 or is_final_epoch):
            print("Store Diagrams")
            _, pred, y, weight, _ = validate(
                model,
                test_dl,
                epoch,
                loss_obj=loss_obj.focal,
                weighted="raw_energy",
                node_feature_dict=node_feature_dict,
            )
            threshold = get_best_threshold(pred, y, weight)
            model.threshold = threshold

            print("weighted by raw energy:")
            plot_binned_validation_results(pred, y, weight, thres=threshold, output_folder=paths["model"], file_suffix=f"epoch_{epoch}_date_{date}")
            plot_validation_results(pred, y, save=True, output_folder=paths["model"], file_suffix=f"epoch_{epoch}_date_{date}", weight=weight)

        early_stopping(model, val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping after {epoch} epochs")
            early_stopping.load_best_model(model)
            save_model(
                model,
                epoch,
                optimizer,
                train_loss_hist,
                val_loss_hist,
                output_folder=paths["model"],
                filename=f"{filename_prefix}_final_loss_{-early_stopping.best_score:.4f}",
                dummy_input=dataset_training[0],
            )
            save_resume_state(resume_state_path, model, optimizer, scheduler, epoch, train_loss_hist, val_loss_hist)
            break

        scheduler.step()
        if args.resume_state_every > 0 and (epoch % args.resume_state_every == 0 or is_final_epoch):
            print(f"Store resume state: {resume_state_path}")
            save_resume_state(resume_state_path, model, optimizer, scheduler, epoch, train_loss_hist, val_loss_hist)
        plt.close()


if __name__ == "__main__":
    main()
