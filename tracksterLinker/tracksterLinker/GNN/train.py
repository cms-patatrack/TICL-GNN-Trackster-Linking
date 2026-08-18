from datetime import datetime
import glob
import os
import os.path as osp

from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from tracksterLinker.GNN.LossFunctions import FocalLoss
from tracksterLinker.GNN.TrackLinkingNet import EarlyStopping
from tracksterLinker.datasets.GNNDataset import GNNDataset
from tracksterLinker.utils.dataUtils import calc_weights
from tracksterLinker.utils.dataStatistics import plot_loss, save_model
from tracksterLinker.utils.plotResults import (
    get_best_threshold,
    plot_binned_validation_results,
    plot_validation_results,
    print_acc_scores_from_precalc,
)
from tracksterLinker.utils.perturbations.inErrorBars import *


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return next(model.buffers()).device


def _sample_to_model_device(sample, model):
    return sample.to(_model_device(model))


def train(model, opt, loader, epoch, weighted="raw_energy", scores=False, emb_out=False, loss_obj=FocalLoss(), node_feature_dict=GNNDataset.node_feature_dict, device=None):

    epoch_loss = 0

    model.train()
    step = 1
    last_loss = 0
    for sample in tqdm(loader, desc=f"Training Epoch {epoch}"):
        sample = sample.to(device or _model_device(model))

        # reset optimizer and enable training mode
        opt.zero_grad(set_to_none=True)
        emb, z = model.run(sample.x, sample.edge_features, sample.edge_index)
        weights = calc_weights(sample.edge_index, sample.x, node_feature_dict, name=weighted)
        
        # rescale weights to interval [0, 1]
        weights /= 300
        weights = torch.clamp(weights, 0.0, 1.0)
        weights = weights.detach()

        # compute the loss
        if scores:
            dupl = perturbate(sample.x, num_samples=1, with_z=True, device=sample.x.device)
            emb_pos, _ = model.run(dupl.squeeze(0), sample.edge_features, sample.edge_index)

            indices = torch.randperm(sample.edge_index.shape[0], device=sample.x.device)
            if sample.edge_index.shape[0] > 1 and torch.equal(indices, torch.arange(sample.edge_index.shape[0], device=sample.x.device)):
                indices = torch.roll(indices, shifts=1)
            emb_neg = emb[indices]

            contrastive_left = torch.cat([emb, emb], dim=0)
            contrastive_right = torch.cat([emb_pos, emb_neg], dim=0)
            contrastive_label = torch.cat(
                [
                    torch.zeros(sample.edge_index.shape[0], device=sample.x.device),
                    torch.ones(sample.edge_index.shape[0], device=sample.x.device),
                ],
                dim=0,
            )

            loss = loss_obj(z.squeeze(-1), contrastive_left, contrastive_right, sample.y, contrastive_label, weights)
        else:
            loss = loss_obj(z.squeeze(-1), torch.ceil(sample.y), weights)

        # back-propagate and update the weight
        if not torch.isfinite(loss): raise RuntimeError("Non-finite loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        # print(f"total grad norm: {grad_norm:.2f}")

        # skip update if grad_norm is suspiciously large
        if (not torch.isfinite(grad_norm)) or (epoch > 5 and grad_norm > 1e4):
            print(f"[WARN] Bad grad norm {grad_norm} at step {step}, skipping update.")
            opt.zero_grad(set_to_none=True)
            continue

        opt.step()
        epoch_loss += loss.item()

        if step % 1000 == 0:
            last_loss = epoch_loss/step
            print(f"Step loss: {last_loss}")
        step += 1

    return float(epoch_loss)/step


def test(model, loader, epoch, weighted="raw_energy", scores=False, loss_obj=FocalLoss(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), node_feature_dict=GNNDataset.node_feature_dict):

    with torch.set_grad_enabled(False):
        model.eval()
        device = torch.device(device or _model_device(model))
        val_loss = 0.0

        # 0: tp, 1: fp, 2: fn, 3: tn
        cross_edges = torch.zeros(4, device=device)
        signal_edges = torch.zeros(4, device=device)
        pu_edges = torch.zeros(4, device=device)
            
        for sample in tqdm(loader, desc=f"Validation Epoch {epoch}"):
            sample = sample.to(device)
            nn_emb, nn_pred = model.run(sample.x, sample.edge_features, sample.edge_index)
            weights = calc_weights(sample.edge_index, sample.x, node_feature_dict, name=weighted)

            y_pred = (model.scale(nn_pred) > model.threshold).squeeze()
            y_true = (sample.y > 0).squeeze()

            cross_edges[0] += torch.sum(weights[sample.PU_info[:, 0]] * (y_true[sample.PU_info[:, 0]] & y_pred[sample.PU_info[:, 0]])).item()
            cross_edges[1] += torch.sum(weights[sample.PU_info[:, 0]] * (~y_true[sample.PU_info[:, 0]] & y_pred[sample.PU_info[:, 0]])).item()
            cross_edges[2] += torch.sum(weights[sample.PU_info[:, 0]] * (y_true[sample.PU_info[:, 0]] & ~y_pred[sample.PU_info[:, 0]])).item()
            cross_edges[3] += torch.sum(weights[sample.PU_info[:, 0]] * (~y_true[sample.PU_info[:, 0]] & ~y_pred[sample.PU_info[:, 0]])).item()

            signal_edges[0] += torch.sum(weights[sample.PU_info[:, 1]] * (y_true[sample.PU_info[:, 1]] & y_pred[sample.PU_info[:, 1]])).item()
            signal_edges[1] += torch.sum(weights[sample.PU_info[:, 1]] * (~y_true[sample.PU_info[:, 1]] & y_pred[sample.PU_info[:, 1]])).item()
            signal_edges[2] += torch.sum(weights[sample.PU_info[:, 1]] * (y_true[sample.PU_info[:, 1]] & ~y_pred[sample.PU_info[:, 1]])).item()
            signal_edges[3] += torch.sum(weights[sample.PU_info[:, 1]] * (~y_true[sample.PU_info[:, 1]] & ~y_pred[sample.PU_info[:, 1]])).item()
            
            pu_edges[0] += torch.sum(weights[sample.PU_info[:, 2]] * (y_true[sample.PU_info[:, 2]] & y_pred[sample.PU_info[:, 2]])).item()
            pu_edges[1] += torch.sum(weights[sample.PU_info[:, 2]] * (~y_true[sample.PU_info[:, 2]] & y_pred[sample.PU_info[:, 2]])).item()
            pu_edges[2] += torch.sum(weights[sample.PU_info[:, 2]] * (y_true[sample.PU_info[:, 2]] & ~y_pred[sample.PU_info[:, 2]])).item()
            pu_edges[3] += torch.sum(weights[sample.PU_info[:, 2]] * (~y_true[sample.PU_info[:, 2]] & ~y_pred[sample.PU_info[:, 2]])).item()
            
            # rescale weights to interval [0, 1]
            weights /= 300
            weights = torch.clamp(weights, 0.0, 1.0)
            weights = weights.detach()

            if scores:
                loss = loss_obj(nn_pred.squeeze(-1), nn_emb.squeeze(-1), sample.y, sample.PU_info, weights).item()
            else:
                loss = loss_obj(nn_pred.squeeze(-1), sample.y, weights).item()
            val_loss += loss

        val_loss /= len(loader)
        return val_loss, cross_edges, signal_edges, pu_edges 


def validate(model, loader, epoch, weighted="raw_energy", scores=False, loss_obj=FocalLoss(), node_feature_dict=GNNDataset.node_feature_dict, device=None):

    with torch.set_grad_enabled(False):
        model.eval()
        device = torch.device(device or _model_device(model))
        val_loss = 0.0

        pred, y, weights = [], [], []
        PU_info = [[], [], []]
            
        for sample in tqdm(loader, desc=f"Validation Epoch {epoch}"):
            sample = sample.to(device)
            nn_emb, nn_pred = model.run(sample.x, sample.edge_features, sample.edge_index)
            pred += model.scale(nn_pred).squeeze(-1).tolist()
            y += sample.y.tolist()
            weight = calc_weights(sample.edge_index, sample.x, node_feature_dict, name=weighted)
            weights += weight.tolist()

            PU_info[0] += sample.PU_info[:, 0].tolist()
            PU_info[1] += sample.PU_info[:, 1].tolist()
            PU_info[2] += sample.PU_info[:, 2].tolist()
            
            # rescale weights to interval [0, 1]
            weight /= 300
            weight = torch.clamp(weight, 0.0, 1.0)
            weight = weight.detach()

            if scores:
                loss = loss_obj(nn_pred.squeeze(-1), nn_emb.squeeze(-1), sample.y, sample.PU_info, weight).item()
            else:
                loss = loss_obj(nn_pred.squeeze(-1), sample.y, weight).item()
            val_loss += loss

        val_loss /= len(loader)
    return val_loss, torch.tensor(pred), torch.tensor(y), torch.tensor(weights), torch.tensor(PU_info)


def latest_standard_checkpoint(output_folder):
    candidates = sorted(glob.glob(osp.join(output_folder, "*_dict.pt")), key=osp.getmtime)
    return candidates[-1] if candidates else None


def run_gnn_training(
    model,
    optimizer,
    train_loader,
    val_loader,
    epochs,
    loss_obj=FocalLoss(),
    validation_loss_obj=None,
    output_folder=None,
    filename=None,
    loss_plot_filename=None,
    dummy_input=None,
    start_epoch=0,
    scores=False,
    weighted="raw_energy",
    node_feature_dict=GNNDataset.node_feature_dict,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    scheduler=None,
    scheduler_eta_min=1e-6,
    early_stopping=None,
    early_stopping_patience=20,
    checkpoint_every=5,
    plot_every=10,
    epoch_label_offset=1,
    print_statistics=True,
    log_prefix=None,
):
    """Shared GNN training loop used by scripts/trainGNN*.py-style workflows."""
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    date = f"{datetime.now():%Y-%m-%d}"
    filename = filename or f"model_{date}"
    loss_plot_filename = loss_plot_filename or f"model_date_{date}_loss_epochs"
    validation_loss_obj = validation_loss_obj or loss_obj

    if scheduler is None:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, start_epoch + epochs), eta_min=scheduler_eta_min)

    if early_stopping is None and early_stopping_patience is not None:
        early_stopping = EarlyStopping(patience=early_stopping_patience, delta=0)

    train_loss_hist = []
    val_loss_hist = []
    final_epoch = start_epoch
    last_epoch = start_epoch + epochs
    prefix = f"[{log_prefix}] " if log_prefix else ""

    for epoch in range(start_epoch, last_epoch):
        loop_epoch = epoch + epoch_label_offset
        final_epoch = loop_epoch
        is_final_epoch = epoch + 1 == last_epoch
        print(f"{prefix}Epoch: {loop_epoch}")

        train_loss = train(
            model,
            optimizer,
            train_loader,
            loop_epoch,
            scores=scores,
            loss_obj=loss_obj,
            node_feature_dict=node_feature_dict,
            weighted=weighted,
            device=device,
        )
        train_loss_hist.append(train_loss)

        val_loss, cross_edges, signal_edges, pu_edges = test(
            model,
            val_loader,
            loop_epoch,
            loss_obj=validation_loss_obj,
            device=device,
            weighted=weighted,
            node_feature_dict=node_feature_dict,
        )
        val_loss_hist.append(val_loss)
        print(f"{prefix}Training loss: {train_loss}, Validation loss: {val_loss}, Learning Rate: {scheduler.get_last_lr()}")

        if output_folder is not None:
            plot_loss(
                train_loss_hist,
                val_loss_hist,
                save=True,
                output_folder=output_folder,
                filename=loss_plot_filename,
            )

        if print_statistics:
            print(f"{prefix}Fast statistic on model threshold:")
            print("Only cross selected:")
            print_acc_scores_from_precalc(*cross_edges)
            print("Only signal trackster:")
            print_acc_scores_from_precalc(*signal_edges)
            print("Only PU trackster:")
            print_acc_scores_from_precalc(*pu_edges)

        should_plot = plot_every > 0 and ((loop_epoch % plot_every == 0) or is_final_epoch)
        if should_plot and output_folder is not None:
            print(f"{prefix}Store Diagrams")
            _, pred, y, weight, _ = validate(
                model,
                val_loader,
                loop_epoch,
                loss_obj=validation_loss_obj,
                weighted=weighted,
                node_feature_dict=node_feature_dict,
                device=device,
            )
            threshold = get_best_threshold(pred, y, weight)
            model.threshold = threshold

            print(f"{prefix}weighted by raw energy:")
            plot_binned_validation_results(
                pred,
                y,
                weight,
                thres=threshold,
                output_folder=output_folder,
                file_suffix=f"epoch_{loop_epoch}_date_{date}",
            )
            plot_validation_results(
                pred,
                y,
                save=True,
                output_folder=output_folder,
                file_suffix=f"epoch_{loop_epoch}_date_{date}",
                weight=weight,
            )

        should_checkpoint = checkpoint_every > 0 and ((loop_epoch % checkpoint_every == 0) or is_final_epoch)
        if should_checkpoint and output_folder is not None:
            print(f"{prefix}Store Model")
            save_model(
                model,
                loop_epoch,
                optimizer,
                train_loss_hist,
                val_loss_hist,
                output_folder=output_folder,
                filename=filename,
                dummy_input=dummy_input,
            )

        if early_stopping is not None:
            early_stopping(model, val_loss)
            if early_stopping.early_stop:
                print(f"{prefix}Early stopping after {loop_epoch} epochs")
                early_stopping.load_best_model(model)
                if output_folder is not None:
                    save_model(
                        model,
                        loop_epoch,
                        optimizer,
                        train_loss_hist,
                        val_loss_hist,
                        output_folder=output_folder,
                        filename=f"{filename}_final_loss_{-early_stopping.best_score:.4f}",
                        dummy_input=dummy_input,
                    )
                break

        scheduler.step()
        plt.close()

    return {
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
        "epochs": final_epoch,
        "checkpoint": latest_standard_checkpoint(output_folder) if output_folder is not None else None,
    }
