import csv
import os.path as osp

import torch
import awkward as ak
import numpy as np

from scipy.optimize import linear_sum_assignment


def to_idx(x, device):
    if isinstance(x, torch.Tensor):
        return x.long().to(device).reshape(-1)
    return ak.to_torch(x).long().to(device).reshape(-1)


def to_float(x, device):
    if isinstance(x, torch.Tensor):
        return x.float().to(device).reshape(-1)
    return ak.to_torch(x).float().to(device).reshape(-1)


def best_associations(score_matrix: torch.Tensor, threshold: float):
    best_score, best_idx = score_matrix.min(dim=1)
    valid = best_score < threshold

    assoc = torch.full((score_matrix.shape[0],), -1, dtype=torch.long, device=score_matrix.device)
    assoc[valid] = best_idx[valid]

    return assoc


def efficiency(sim_to_reco: torch.Tensor):
    # fraction of sim tracksters matched to at least one reco trackster
    return (sim_to_reco >= 0).float().mean()


def fake_rate(reco_to_sim: torch.Tensor):
    # fraction of reco tracksters with no good sim match
    return (reco_to_sim < 0).float().mean()


def split_rate(reco_to_sim: torch.Tensor, num_sim: int):
    valid = reco_to_sim >= 0
    counts = torch.bincount(reco_to_sim[valid], minlength=num_sim)
    return (counts > 1).float().sum() / num_sim


def duplicate_rate(sim_to_reco: torch.Tensor, num_reco: int):
    # several sim tracksters matched to the same reco trackster
    valid = sim_to_reco >= 0
    counts = torch.bincount(sim_to_reco[valid], minlength=num_reco)
    return (counts > 1).float().sum() / num_reco


def metrics_from_score_matrices(reco_to_sim: torch.Tensor, sim_to_reco: torch.Tensor, threshold: float = 0.2):
    reco_to_sim_vec = best_associations(reco_to_sim, threshold)
    sim_to_reco_vec = best_associations(sim_to_reco, threshold)

    num_reco = reco_to_sim.shape[0]
    num_sim = sim_to_reco.shape[0]

    return {
        "reco_to_sim": reco_to_sim_vec,
        "sim_to_reco": sim_to_reco_vec,
        "efficiency": efficiency(sim_to_reco_vec),
        "fake_rate": fake_rate(reco_to_sim_vec),
        "split_rate": split_rate(reco_to_sim_vec, num_sim),
        "duplicate_rate": duplicate_rate(sim_to_reco_vec, num_reco),
    }


def missing_energy_score(source_idx, target_idx, energies, source_weight=None, target_coverage=None):
    source_idx = to_idx(source_idx, energies.device)
    target_idx = to_idx(target_idx, energies.device)

    source_energy = energies[source_idx]

    if source_weight is not None:
        source_weight = to_float(source_weight, energies.device)
        source_energy = source_energy * source_weight

    max_idx = int(torch.max(torch.cat([source_idx, target_idx])).item()) + 1
    coverage = torch.zeros(max_idx, dtype=energies.dtype, device=energies.device)

    if target_coverage is None:
        coverage[target_idx] = 1.0
    else:
        target_coverage = to_float(target_coverage, energies.device)
        coverage.scatter_reduce_(0, target_idx, target_coverage, reduce="amax", include_self=True)

    covered = coverage[source_idx].clamp(0, 1)
    missing_energy = source_energy * (1.0 - covered)

    denom = source_energy.square().sum()
    if denom == 0:
        return torch.tensor(float("nan"), dtype=energies.dtype, device=energies.device)

    return missing_energy.square().sum() / denom


def binary_pairwise_sim_to_reco_scores(sim_clusters, reco_clusters, energies, multiplicity):
    scores = torch.zeros((len(sim_clusters), len(reco_clusters)))

    for s, sim_idx in enumerate(sim_clusters):
        sim_weight = 1.0 / to_float(multiplicity[s], energies.device).clamp_min(1)
        for t, reco_idx in enumerate(reco_clusters):
            scores[s, t] = missing_energy_score(sim_idx, reco_idx, energies, source_weight=sim_weight)

    return scores


def binary_pairwise_reco_to_sim_scores(sim_clusters, reco_clusters, energies, multiplicity):
    scores = torch.zeros((len(sim_clusters), len(reco_clusters)))

    for s, sim_idx in enumerate(sim_clusters):
        sim_coverage = 1.0 / to_float(multiplicity[s], energies.device).clamp_min(1)
        for t, reco_idx in enumerate(reco_clusters):
            scores[s, t] = missing_energy_score(reco_idx, sim_idx, energies, target_coverage=sim_coverage)

    return scores


def build_associators(reco_clusters, sim_clusters, multiplicity, energies):
    reco_to_sim = binary_pairwise_reco_to_sim_scores(sim_clusters, reco_clusters, energies, multiplicity)
    sim_to_reco = binary_pairwise_sim_to_reco_scores(sim_clusters, reco_clusters, energies, multiplicity)
    return reco_to_sim, sim_to_reco


def get_top_associations(reco_clusters, sim_clusters, multiplicity, energy, all_recos=False):
    _, sim_to_reco = build_associators(reco_clusters, sim_clusters, multiplicity, energy)

    if all_recos:
        return sim_to_reco.argmin(dim=0)

    _, sim_assoc = linear_sum_assignment(sim_to_reco)
    top_assoc = torch.full((sim_to_reco.shape[1],), -1)
    top_assoc[sim_assoc] = torch.arange(sim_assoc.shape[0])
    return top_assoc.long()


def _edge_index_as_pairs(edge_index):
    edge_index = edge_index.detach().cpu().long()
    if edge_index.ndim != 2:
        raise ValueError("edge_index must be a 2D tensor")
    if edge_index.shape[1] == 2:
        return edge_index
    if edge_index.shape[0] == 2:
        return edge_index.t().contiguous()
    raise ValueError(f"Cannot interpret edge_index with shape {tuple(edge_index.shape)}")


def connected_components_from_edges(edge_index, num_nodes, edge_mask=None):
    """Build undirected connected components from the selected graph edges."""
    edges = _edge_index_as_pairs(edge_index)
    if edge_mask is not None:
        edge_mask = edge_mask.detach().cpu().bool().reshape(-1)
        edges = edges[edge_mask]

    parent = list(range(int(num_nodes)))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        root_left = find(int(left))
        root_right = find(int(right))
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edges.tolist():
        union(left, right)

    groups = {}
    for node in range(int(num_nodes)):
        groups.setdefault(find(node), []).append(node)
    return list(groups.values())


def singleton_components(num_nodes):
    return [[node] for node in range(int(num_nodes))]


def component_labels(components, num_nodes):
    labels = torch.full((int(num_nodes),), -1, dtype=torch.long)
    for comp_idx, component in enumerate(components):
        labels[torch.as_tensor(component, dtype=torch.long)] = comp_idx
    return labels


def _selection_mask(labels, is_pu=None, selection="signal"):
    labels = labels.detach().cpu().long().reshape(-1)
    selected = labels >= 0
    if selection == "all" or is_pu is None:
        return selected

    is_pu = is_pu.detach().cpu().bool().reshape(-1)
    if selection == "signal":
        return selected & ~is_pu
    if selection == "pu":
        return selected & is_pu
    raise ValueError("selection must be one of: all, signal, pu")


def truth_components_from_labels(labels, is_pu=None, selection="signal"):
    labels = labels.detach().cpu().long().reshape(-1)
    selected = _selection_mask(labels, is_pu=is_pu, selection=selection)
    components = []
    for label in torch.unique(labels[selected]).tolist():
        nodes = torch.nonzero(selected & (labels == int(label)), as_tuple=False).reshape(-1)
        if nodes.numel() > 0:
            components.append(nodes.tolist())
    return components


def energy_weighted_b_cubed(labels, pred_components, energies, is_pu=None, selection="signal", eps=1e-12):
    labels = labels.detach().cpu().long().reshape(-1)
    energies = energies.detach().cpu().float().reshape(-1).clamp_min(0)
    selected = _selection_mask(labels, is_pu=is_pu, selection=selection)
    selected_idx = torch.nonzero(selected, as_tuple=False).reshape(-1)
    if selected_idx.numel() == 0:
        return {"b3_precision": float("nan"), "b3_recall": float("nan"), "b3_f1": float("nan")}

    pred_label = component_labels(pred_components, labels.numel())
    total_energy = energies[selected].sum().clamp_min(eps)
    precision = torch.tensor(0.0)
    recall = torch.tensor(0.0)

    for node in selected_idx.tolist():
        same_pred = pred_label == pred_label[node]
        same_truth = selected & (labels == labels[node])
        overlap_energy = energies[same_pred & same_truth].sum()
        pred_energy = energies[same_pred].sum().clamp_min(eps)
        truth_energy = energies[same_truth].sum().clamp_min(eps)
        weight = energies[node] / total_energy
        precision += weight * overlap_energy / pred_energy
        recall += weight * overlap_energy / truth_energy

    f1 = 2 * precision * recall / (precision + recall).clamp_min(eps)
    return {
        "b3_precision": float(precision.item()),
        "b3_recall": float(recall.item()),
        "b3_f1": float(f1.item()),
    }


def reconstruction_metrics_from_components(
    pred_components,
    labels,
    energies,
    is_pu=None,
    selection="signal",
    containment_threshold=0.20,
    association_iou_threshold=0.20,
    fake_purity_threshold=0.20,
    split_fraction_threshold=0.20,
    eps=1e-12,
):
    labels = labels.detach().cpu().long().reshape(-1)
    energies = energies.detach().cpu().float().reshape(-1).clamp_min(0)
    selected = _selection_mask(labels, is_pu=is_pu, selection=selection)
    truth_components = truth_components_from_labels(labels, is_pu=is_pu, selection=selection)
    pred_components = [component for component in pred_components if torch.as_tensor(component, dtype=torch.long).numel() > 0]
    pred_components = [
        component
        for component in pred_components
        if selected[torch.as_tensor(component, dtype=torch.long)].any()
    ]

    out = {
        "n_truth": float(len(truth_components)),
        "n_pred": float(len(pred_components)),
    }
    if len(truth_components) == 0 or len(pred_components) == 0:
        out.update(
            {
                "containment_efficiency_40": float("nan"),
                "association_iou_efficiency": float("nan"),
                "fake_rate": float("nan"),
                "duplicate_rate": float("nan"),
                "split_rate": float("nan"),
                "merge_rate": float("nan"),
                "pu_contamination_rate": float("nan"),
                "mean_best_iou": float("nan"),
                "energy_weighted_iou": float("nan"),
                "energy_response_mean": float("nan"),
                "energy_response_std": float("nan"),
            }
        )
        out.update(energy_weighted_b_cubed(labels, pred_components or singleton_components(labels.numel()), energies, is_pu=is_pu, selection=selection))
        return out

    truth_energy = torch.tensor([energies[torch.as_tensor(component, dtype=torch.long)].sum() for component in truth_components]).float()
    pred_energy = torch.tensor([energies[torch.as_tensor(component, dtype=torch.long)].sum() for component in pred_components]).float()
    overlap = torch.zeros((len(truth_components), len(pred_components)), dtype=torch.float32)
    label_to_truth = {
        int(labels[component[0]].item()): truth_idx
        for truth_idx, component in enumerate(truth_components)
    }

    for pred_idx, component in enumerate(pred_components):
        for node in component:
            node = int(node)
            if selected[node] and int(labels[node].item()) in label_to_truth:
                overlap[label_to_truth[int(labels[node].item())], pred_idx] += energies[node]

    containment = overlap / truth_energy[:, None].clamp_min(eps)
    pred_purity = overlap / pred_energy[None, :].clamp_min(eps)
    union = truth_energy[:, None] + pred_energy[None, :] - overlap
    iou = overlap / union.clamp_min(eps)

    best_iou, best_pred = iou.max(dim=1)
    best_containment = containment.max(dim=1).values
    best_pred_energy = pred_energy[best_pred]
    energy_response = best_pred_energy / truth_energy.clamp_min(eps)

    fake_rate_value = (pred_purity.max(dim=0).values < fake_purity_threshold).float().mean()
    split_rate_value = ((containment >= split_fraction_threshold).sum(dim=1) > 1).float().mean()
    duplicate_rate_value = split_rate_value
    merge_rate_value = ((pred_purity >= split_fraction_threshold).sum(dim=0) > 1).float().mean()

    if is_pu is not None and selection == "signal":
        is_pu_cpu = is_pu.detach().cpu().bool().reshape(-1)
        contamination = []
        for component in pred_components:
            nodes = torch.as_tensor(component, dtype=torch.long)
            total = energies[nodes].sum().clamp_min(eps)
            contamination.append(float((energies[nodes] * is_pu_cpu[nodes].float()).sum() / total))
        pu_contamination_rate = float(np.mean([value > 0.10 for value in contamination])) if contamination else float("nan")
    else:
        pu_contamination_rate = float("nan")

    out.update(
        {
            "containment_efficiency_40": float((best_containment >= containment_threshold).float().mean().item()),
            "association_iou_efficiency": float((best_iou >= association_iou_threshold).float().mean().item()),
            "fake_rate": float(fake_rate_value.item()),
            "duplicate_rate": float(duplicate_rate_value.item()),
            "split_rate": float(split_rate_value.item()),
            "merge_rate": float(merge_rate_value.item()),
            "pu_contamination_rate": pu_contamination_rate,
            "mean_best_iou": float(best_iou.mean().item()),
            "energy_weighted_iou": float((best_iou * truth_energy).sum().item() / truth_energy.sum().clamp_min(eps).item()),
            "energy_response_mean": float(energy_response.mean().item()),
            "energy_response_std": float(energy_response.std(unbiased=False).item()),
        }
    )
    out.update(energy_weighted_b_cubed(labels, pred_components, energies, is_pu=is_pu, selection=selection))
    return out


def aggregate_metric_dicts(metrics):
    if not metrics:
        return {}
    keys = sorted({key for item in metrics for key in item})
    out = {}
    for key in keys:
        values = np.asarray([item[key] for item in metrics if key in item], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            out[key] = float("nan")
        else:
            out[key] = float(values.mean())
            out[f"{key}_std"] = float(values.std())
    return out


def evaluate_components_loader(loader, component_builder, node_feature_dict, selection="signal"):
    metrics = []
    energy_index = node_feature_dict["raw_energy"]
    for sample in loader:
        num_nodes = sample.x.shape[0]
        components = component_builder(sample)
        metrics.append(
            reconstruction_metrics_from_components(
                components,
                sample.cluster,
                sample.x[:, energy_index],
                is_pu=getattr(sample, "isPU", None),
                selection=selection,
            )
        )
    return aggregate_metric_dicts(metrics)


def evaluate_unlinked_reconstruction(loader, node_feature_dict, selection="signal"):
    return evaluate_components_loader(
        loader,
        lambda sample: singleton_components(sample.x.shape[0]),
        node_feature_dict,
        selection=selection,
    )


def evaluate_model_reconstruction(model, loader, node_feature_dict, threshold=None, selection="signal"):
    model.eval()
    if threshold is None:
        threshold = model.threshold
    device = next(model.parameters()).device

    def model_components(sample):
        with torch.no_grad():
            sample = sample.to(device)
            _, logits = model.run(sample.x, sample.edge_features, sample.edge_index)
            scores = model.scale(logits).squeeze(-1)
            return connected_components_from_edges(sample.edge_index, sample.x.shape[0], scores > threshold)

    return evaluate_components_loader(loader, model_components, node_feature_dict, selection=selection)


def edge_classification_metrics(scores, labels, weights, threshold):
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


def find_best_edge_threshold(scores, labels, weights=None, thresholds=None, beta=1.0):
    scores = scores.detach().cpu().float().reshape(-1)
    labels = (labels.detach().cpu().float().reshape(-1) > 0).float()
    if weights is None:
        weights = torch.ones_like(scores)
    else:
        weights = weights.detach().cpu().float().reshape(-1).clamp_min(0)
    if thresholds is None:
        thresholds = torch.linspace(0.05, 0.95, 19)

    best_threshold = float(thresholds[0])
    best_f = -1.0
    beta2 = beta * beta
    for threshold in thresholds:
        pred = scores >= threshold
        truth = labels.bool()
        tp = weights[pred & truth].sum()
        fp = weights[pred & ~truth].sum()
        fn = weights[~pred & truth].sum()
        precision = tp / (tp + fp).clamp_min(1e-12)
        recall = tp / (tp + fn).clamp_min(1e-12)
        f_score = (1 + beta2) * precision * recall / (beta2 * precision + recall).clamp_min(1e-12)
        if f_score.item() > best_f:
            best_f = float(f_score.item())
            best_threshold = float(threshold.item())
    return best_threshold, best_f


def write_metric_csv(metrics, output_dir, filename="metrics.csv"):
    rows = []
    for model_name, values in metrics.items():
        for key, value in sorted(values.items()):
            rows.append({"model": model_name, "metric": key, "value": value})
    with open(osp.join(output_dir, filename), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
