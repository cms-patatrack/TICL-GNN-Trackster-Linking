import torch
import numpy as np

from tracksterLinker.datasets.NeoGNNDataset import NeoGNNDataset
from tracksterLinker.utils.graphUtils import *

def filter_nested_list(data, allowed):
    if isinstance(data, list):
        # recurse if list
        return [filter_nested_list(item, allowed) for item in data if not isinstance(item, list) and item in allowed or isinstance(item, list)]
    else:
        # keep only if in allowed
        return data if data in allowed else None

def calc_overlapping_components(graph_pred, node_values, true_components, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    node_energy = node_values[:, NeoGNNDataset.node_feature_dict["raw_energy"]]

    pred_components = find_connected_components(graph_pred, node_energy.shape[0], device=device)
    overlapping_components = []
    for component in true_components:
        filtered_components = ak.from_iter(filter_nested_list(pred_components, component))
        filtered_components = filtered_components[ak.num(filtered_components) > 0]

        energys = []
        for comp in filtered_components:
            energys.append(torch.sum(node_energy[comp]))
        
        idx = torch.argmax(torch.tensor(energys))
        overlapping_components.append(pred_components[idx])
    return overlapping_components, get_component_features(overlapping_components, node_values)


def component_dist(pred_component, true_component, node_features, cap_energy=None):
    T_energy = torch.sum(node_features[true_component, NeoGNNDataset.node_feature_dict["raw_energy"]])
    P_energy = torch.sum(node_features[pred_component, NeoGNNDataset.node_feature_dict["raw_energy"]])

    if (cap_energy is None):
        print("IMPLEMENT P cap T calculation")

    dU = T_energy - cap_energy
    dO = P_energy - cap_energy

    return dU, dO


def graph_dist(graph_true, graph_pred, node_features, pu, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), verbose=False):
    true_components = find_connected_components(graph_true, node_features.shape[0], device=device)
    pred_components = find_connected_components(graph_pred, node_features.shape[0], device=device)
    cnt_components = len(true_components)

    cnt_pu = 0
    cnt_signal = 0
    isPU = np.zeros(cnt_components, dtype=bool)

    res = {}
    dU_PU = np.zeros(cnt_components)
    dO_PU = np.zeros(cnt_components)
    dU_Signal = np.zeros(cnt_components)
    dO_Signal = np.zeros(cnt_components)

    energy_PU = 0
    energy_Signal = 0

    for j, component in enumerate(true_components):
        max_energy = 0
        max_idx = -1

        for i, comp in enumerate(pred_components):
            overlap = np.intersect1d(component, comp)
            if(overlap.shape[0] > 0):
                energy = torch.sum(node_features[overlap, NeoGNNDataset.node_feature_dict["raw_energy"]])
                if (max_energy < energy):
                    max_energy = energy 
                    max_idx = i
        
        comp_dU, comp_dO = component_dist(pred_components[max_idx], component, node_features, max_energy)

        if (pu[component[0]]):
            energy_PU += torch.sum(node_features[component, NeoGNNDataset.node_feature_dict["raw_energy"]]).item()
            dU_PU[cnt_pu] = comp_dU.item()
            dO_PU[cnt_pu] = comp_dO.item()
            cnt_pu += 1
            isPU[j] = 1
        else:
            energy_Signal += torch.sum(node_features[component, NeoGNNDataset.node_feature_dict["raw_energy"]]).item()
            dU_Signal[cnt_signal] = comp_dU.item()
            dO_Signal[cnt_signal] = comp_dO.item()
            cnt_signal += 1

    res["energy"] = energy_PU + energy_Signal
    res["energy_Signal"] = energy_Signal
    res["energy_PU"] = energy_PU
    res["dU_Signal"] = np.sum(dU_Signal)
    res["dO_Signal"] = np.sum(dO_Signal)
    res["dU_PU"] = np.sum(dU_PU)
    res["dO_PU"] = np.sum(dO_PU)

    if verbose:
        res["components"] = true_components
        res["features"] = get_component_features(true_components, node_features)

        res["comp_dU_Signal"] = dU_Signal[:cnt_signal]
        res["comp_dO_Signal"] = dO_Signal[:cnt_signal]
        res["comp_dU_PU"] = dU_PU[:cnt_pu]
        res["comp_dO_PU"] = dO_PU[:cnt_pu]
        res["isPU"] = isPU

    return res 

