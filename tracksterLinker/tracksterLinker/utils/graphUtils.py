import torch
import cupy as cp
import awkward as ak

from typing import List, Dict, Any
from collections import defaultdict

from tracksterLinker.datasets.NeoGNNDataset import NeoGNNDataset

class TileConstants:
    minEta = -cp.pi
    maxEta = cp.pi
    nPhiBins = 72  # Typically 0-2π mapped to bins


class Node:
    def __init__(self, idx):
        self.idx = idx
        self.inner = []
        self.outer = []

    def addInnerNeighbour(self, idx):
        self.inner.append(idx)

    def addOuterNeighbour(self, idx):
        self.outer.append(idx)


# Tile class similar to TICLLayerTile
class TICLLayerTile:
    def __init__(self):
        self.tiles = defaultdict(list)

    def fill(self, eta, phi, idx):
        eta_bin = int((eta - TileConstants.minEta) * 10)  # example binning
        phi_bin = int((phi + cp.pi) / (2 * cp.pi) * TileConstants.nPhiBins)
        self.tiles[(eta_bin, phi_bin)].append(idx)

    def __getitem__(self, bin_idx):
        return self.tiles.get(bin_idx, [])

    def globalBin(self, eta_idx, phi_idx):
        return (eta_idx, phi_idx)

    def searchBoxEtaPhi(self, eta_min, eta_max, phi_min, phi_max):
        eta_min_bin = int((eta_min - TileConstants.minEta) * 10)
        eta_max_bin = int((eta_max - TileConstants.minEta) * 10)
        phi_min_bin = int((phi_min + cp.pi) / (2 * cp.pi) * TileConstants.nPhiBins)
        phi_max_bin = int((phi_max + cp.pi) / (2 * cp.pi) * TileConstants.nPhiBins)
        return [eta_min_bin, eta_max_bin, phi_min_bin, phi_max_bin]


def build_subgraph(graph, root, neighborhood=1):
    neighbors = graph[1][graph[0] == root]

    if (neighborhood == 0):
        return neighbors
    subgraph = cp.copy(neighbors)

    for n in neighbors:
        subgraph = cp.append(subgraph, build_subgraph(graph, n, neighborhood-1))

    return cp.unique(subgraph)


def find_connected_components(graph, num_nodes, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    components = []

    for node in range(num_nodes):
        if not visited[node]:
            visited[node] = True
            component = [node]
            queue = torch.tensor([node], dtype=torch.long, device=device)

            while queue.numel() > 0:
                root = queue[0].item()
                queue = queue[1:]

                # Get neighbors (treat as undirected)
                out_neighbors = graph[:, 1][graph[:, 0] == root]
                in_neighbors = graph[:, 0][graph[:, 1] == root]
                neighbors = torch.cat((out_neighbors, in_neighbors))

                unvisited = neighbors[~visited[neighbors]]
                if unvisited.numel() > 0:
                    unique_unvisited = torch.unique(unvisited)
                    visited[unique_unvisited] = True
                    queue = torch.cat((queue, unique_unvisited))
                    component.extend(unique_unvisited.tolist())

            components.append(component)

    return components

def get_component_features(components, node_values):
    all_features =  []
    for component in components:
       feature_components = node_values[component, :] 
       energy = node_values[component, NeoGNNDataset.node_feature_dict["raw_energy"]]
       sum_energy = energy.expand(node_values.shape[1], energy.shape[0]) 

       features = torch.sum(node_values[component] * sum_energy.T, axis=0) / torch.sum(energy)
       all_features.append(features)
    return torch.stack(all_features)

def calc_missing_energy(graph_true, graph_pred, node_values, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    true_energy = []
    pred_energy = []
    node_energy = node_values[:, NeoGNNDataset.node_feature_dict["raw_energy"]]

    true_components = find_connected_components(graph_true, node_energy.shape[0], device=device)
    pred_components = find_connected_components(graph_pred, node_energy.shape[0], device=device)

    for component in true_components:
        true_energy.append(torch.sum(node_energy[component]).item())

        filtered_components = ak.from_iter(filter_nested_list(pred_components, component))
        filtered_components = filtered_components[ak.num(filtered_components) > 0]
        energys = []
        for comp in filtered_components:
            energys.append(torch.sum(node_energy[comp]))

        pred_energy.append(torch.max(torch.tensor(energys)).item())

    true_energy = torch.tensor(true_energy)
    pred_energy = torch.tensor(pred_energy)
    return (true_energy - pred_energy), true_energy

def build_ticl_graph(NTrackster, trackster):

    tracksterTilePos = TICLLayerTile()
    tracksterTileNeg = TICLLayerTile()

    for id_t in range(NTrackster):
        if trackster.barycenter_eta[id_t] > 0.0:
            tracksterTilePos.fill(trackster.barycenter_eta[id_t], trackster.barycenter_phi[id_t], id_t)
        elif trackster.barycenter_eta[id_t] < 0.0:
            tracksterTileNeg.fill(trackster.barycenter_eta[id_t], trackster.barycenter_phi[id_t], id_t)

    allNodes = {}
    allNodes["inner"] = []
    allNodes["outer"] = []

    for id_t in range(NTrackster):
        tNode = Node(id_t)
        delta = 0.1

        eta_min = max(abs(trackster.barycenter_eta[id_t]) - delta, TileConstants.minEta)
        eta_max = min(abs(trackster.barycenter_eta[id_t]) + delta, TileConstants.maxEta)

        if trackster.barycenter_eta[id_t] > 0.0:
            search_box = tracksterTilePos.searchBoxEtaPhi(eta_min, eta_max, trackster.barycenter_phi[id_t] - delta, trackster.barycenter_phi[id_t] + delta)
            if search_box[2] > search_box[3]:
                search_box[3] += TileConstants.nPhiBins

            for eta_i in range(search_box[0], search_box[1] + 1):
                for phi_i in range(search_box[2], search_box[3] + 1):
                    phi_mod = phi_i % TileConstants.nPhiBins
                    neighbours = tracksterTilePos[tracksterTilePos.globalBin(eta_i, phi_mod)]
                    for n in neighbours:
                        if trackster.barycenter_z[n] < trackster.barycenter_z[id_t]:
                            tNode.addInnerNeighbour(n)
                        elif trackster.barycenter_z[n] > trackster.barycenter_z[id_t]:
                            tNode.addOuterNeighbour(n)

        elif trackster.barycenter_eta[id_t] < 0.0:
            search_box = tracksterTileNeg.searchBoxEtaPhi(eta_min, eta_max, trackster.barycenter_phi[id_t] - delta, trackster.barycenter_phi[id_t] + delta)
            if search_box[2] > search_box[3]:
                search_box[3] += TileConstants.nPhiBins

            for eta_i in range(search_box[0], search_box[1] + 1):
                for phi_i in range(search_box[2], search_box[3] + 1):
                    phi_mod = phi_i % TileConstants.nPhiBins
                    neighbours = tracksterTileNeg[tracksterTileNeg.globalBin(eta_i, phi_mod)]
                    for n in neighbours:
                        if abs(trackster[n].barycenter_z) < abs(trackster.barycenter_z[id_t]):
                            tNode.addInnerNeighbour(n)
                        elif abs(trackster[n].barycenter_z) > abs(trackster.barycenter_z[id_t]):
                            tNode.addOuterNeighbour(n)

        allNodes["inner"].append(tNode.inner)
        allNodes["outer"].append(tNode.outer)
    return ak.Array(allNodes)

def negative_edge_imbalance(dataset, epsilon=0):
    num_edges = 0
    neg_edges = 0

    for ev in dataset:
        num_edges += len(ev.y)
        neg_edges += (ev.y <= epsilon).sum()

    return neg_edges / num_edges


def print_graph_statistics(dataset, epsilon=0):
    num_events = len(dataset)
    print(f"Number of events in training dataset: {num_events}")

    num_nodes, num_edges, num_neg, num_pos = 0, 0, 0, 0
    max_nodes = -1
    for ev in dataset:
        num_nodes += ev.num_nodes
        num_edges += len(ev.y)
        num_pos += (ev.y > epsilon).sum()
        num_neg += (ev.y <= epsilon).sum()

        if (ev.num_nodes > max_nodes):
            max_nodes = ev.num_nodes

    print(f"Number of nodes: {num_nodes}")
    print(f"Mean Number of nodes: {num_nodes/num_events}")
    print(f"Max Number of nodes: {max_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Mean Number of edges: {num_edges/num_events}")
    print(f"Number of positive edges: {num_pos}")
    print(f"Mean Number of positive edges: {num_pos/num_events}")
    print(f"Number of negative edges: {num_neg}")
    print(f"Mean Number of negative edges: {num_neg/num_events}")
