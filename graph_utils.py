import torch
import numpy as np
import awkward as ak

from typing import List, Dict, Any
from collections import defaultdict


class TileConstants:
    minEta = -np.pi
    maxEta = np.pi
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
        phi_bin = int((phi + np.pi) / (2 * np.pi) * TileConstants.nPhiBins)
        self.tiles[(eta_bin, phi_bin)].append(idx)

    def __getitem__(self, bin_idx):
        return self.tiles.get(bin_idx, [])

    def globalBin(self, eta_idx, phi_idx):
        return (eta_idx, phi_idx)

    def searchBoxEtaPhi(self, eta_min, eta_max, phi_min, phi_max):
        eta_min_bin = int((eta_min - TileConstants.minEta) * 10)
        eta_max_bin = int((eta_max - TileConstants.minEta) * 10)
        phi_min_bin = int((phi_min + np.pi) / (2 * np.pi) * TileConstants.nPhiBins)
        phi_max_bin = int((phi_max + np.pi) / (2 * np.pi) * TileConstants.nPhiBins)
        return [eta_min_bin, eta_max_bin, phi_min_bin, phi_max_bin]


def build_subgraph(graph, root, neighborhood=1):
    neighbors = graph[1][graph[0] == root]

    if (neighborhood == 0):
        return neighbors
    subgraph = np.copy(neighbors)

    for n in neighbors:
        subgraph = np.append(subgraph, build_subgraph(graph, n, neighborhood-1))

    return np.unique(subgraph)


def find_connected_components(graph, num_nodes):
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    components = []

    for node in range(num_nodes):
        if not visited[node]:
            visited[node] = True
            component = [node]
            queue = torch.tensor([node], dtype=torch.long)

            while queue.numel() > 0:
                root = queue[0].item()
                queue = queue[1:]

                # Get neighbors (treat as undirected)
                out_neighbors = graph[1][graph[0] == root]
                in_neighbors = graph[0][graph[1] == root]
                neighbors = torch.cat((out_neighbors, in_neighbors))

                unvisited = neighbors[~visited[neighbors]]
                if unvisited.numel() > 0:
                    unique_unvisited = torch.unique(unvisited)
                    visited[unique_unvisited] = True
                    queue = torch.cat((queue, unique_unvisited))
                    component.extend(unique_unvisited.tolist())

            components.append(component)

    return components


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
