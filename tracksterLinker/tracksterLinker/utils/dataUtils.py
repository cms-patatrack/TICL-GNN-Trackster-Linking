import cupy as cp
import numpy as np
import awkward as ak

from sklearn.neighbors import KDTree


def calc_LC_density(num_LCs):
    return num_LCs / (2*(3 - 1.5) * (2 * 47))


def calc_trackster_density(NTracksters):
    return NTracksters / (2*(3 - 1.5) * (2 * 47))


def calc_group_score(edges, y, score, shared_energy, raw_energy):
    res = cp.round((1-ak.to_cupy(score)[edges[:, 0]]) * ak.to_cupy(shared_energy)[edges[:, 0]] / ak.to_cupy(raw_energy)[edges[:, 0]] +
            (1-ak.to_cupy(score)[edges[:, 1]]) * ak.to_cupy(shared_energy)[edges[:, 1]] / ak.to_cupy(raw_energy)[edges[:, 1]], 3)/2
    
    y = ak.to_cupy(y)
    res[y[edges[:, 0]] != y[edges[:, 1]]] = 0
    res[y[edges[:, 0]] == -1] = 0
    res[y[edges[:, 1]] == -1] = 0
    return res


# Store best fitting sim trackster with shared_e and score for each trackster
# -1 if no fit less than 0.2
def calc_reco_2_sim_trackster_fit(groups, score, shared_energy):
    idx = score < 0.2
    simTracksters = groups[idx]
    emptys = ak.unflatten(np.full_like(ak.count(groups, axis=-1), -1), 1, axis=-1)

    y = ak.flatten(ak.where(ak.count(simTracksters, axis=-1) == 1, groups[idx], emptys), axis=-1)
    shared_energy = ak.flatten(ak.where(ak.count(simTracksters, axis=-1) == 1, shared_energy[idx], emptys), axis=-1)
    score = ak.flatten(ak.where(ak.count(simTracksters, axis=-1) == 1, score[idx], emptys), axis=-1)

    return y, shared_energy, score


# For each trackster: number of layer cluster, sum of hits, number of layers normalized
def calc_trackster_size(tracksters, clusters):
    num_LCs = ak.count(tracksters.vertices_indexes, axis=2)
    hits = ak.to_list(np.zeros_like(num_LCs))
    length = ak.to_list(np.zeros_like(num_LCs))

    cluster_layer_id = clusters.cluster_layer_id
    vertices_indexes = tracksters.vertices_indexes

    cluster_hits = clusters.cluster_number_of_hits[ak.flatten(vertices_indexes, axis=-1)]
    cluster_layer_ids = cluster_layer_id[ak.flatten(vertices_indexes, axis=-1)]
    vertices_count = ak.count(vertices_indexes, axis=-1)

    for i in range(len(num_LCs)):
        hits[i] = ak.sum(ak.unflatten(cluster_hits[i], vertices_count[i]), axis=-1)
        length[i] = (ak.max(ak.unflatten(cluster_layer_ids[i], vertices_count[i]), axis=-1) -
                     ak.min(ak.unflatten(cluster_layer_ids[i], vertices_count[i]), axis=-1)) / 47

    return num_LCs, hits, length


def calc_spatial_compatibility(edges, features, feature_dict):
    principal_comp_vectors = [feature_dict["eVector0_x"], feature_dict["eVector0_y"], feature_dict["eVector0_z"]]
    return cp.arccos(cp.clip(cp.sum(cp.multiply(features[cp.ix_(edges[:, 1], principal_comp_vectors)], features[cp.ix_(edges[:, 0], principal_comp_vectors)]), axis=1), a_min=-1, a_max=1))


def calc_transverse_plane_separation(edges, features, feature_dict):
    plane = [feature_dict["barycenter_x"], feature_dict["barycenter_y"]]
    return cp.linalg.norm(features[cp.ix_(edges[:, 0], plane)] - features[cp.ix_(edges[:, 0], plane)], axis=1)


def calc_edge_difference(edges, features, feature_dict, key=None):
    if (key is not None):
        return cp.abs(features[edges[:, 1], feature_dict[key]] - features[edges[:, 0], feature_dict[key]])


def calc_min_max_skeleton_dist(nTracksters, edges, vertices):
    edge_indices = cp.zeros((nTracksters, nTracksters, ), dtype='i')

    min_dist = cp.zeros((len(edges[:, 0])), dtype='f')
    max_dist = cp.zeros((len(edges[:, 0])), dtype='f')

    for i in range(len(edges[:, 0])):
        edge_indices[edges[i, 0], edges[i, 1]] = i

    for root in range(nTracksters):
        tree = KDTree(vertices[root], leaf_size=2)
        num = len(vertices[root])
        for target in range(root, nTracksters):
            dist, _ = tree.query(vertices[target], k=num)
            min_dist[edge_indices[root, target]] = cp.min(dist)
            max_dist[edge_indices[root, target]] = cp.max(dist)

            min_dist[edge_indices[target, root]] = min_dist[edge_indices[root, target]]
            max_dist[edge_indices[target, root]] = max_dist[edge_indices[root, target]]
