import torch


def build_subgraph(graph, root, neighborhood=1):
    neighbors = graph[1][graph[0] == root]

    if (neighborhood == 0):
        return neighbors
    subgraph = np.copy(neighbors)

    for n in neighbors:
        subgraph = np.append(subgraph, self.build_subgraph(graph, n, neighborhood-1))

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
