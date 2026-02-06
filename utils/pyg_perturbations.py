import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops


def remove_edges_pyg(data: Data, ratio: float, seed: int = 42) -> Data:
    g = data.clone()
    ei = g.edge_index
    m = ei.size(1)
    if m == 0:
        return g

    torch.manual_seed(seed)
    k = max(1, int(ratio * m))

    keep = torch.ones(m, dtype=torch.bool)
    drop = torch.randperm(m)[:k]
    keep[drop] = False
    ei = ei[:, keep]

    ei, _ = remove_self_loops(ei)
    ei = to_undirected(ei, num_nodes=g.num_nodes)
    ei, _ = coalesce(ei, None, g.num_nodes, g.num_nodes)
    g.edge_index = ei
    return g


def add_edges_pyg(data: Data, ratio: float, seed: int = 42) -> Data:
    g = data.clone()
    n = g.num_nodes
    if n is None or n < 2:
        return g

    ei = g.edge_index
    m = ei.size(1)
    k = max(1, int(ratio * m))

    torch.manual_seed(seed)
    src = torch.randint(0, n, (k,))
    dst = torch.randint(0, n, (k,))
    new_ei = torch.stack([src, dst], dim=0)

    ei = torch.cat([ei, new_ei], dim=1)

    ei, _ = remove_self_loops(ei)
    ei = to_undirected(ei, num_nodes=n)
    ei, _ = coalesce(ei, None, n, n)
    g.edge_index = ei
    return g


def shuffle_node_features_pyg(data: Data, seed: int = 42) -> Data:
    g = data.clone()
    if g.x is None:
        return g

    torch.manual_seed(seed)
    perm = torch.randperm(g.x.size(0))
    g.x = g.x[perm].clone()
    return g
