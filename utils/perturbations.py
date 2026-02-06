import random
from copy import deepcopy
import networkx as nx


def remove_edges_random(G: nx.Graph, ratio: float, seed: int = 42) -> nx.Graph:
    rng = random.Random(seed)
    Gp = deepcopy(G)

    edges = list(Gp.edges())
    m = len(edges)
    if m == 0:
        return Gp

    k = int(ratio * m)
    if k < 1:
        k = 1

    to_remove = rng.sample(edges, k)
    Gp.remove_edges_from(to_remove)
    return Gp


def add_edges_random(G: nx.Graph, ratio: float, seed: int = 42) -> nx.Graph:
    rng = random.Random(seed)
    Gp = deepcopy(G)

    nodes = list(Gp.nodes())
    if len(nodes) < 2:
        return Gp

    m = Gp.number_of_edges()
    k = int(ratio * m)
    if k < 1:
        k = 1

    non_edges = list(nx.non_edges(Gp))
    if len(non_edges) == 0:
        return Gp

    k = min(k, len(non_edges))
    to_add = rng.sample(non_edges, k)
    Gp.add_edges_from(to_add)
    return Gp


def shuffle_node_features(G: nx.Graph, seed: int = 42, feature_key: str = "feature") -> nx.Graph:
    rng = random.Random(seed)
    Gp = deepcopy(G)

    nodes = list(Gp.nodes())
    feats = [Gp.nodes[u].get(feature_key, None) for u in nodes]

    if all(f is None for f in feats):
        return Gp

    rng.shuffle(feats)
    for u, f in zip(nodes, feats):
        if f is None:
            Gp.nodes[u].pop(feature_key, None)
        else:
            Gp.nodes[u][feature_key] = f
    return Gp
