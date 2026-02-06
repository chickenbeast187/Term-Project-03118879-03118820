from typing import List, Tuple
import numpy as np
import networkx as nx
from karateclub import Graph2Vec
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_networkx


def pyg_to_networkx_graph(data: Data, use_node_attr: bool = True) -> nx.Graph:
    G = to_networkx(data, to_undirected=True)

    if use_node_attr and hasattr(data, "x") and data.x is not None:
        x = data.x.detach().cpu().numpy()
        for node in G.nodes():
            G.nodes[node]["feature"] = x[node]
    return G


def graph2vec_embeddings_for_dataset(
    dataset,
    use_node_attr: bool = True,
    dimensions: int = 128,
    wl_iterations: int = 2,
    workers: int = 4,
    epochs: int = 10,
) -> np.ndarray:
    graphs: List[nx.Graph] = []

    for i in range(len(dataset)):
        item = dataset[i]
        if isinstance(item, nx.Graph):
            graphs.append(item)
        else:
            G = pyg_to_networkx_graph(item, use_node_attr=use_node_attr)
            graphs.append(G)

    model = Graph2Vec(
        dimensions=dimensions,
        wl_iterations=wl_iterations,
        workers=workers,
        epochs=epochs,
        attributed=use_node_attr, 
    )

    model.fit(graphs)
    embeddings = model.get_embedding() 
    return embeddings


def labels_for_dataset(dataset: Dataset) -> np.ndarray:
    y_list = []
    for i in range(len(dataset)):
        y_i = dataset[i].y
        y_list.append(int(y_i.item()))
    return np.array(y_list, dtype=int)
