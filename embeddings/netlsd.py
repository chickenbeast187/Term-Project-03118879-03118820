from typing import List
from karateclub.graph_embedding.netlsd import NetLSD as KC_NetLSD
import numpy as np
import scipy.sparse as sps
import networkx as nx
from torch_geometric.data import Dataset, Data
from embeddings.graph2vec import pyg_to_networkx_graph
from embeddings.graph2vec import labels_for_dataset


class SafeNetLSD(KC_NetLSD):
    def _calculate_eigenvalues(self, laplacian_matrix):
        number_of_nodes = laplacian_matrix.shape[0]
        if number_of_nodes <= 2:
            eigenvalues = np.linalg.eigvalsh(laplacian_matrix.toarray())
            return eigenvalues

        
        if 2 * self.approximations < number_of_nodes:
            lower_eigenvalues = sps.linalg.eigsh(
                laplacian_matrix,
                self.approximations,
                which="SM",
                ncv=5 * self.approximations,
                return_eigenvectors=False,
            )[::-1]
            upper_eigenvalues = sps.linalg.eigsh(
                laplacian_matrix,
                self.approximations,
                which="LM",
                ncv=5 * self.approximations,
                return_eigenvectors=False,
            )
            eigenvalues = self._updown_linear_approx(
                lower_eigenvalues, upper_eigenvalues, number_of_nodes
            )
        else:
            k = max(1, number_of_nodes - 2)
            eigenvalues = sps.linalg.eigsh(
                laplacian_matrix,
                k,
                which="LM",
                return_eigenvectors=False,
            )
        return eigenvalues


def netlsd_embeddings_for_dataset(
    dataset: Dataset,
    scale_min: float = -2.0,
    scale_max: float = 2.0,
    scale_steps: int = 250,
    approximations: int = 200,
) -> np.ndarray:
    graphs: List[nx.Graph] = []

    for i in range(len(dataset)):
        data: Data = dataset[i]
        G = pyg_to_networkx_graph(data, use_node_attr=False)
        graphs.append(G)

    model = SafeNetLSD(
        scale_min=scale_min,
        scale_max=scale_max,
        scale_steps=scale_steps,
        approximations=approximations,
        seed=42,
    )

    model.fit(graphs)
    embeddings = model.get_embedding()
    return embeddings
