import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np

from experiments.clustering_utils import kmeans_clustering

from data.mutag import load_mutag
from data.enzymes import load_enzymes
from data.imdb_multi import load_imdb_multi

from embeddings.graph2vec import graph2vec_embeddings_for_dataset, labels_for_dataset
from embeddings.netlsd import netlsd_embeddings_for_dataset


def get_dataset(name: str):
    name = name.upper()
    if name == "MUTAG":
        return load_mutag()
    if name == "ENZYMES":
        return load_enzymes()
    if name in ("IMDB-MULTI", "IMDB_MULTI", "IMDB"):
        return load_imdb_multi()
    raise ValueError(f"Unknown dataset: {name}")


def compute_embeddings(method: str, dataset, dim: int):
    method = method.upper()

    if method == "GRAPH2VEC":
        use_node_attr = dataset.num_features > 0
        X = graph2vec_embeddings_for_dataset(
            dataset,
            use_node_attr=use_node_attr,
            dimensions=dim,
            wl_iterations=2,
            workers=4,
            epochs=10,
        )
        return X

    if method == "NETLSD":
        X = netlsd_embeddings_for_dataset(
            dataset,
            scale_steps=dim,
            scale_min=-2.0,
            scale_max=2.0,
            approximations=200,
        )
        return X

    raise ValueError(f"Unknown method: {method}")


def run_kmeans(dataset_name: str, method: str, dim: int = 128, seed: int = 42):
    dataset = get_dataset(dataset_name)
    y = labels_for_dataset(dataset)

    print(f"\n=== {method.upper()} + k-means on {dataset_name.upper()} ===")
    print(f"Graphs: {len(dataset)}, Classes: {dataset.num_classes}, Dim: {dim}")

    t0 = time.time()
    X = compute_embeddings(method, dataset, dim)
    embed_time = time.time() - t0

    ari, _ = kmeans_clustering(X, y, n_clusters=dataset.num_classes, seed=seed)

    print(f"Embedding time: {embed_time:.3f}s")
    print(f"ARI: {ari:.4f}")

    return {"dataset": dataset_name, "method": method, "dim": dim, "ari": ari, "embed_time_sec": embed_time}


if __name__ == "__main__":
    # Run a default batch (you can change dimensions later)
    for dataset_name in ["MUTAG", "ENZYMES", "IMDB-MULTI"]:
        run_kmeans(dataset_name, "Graph2Vec", dim=128)
        run_kmeans(dataset_name, "NetLSD", dim=250)  # NetLSD default dimension
