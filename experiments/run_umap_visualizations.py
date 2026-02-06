import os
from pathlib import Path

from data.mutag import load_mutag
from data.enzymes import load_enzymes
from data.imdb_multi import load_imdb_multi

from embeddings.graph2vec import graph2vec_embeddings_for_dataset, labels_for_dataset
from embeddings.netlsd import netlsd_embeddings_for_dataset

from experiments.visualization_utils import umap_2d, plot_2d


def get_dataset(name: str):
    name = name.upper()
    if name == "MUTAG":
        return load_mutag()
    if name == "ENZYMES":
        return load_enzymes()
    if name in ("IMDB-MULTI", "IMDB_MULTI", "IMDB"):
        return load_imdb_multi()
    raise ValueError(f"Unknown dataset: {name}")


def run_umap_for(method: str, dataset_name: str, out_dir: str = "outputs/umap"):
    dataset = get_dataset(dataset_name)
    y = labels_for_dataset(dataset)

    method = method.upper()
    dataset_name_upper = dataset_name.upper()

    print(f"\n=== UMAP for {method} on {dataset_name_upper} ===")

    if method == "GRAPH2VEC":
        use_node_attr = dataset.num_features > 0
        X = graph2vec_embeddings_for_dataset(
            dataset,
            use_node_attr=use_node_attr,
            dimensions=128,
            wl_iterations=2,
            workers=4,
            epochs=10,
        )
        title = f"UMAP: Graph2Vec ({dataset_name_upper})"
        filename = f"umap_graph2vec_{dataset_name_upper}.png"

    elif method == "NETLSD":
        X = netlsd_embeddings_for_dataset(
            dataset,
            scale_steps=250,
            scale_min=-2.0,
            scale_max=2.0,
            approximations=200,
        )
        title = f"UMAP: NetLSD ({dataset_name_upper})"
        filename = f"umap_netlsd_{dataset_name_upper}.png"

    else:
        raise ValueError(f"Unknown method: {method}")

    Z = umap_2d(X, seed=42)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(out_dir, filename)

    plot_2d(Z, y, title=title, save_path=save_path, show=False)



if __name__ == "__main__":
    for d in ["MUTAG", "ENZYMES", "IMDB-MULTI"]:
        run_umap_for("Graph2Vec", d)
        run_umap_for("NetLSD", d)
