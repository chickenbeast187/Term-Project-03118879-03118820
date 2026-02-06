import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from utils.similarity import cosine_similarity
from utils.perturbations import remove_edges_random, add_edges_random, shuffle_node_features

from data.mutag import load_mutag
from data.enzymes import load_enzymes
from data.imdb_multi import load_imdb_multi

from embeddings.graph2vec import pyg_to_networkx_graph, graph2vec_embeddings_for_dataset
from embeddings.netlsd import SafeNetLSD


def get_dataset(name: str):
    name = name.upper()
    if name == "MUTAG":
        return load_mutag()
    if name == "ENZYMES":
        return load_enzymes()
    if name in ("IMDB-MULTI", "IMDB_MULTI", "IMDB"):
        return load_imdb_multi()
    raise ValueError(f"Unknown dataset {name}")


def split_indices(y, seed=42):
    n = len(y)
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    return idx[:int(0.8*n)], idx[int(0.8*n):]


def netlsd_embeddings_from_nx(graphs, seed=42, scale_steps=250):
    model = SafeNetLSD(scale_steps=scale_steps, seed=seed)
    model.fit(graphs)
    return model.get_embedding()


def perturb_graph(G, perturb_type: str, ratio: float, seed: int):
    pt = perturb_type.lower()
    if pt == "remove":
        return remove_edges_random(G, ratio, seed=seed)
    if pt == "add":
        return add_edges_random(G, ratio, seed=seed)
    if pt == "shuffle_attr":
        return shuffle_node_features(G, seed=seed, feature_key="feature")
    raise ValueError(f"Unknown perturb_type: {perturb_type}")


def compute_embeddings(method: str, graphs, use_node_attr: bool, seed: int = 42):
    method = method.upper()
    if method == "GRAPH2VEC":
        return graph2vec_embeddings_for_dataset(graphs, use_node_attr=use_node_attr, dimensions=128)
    elif method == "NETLSD":
        return netlsd_embeddings_from_nx(graphs, seed=seed, scale_steps=250)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_stability_analysis(dataset_name: str, method: str, perturb_type: str, ratio: float, seed: int = 42):
    dataset = get_dataset(dataset_name)
    y = np.array([int(dataset[i].y.item()) for i in range(len(dataset))])
    train_idx, test_idx = split_indices(y, seed=seed)

    use_node_attr = dataset.num_features > 0

    graphs_orig = [
        pyg_to_networkx_graph(dataset[i], use_node_attr=use_node_attr)
        for i in range(len(dataset))
    ]

    graphs_pert = [
        perturb_graph(G, perturb_type, ratio, seed=seed + i)
        for i, G in enumerate(graphs_orig)
    ]

    all_graphs = graphs_orig + graphs_pert
    X_all = compute_embeddings(method, all_graphs, use_node_attr, seed=seed)
    
    n_graphs = len(graphs_orig)
    X_orig = X_all[:n_graphs]
    X_pert = X_all[n_graphs:]

    sims = [cosine_similarity(X_orig[i], X_pert[i]) for i in range(n_graphs)]
    mean_sim = float(np.mean(sims))
    std_sim = float(np.std(sims))

    clf = SVC(kernel="rbf")
    clf.fit(X_orig[train_idx], y[train_idx])
    
    acc_orig = accuracy_score(y[test_idx], clf.predict(X_orig[test_idx]))
    acc_pert = accuracy_score(y[test_idx], clf.predict(X_pert[test_idx]))
    acc_drop = acc_orig - acc_pert

    return {
        "cos_mean": mean_sim,
        "cos_std": std_sim,
        "acc_orig": acc_orig,
        "acc_pert": acc_pert,
        "acc_drop": acc_drop,
    }


if __name__ == "__main__":
    datasets = ["MUTAG", "ENZYMES", "IMDB-MULTI"]
    methods = ["GRAPH2VEC", "NETLSD"]
    perturb_types = ["remove", "add", "shuffle_attr"]
    ratios = [0.10, 0.20]

    print(f"{'Dataset':<12} | {'Method':<10} | {'Perturb':<12} | {'Ratio':<5} | "
          f"{'Cos Sim':<14} | {'Acc Orig':<8} | {'Acc Pert':<8} | {'Δ Acc':<8}")
    print("-" * 95)

    for d in datasets:
        for m in methods:
            for pt in perturb_types:
                for r in ratios:
                    res = run_stability_analysis(d, m, pt, r, seed=42)
                    print(f"{d:<12} | {m:<10} | {pt:<12} | {r:<5.2f} | "
                          f"{res['cos_mean']:.4f} ± {res['cos_std']:.4f} | "
                          f"{res['acc_orig']:.4f}   | {res['acc_pert']:.4f}   | {res['acc_drop']:+.4f}")
