import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from data.mutag import load_mutag
from data.enzymes import load_enzymes
from data.imdb_multi import load_imdb_multi
from models.gin import GIN


def run_gin_clustering(dataset_name):
    if dataset_name == "MUTAG":
        dataset = load_mutag()
        num_classes = 2
    elif dataset_name == "ENZYMES":
        dataset = load_enzymes()
        num_classes = 6
    elif dataset_name == "IMDB-MULTI":
        dataset = load_imdb_multi()
        num_classes = 3
    else:
        raise ValueError("Unknown dataset")

    graphs = list(dataset)
    labels = [g.y.item() for g in graphs]

    for g in graphs:
        if g.x is None:
            g.x = torch.ones((g.num_nodes, 1), dtype=torch.float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GIN(
        num_node_features=graphs[0].x.shape[1],
        num_classes=num_classes
    ).to(device)

    model.eval()

    embeddings = []

    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            x = g.x
            edge_index = g.edge_index
            batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)
            
            for conv, bn in zip(model.convs, model.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
            
            from torch_geometric.nn import global_add_pool
            graph_emb = global_add_pool(x, batch)
            embeddings.append(graph_emb.cpu().numpy())

    X = np.vstack(embeddings)

    kmeans = KMeans(n_clusters=num_classes, random_state=42)
    clusters = kmeans.fit_predict(X)

    ari = adjusted_rand_score(labels, clusters)

    print(f"=== GIN + k-means on {dataset_name} ===")
    print(f"Graphs: {len(graphs)}, Classes: {num_classes}")
    print(f"Embedding dim: {X.shape[1]}")
    print(f"ARI: {ari:.4f}")


if __name__ == "__main__":
    for dataset in ["MUTAG", "ENZYMES", "IMDB-MULTI"]:
        run_gin_clustering(dataset)
