import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

try:
    import umap
except ImportError as e:
    raise ImportError(
        "Missing dependency: umap-learn. Install with: pip install umap-learn"
    ) from e

from models.gin import GIN

from data.mutag import load_mutag
from data.enzymes import load_enzymes
from data.imdb_multi import load_imdb_multi


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataset(name: str):
    name = name.upper()
    if name == "MUTAG":
        dataset = load_mutag()
        num_classes = 2
    elif name == "ENZYMES":
        dataset = load_enzymes()
        num_classes = 6
    elif name in ["IMDB-MULTI", "IMDB_MULTI", "IMDBMULTI"]:
        dataset = load_imdb_multi()
        num_classes = 3
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Support both: (graphs, labels) OR a PyG dataset with .y
    if isinstance(dataset, tuple) and len(dataset) == 2:
        graphs, labels = dataset
        return graphs, np.array(labels), num_classes

    graphs = list(dataset)
    labels = np.array([int(g.y.item()) for g in graphs])
    return graphs, labels, num_classes


def train_gin_and_get_best_model(graphs, labels, num_features, num_classes, device,
                                 batch_size=32, lr=1e-3, weight_decay=5e-4,
                                 max_epochs=200, patience=20):

    idx = np.arange(len(graphs))
    train_idx, tmp_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, random_state=42, stratify=labels[tmp_idx])

    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=batch_size, shuffle=False)

    model = GIN(num_node_features=num_features, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct += int((pred == batch.y).sum().item())
                total += batch.y.size(0)

        val_acc = correct / max(total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    return model


def extract_graph_embeddings(model, graphs, device):
    from torch_geometric.nn import global_add_pool
    
    model.eval()
    loader = DataLoader(graphs, batch_size=64, shuffle=False)
    embs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x = batch.x
            edge_index = batch.edge_index
            
            for conv, bn in zip(model.convs, model.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
            
            graph_emb = global_add_pool(x, batch.batch)
            embs.append(graph_emb.detach().cpu().numpy())

    X = np.vstack(embs)
    return X


def umap_plot(X, y, title, out_path):
    reducer = umap.UMAP(n_components=2, random_state=42)
    Z = reducer.fit_transform(X)

    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=y, s=10)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs/umap", exist_ok=True)

    for ds_name in ["MUTAG", "ENZYMES", "IMDB-MULTI"]:
        graphs, labels, num_classes = get_dataset(ds_name)

        for g in graphs:
            if g.x is None:
                g.x = torch.ones((g.num_nodes, 1), dtype=torch.float)
        
        num_features = graphs[0].x.shape[1]

        print(f"\n=== GIN UMAP on {ds_name} ===")
        print(f"Graphs: {len(graphs)}, Classes: {num_classes}, Num features: {num_features}")

        model = train_gin_and_get_best_model(
            graphs, labels, num_features=num_features, num_classes=num_classes, device=device
        )

        X = extract_graph_embeddings(model, graphs, device)

        out_path = f"outputs/umap/umap_gin_{ds_name.replace('-', '_')}.png"
        umap_plot(X, labels, f"UMAP of GIN embeddings on {ds_name}", out_path)


if __name__ == "__main__":
    main()
