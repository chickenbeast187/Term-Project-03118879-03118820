import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.gin import GIN
from utils.similarity import cosine_similarity
from utils.pyg_perturbations import remove_edges_pyg, add_edges_pyg, shuffle_node_features_pyg

from data.mutag import load_mutag
from data.enzymes import load_enzymes
from data.imdb_multi import load_imdb_multi

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset(name: str):
    name = name.upper()
    if name == "MUTAG":
        return load_mutag()
    if name == "ENZYMES":
        return load_enzymes()
    if name in ("IMDB-MULTI", "IMDB_MULTI", "IMDB"):
        return load_imdb_multi()
    raise ValueError(name)


def ensure_features(dataset):
    if dataset[0].x is not None:
        return dataset

    from torch_geometric.utils import degree
    data_list = []
    for i in range(len(dataset)):
        data = dataset[i].clone()
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes).view(-1, 1)
        mx = float(deg.max()) if deg.numel() > 0 else 1.0
        if mx == 0:
            mx = 1.0
        data.x = deg / mx
        data_list.append(data)
    
    class DatasetWrapper:
        def __init__(self, data_list, original_dataset):
            self._data_list = data_list
            self.num_features = data_list[0].x.size(1) if data_list[0].x is not None else 0
            self.num_classes = original_dataset.num_classes
        
        def __len__(self):
            return len(self._data_list)
        
        def __getitem__(self, idx):
            return self._data_list[idx]
    
    return DatasetWrapper(data_list, dataset)


def stratified_split_indices(y, seed=42):
    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    y_temp = y[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=y_temp, random_state=seed)
    return train_idx, val_idx, test_idx


def train_gin(dataset, train_idx, val_idx, hidden_dim=64, num_layers=5, dropout=0.5,
              epochs=200, lr=1e-3, batch_size=32, patience=20):
    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=batch_size, shuffle=False)

    model = GIN(
        num_node_features=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    opt = Adam(model.parameters(), lr=lr)
    best_val = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y.view(-1))
            loss.backward()
            opt.step()

        val_acc = eval_accuracy(model, val_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_accuracy(model, loader):
    model.eval()
    ys, preds = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1).cpu().numpy()
        y = batch.y.view(-1).cpu().numpy()
        ys.append(y)
        preds.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return accuracy_score(y_true, y_pred)


@torch.no_grad()
def get_embeddings(model, data_list, batch_size=64):
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    model.eval()
    embs = []
    for batch in loader:
        batch = batch.to(DEVICE)
        hg = model(batch.x, batch.edge_index, batch.batch, return_embedding=True)
        embs.append(hg.cpu().numpy())
    return np.vstack(embs)


def perturb_list(data_list, pt: str, ratio: float, seed: int):
    out = []
    for i, d in enumerate(data_list):
        s = seed + i
        if pt == "remove":
            out.append(remove_edges_pyg(d, ratio=ratio, seed=s))
        elif pt == "add":
            out.append(add_edges_pyg(d, ratio=ratio, seed=s))
        elif pt == "shuffle_attr":
            out.append(shuffle_node_features_pyg(d, seed=s))
        else:
            raise ValueError(pt)
    return out


def run_dataset(name: str):
    dataset = ensure_features(get_dataset(name))
    y = np.array([int(dataset[i].y.item()) for i in range(len(dataset))], dtype=int)

    train_idx, val_idx, test_idx = stratified_split_indices(y, seed=42)

    test_set = [dataset[i] for i in test_idx]

    print(f"\n=== GIN stability: {name.upper()} ===")
    print(f"Graphs={len(dataset)} | Classes={dataset.num_classes} | Features={dataset.num_features} | Device={DEVICE}")

    model = train_gin(dataset, train_idx, val_idx)

    base_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    base_acc = eval_accuracy(model, base_loader)
    base_emb = get_embeddings(model, test_set)
    print(f"Baseline test accuracy: {base_acc:.4f}")

    perturb_types = ["remove", "add", "shuffle_attr"]
    ratios = [0.10, 0.20]

    for pt in perturb_types:
        for r in ratios:
            test_pert = perturb_list(test_set, pt, r, seed=100)
            pert_loader = DataLoader(test_pert, batch_size=64, shuffle=False)

            acc = eval_accuracy(model, pert_loader)

            emb_pert = get_embeddings(model, test_pert)
            sims = [cosine_similarity(base_emb[i], emb_pert[i]) for i in range(len(test_set))]
            mean_sim = float(np.mean(sims))
            std_sim = float(np.std(sims))

            print(f"{pt:11s} r={r:.2f} | emb_cos={mean_sim:.4f} ± {std_sim:.4f} | "
                  f"acc={acc:.4f} | Δacc={base_acc-acc:.4f}")


if __name__ == "__main__":
    for d in ["MUTAG", "ENZYMES", "IMDB-MULTI"]:
        run_dataset(d)
