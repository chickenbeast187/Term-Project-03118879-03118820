import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data.enzymes import load_enzymes
from data.splits import create_loaders
from models.gin import GIN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_multiclass(model, loader) -> Dict[str, float]:
    model.eval()
    all_y = []
    all_logits = []

    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, batch.batch)
        all_logits.append(logits.cpu())
        all_y.append(batch.y.view(-1).cpu())

    import torch as T
    logits = T.cat(all_logits, dim=0)
    y = T.cat(all_y, dim=0)

    y_pred = logits.argmax(dim=1).numpy()
    y_true = y.numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    probs = logits.softmax(dim=1).numpy()
    try:
        auc = roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        auc = float("nan")

    return {"acc": acc, "f1": f1, "auc": auc}


def run_enzymes_gin_experiment(
    hidden_dim: int = 64,
    num_layers: int = 5,
    dropout: float = 0.5,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    batch_size: int = 32,
    max_epochs: int = 200,
    patience: int = 20,
):
    dataset = load_enzymes()
    loaders, splits = create_loaders(
        dataset,
        batch_size=batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    num_features = dataset.num_features
    num_classes = dataset.num_classes

    print(f"Device: {DEVICE}")
    print(f"Num features: {num_features}, num classes: {num_classes}")

    # 2. Model, optimizer, loss
    model = GIN(
        num_node_features=num_features,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_metrics = None
    best_epoch = 0
    epochs_no_improve = 0

    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate_multiclass(model, val_loader)

        val_acc = val_metrics["acc"]

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val acc: {val_acc:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )

        # Early stopping on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_metrics = evaluate_multiclass(model, test_loader)
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    total_train_time = time.time() - t_start

    print("\n=== Best epoch ===")
    print(f"Epoch: {best_epoch}")
    print(f"Best Val acc: {best_val_acc:.4f}")
    print("\n=== Test metrics at best val epoch ===")
    print(
        f"Test Accuracy: {best_test_metrics['acc']:.4f}\n"
        f"Test F1-macro: {best_test_metrics['f1']:.4f}\n"
        f"Test AUC (macro OVR): {best_test_metrics['auc']:.4f}"
    )

    results = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "best_epoch": best_epoch,
        "train_time_sec": total_train_time,
        "best_val_acc": best_val_acc,
        "test_acc": best_test_metrics["acc"],
        "test_f1_macro": best_test_metrics["f1"],
        "test_auc_macro_ovr": best_test_metrics["auc"],
    }
    return results


if __name__ == "__main__":
    run_enzymes_gin_experiment()
