import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC

from data.mutag import load_mutag
from data.splits import stratified_split
from embeddings.graph2vec import (
    graph2vec_embeddings_for_dataset,
    labels_for_dataset,
)


def run_mutag_graph2vec_experiment(
    dimensions: int = 128,
    wl_iterations: int = 2,
    use_node_attr: bool = True,
):
    dataset = load_mutag()

    print("Computing Graph2Vec embeddings for MUTAG...")
    t_start_embed = time.time()
    X = graph2vec_embeddings_for_dataset(
        dataset,
        use_node_attr=use_node_attr,
        dimensions=dimensions,
        wl_iterations=wl_iterations,
        workers=4,
        epochs=10,
    )
    t_end_embed = time.time()
    embed_time = t_end_embed - t_start_embed

    print(f"Embeddings shape: {X.shape}")
    print(f"Embedding generation time: {embed_time:.3f} seconds")

    y = labels_for_dataset(dataset)

    splits = stratified_split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    idx_train = splits["train"]
    idx_val = splits["val"]
    idx_test = splits["test"]

    print("\nSplit sizes:")
    print(f"Train: {len(idx_train)}")
    print(f"Val:   {len(idx_val)}")
    print(f"Test:  {len(idx_test)}")

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    print("\nTraining SVM classifier on Graph2Vec embeddings...")
    t_start_clf = time.time()
    clf = SVC(kernel="rbf", probability=True, random_state=42)
    clf.fit(X_train, y_train)
    t_end_clf = time.time()
    train_time = t_end_clf - t_start_clf
    print(f"Classifier training time: {train_time:.3f} seconds")

    def evaluate_split(name: str, X_split: np.ndarray, y_split: np.ndarray):
        y_pred = clf.predict(X_split)

        acc = accuracy_score(y_split, y_pred)
        f1 = f1_score(y_split, y_pred, average="binary")

        y_proba = clf.predict_proba(X_split)[:, 1]
        try:
            auc = roc_auc_score(y_split, y_proba)
        except ValueError:
            auc = float("nan")

        print(f"\n=== {name} performance ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUC:      {auc:.4f}")

    evaluate_split("Validation", X_val, y_val)
    evaluate_split("Test", X_test, y_test)

    results = {
        "dimensions": dimensions,
        "wl_iterations": wl_iterations,
        "use_node_attr": use_node_attr,
        "embed_time_sec": embed_time,
        "train_time_sec": train_time,
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
    }
    return results


if __name__ == "__main__":
    run_mutag_graph2vec_experiment(dimensions=128, wl_iterations=2, use_node_attr=True)
