import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC

from data.enzymes import load_enzymes
from data.splits import stratified_split
from embeddings.graph2vec import (
    graph2vec_embeddings_for_dataset,
    labels_for_dataset,
)


def run_enzymes_graph2vec_experiment(
    dimensions: int = 128,
    wl_iterations: int = 2,
    use_node_attr: bool = True,
):
    dataset = load_enzymes()

    print("Computing Graph2Vec embeddings for ENZYMES...")
    t0 = time.time()
    X = graph2vec_embeddings_for_dataset(
        dataset,
        use_node_attr=use_node_attr,
        dimensions=dimensions,
        wl_iterations=wl_iterations,
        workers=4,
        epochs=10,
    )
    embed_time = time.time() - t0

    print(f"Embeddings shape: {X.shape}")
    print(f"Embedding time: {embed_time:.3f} seconds")

    y = labels_for_dataset(dataset)

    splits = stratified_split(dataset, 0.8, 0.1, 0.1, seed=42)

    idx_train = splits["train"]
    idx_val = splits["val"]
    idx_test = splits["test"]

    print("\nSplit sizes:")
    print(len(idx_train), len(idx_val), len(idx_test))

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    print("\nTraining SVM (RBF)...")
    t1 = time.time()
    clf = SVC(kernel="rbf", probability=True)
    clf.fit(X_train, y_train)
    train_time = time.time() - t1
    print(f"Train time: {train_time:.3f} seconds")

    def eval_split(name, Xs, ys):
        y_pred = clf.predict(Xs)

        acc = accuracy_score(ys, y_pred)
        f1 = f1_score(ys, y_pred, average="macro")

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 (macro): {f1:.4f}")

    eval_split("Validation", X_val, y_val)
    eval_split("Test", X_test, y_test)


if __name__ == "__main__":
    run_enzymes_graph2vec_experiment()
