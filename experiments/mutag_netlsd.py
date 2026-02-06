import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC

from data.mutag import load_mutag
from data.splits import stratified_split
from embeddings.netlsd import netlsd_embeddings_for_dataset
from embeddings.graph2vec import labels_for_dataset


def run_mutag_netlsd_experiment(
    scale_steps: int = 250,
    scale_min: float = -2.0,
    scale_max: float = 2.0,
    approximations: int = 200,
):
    dataset = load_mutag()

    print("Computing NetLSD embeddings for MUTAG...")
    t_start_embed = time.time()
    X = netlsd_embeddings_for_dataset(
        dataset,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_steps=scale_steps,
        approximations=approximations,
    )
    embed_time = time.time() - t_start_embed

    print(f"Embeddings shape: {X.shape}")
    print(f"Embedding generation time: {embed_time:.3f} seconds")

    y = labels_for_dataset(dataset)

    splits = stratified_split(dataset, 0.8, 0.1, 0.1, seed=42)
    idx_train, idx_val, idx_test = splits["train"], splits["val"], splits["test"]

    print("\nSplit sizes:")
    print(f"Train: {len(idx_train)}")
    print(f"Val:   {len(idx_val)}")
    print(f"Test:  {len(idx_test)}")

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    print("\nTraining SVM classifier on NetLSD embeddings...")
    t_start_clf = time.time()
    clf = SVC(kernel="rbf", probability=True, random_state=42)
    clf.fit(X_train, y_train)
    train_time = time.time() - t_start_clf
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
        "scale_steps": scale_steps,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "approximations": approximations,
        "embed_time_sec": embed_time,
        "train_time_sec": train_time,
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
    }
    return results


if __name__ == "__main__":
    run_mutag_netlsd_experiment()
