import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import umap


def umap_2d(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    X = StandardScaler().fit_transform(embeddings)

    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=15,
        min_dist=0.1,
    )
    Z = reducer.fit_transform(X)
    return Z


def plot_2d(
    points_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str = None,
    show: bool = False,
):
    fig = plt.figure(figsize=(7, 5))
    scatter = plt.scatter(points_2d[:, 0], points_2d[:, 1], c=labels, s=25)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(scatter, label="Class")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
