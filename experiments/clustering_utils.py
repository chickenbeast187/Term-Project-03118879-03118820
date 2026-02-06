import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def kmeans_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    seed: int = 42,
):
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=20,
    )
    clusters = kmeans.fit_predict(embeddings)

    ari = adjusted_rand_score(labels, clusters)
    return ari, clusters
