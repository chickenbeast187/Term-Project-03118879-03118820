from torch_geometric.datasets import TUDataset
from .splits import create_loaders


def load_imdb_multi(root: str = "data/TUDataset"):
    dataset = TUDataset(root=root, name="IMDB-MULTI")
    return dataset


def inspect_imdb_multi():
    dataset = load_imdb_multi()

    print("=== IMDB-MULTI DATASET INFO ===")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of node features: {dataset.num_features}")

    data = dataset[0]
    print("\n=== FIRST GRAPH (data = dataset[0]) ===")
    print(data)

    x_attr = getattr(data, "x", None) if hasattr(data, "x") else None
    print("\nNode feature matrix (data.x):", x_attr)
    print("Edge index shape (data.edge_index):", data.edge_index.shape)
    print("Graph label (data.y):", data.y)
    print("Number of nodes:", data.num_nodes)
    print("Number of edges:", data.num_edges)


def demo_imdb_multi_splits():
    dataset = load_imdb_multi()
    loaders, splits = create_loaders(dataset, batch_size=32, seed=42)

    print("=== IMDB-MULTI SPLIT SIZES ===")
    for split_name, idx in splits.items():
        print(f"{split_name}: {len(idx)} graphs")

    train_loader = loaders["train"]
    batch = next(iter(train_loader))

    print("\n=== ONE TRAIN BATCH (IMDB-MULTI) ===")
    print(batch)
    print("Batch properties:")
    x_attr = batch.x if hasattr(batch, "x") else None
    print("Batch.x:", x_attr)
    print("Batch.edge_index shape:", batch.edge_index.shape)
    print("Batch.y shape:", batch.y.shape)
    print("Number of graphs in batch (batch.num_graphs):", batch.num_graphs)


if __name__ == "__main__":
    inspect_imdb_multi()
    print("\n" + "=" * 60 + "\n")
    demo_imdb_multi_splits()
