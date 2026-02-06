from torch_geometric.datasets import TUDataset
from .splits import create_loaders


def load_mutag(root: str = "data/TUDataset"):
    dataset = TUDataset(root=root, name="MUTAG")
    return dataset


def inspect_mutag():
    dataset = load_mutag()

    print("=== MUTAG DATASET INFO ===")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of node features: {dataset.num_features}")

    data = dataset[0]
    print("\n=== FIRST GRAPH (data = dataset[0]) ===")
    print(data)

    print("\nNode feature matrix shape (data.x):", getattr(data, "x", None).shape if hasattr(data, "x") else None)
    print("Edge index shape (data.edge_index):", data.edge_index.shape)
    print("Graph label (data.y):", data.y)
    print("Number of nodes:", data.num_nodes)
    print("Number of edges:", data.num_edges)


def demo_mutag_splits():
    dataset = load_mutag()
    loaders, splits = create_loaders(dataset, batch_size=32, seed=42)

    print("=== MUTAG SPLIT SIZES ===")
    for split_name, idx in splits.items():
        print(f"{split_name}: {len(idx)} graphs")

    train_loader = loaders["train"]
    batch = next(iter(train_loader))

    print("\n=== ONE TRAIN BATCH ===")
    print(batch)
    print("Batch properties:")
    print("Batch.x shape:", batch.x.shape if hasattr(batch, "x") else None)
    print("Batch.edge_index shape:", batch.edge_index.shape)
    print("Batch.y shape:", batch.y.shape)
    print("Number of graphs in batch (batch.num_graphs):", batch.num_graphs)


if __name__ == "__main__":
    inspect_mutag()
    print("\n" + "=" * 60 + "\n")
    demo_mutag_splits()
