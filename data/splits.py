
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset


def stratified_split(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, np.ndarray]:

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    num_graphs = len(dataset)
    indices = np.arange(num_graphs)

   
    y = []
    for i in indices:
        y_i = dataset[i].y
        y.append(int(y_i.item()))
    y = np.array(y)

    temp_ratio = val_ratio + test_ratio
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices,
        y,
        test_size=temp_ratio,
        stratify=y,
        random_state=seed,
    )

    relative_test_size = test_ratio / (val_ratio + test_ratio)

    idx_val, idx_test, _, _ = train_test_split(
        idx_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=seed + 1,
    )

    splits = {
        "train": np.array(idx_train),
        "val": np.array(idx_val),
        "test": np.array(idx_test),
    }

    return splits


def create_loaders(
    dataset: Dataset,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[Dict[str, DataLoader], Dict[str, np.ndarray]]:
    splits = stratified_split(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    loaders: Dict[str, DataLoader] = {}

    for split_name, idx in splits.items():
        subset = Subset(dataset, idx)
        shuffle = split_name == "train"
        loaders[split_name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    return loaders, splits
