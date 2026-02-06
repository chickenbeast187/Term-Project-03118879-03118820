from torch_geometric.data import Dataset, Data
from torch_geometric.utils import degree
from typing import List


def add_degree_as_feature(dataset: Dataset) -> List[Data]:
    data_list = []
    for i in range(len(dataset)):
        data = dataset[i].clone()
        if data.x is None:
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes).view(-1, 1).float()
            max_deg = deg.max().item() if deg.numel() > 0 else 1.0
            if max_deg == 0:
                max_deg = 1.0
            data.x = deg / max_deg
        data_list.append(data)
    return data_list
