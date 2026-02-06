from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, global_add_pool


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


class GIN(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 5,
        dropout: float = 0.5,
        eps_trainable: bool = True,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers should be >= 2"

        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        nn1 = MLP(num_node_features, hidden_dim, hidden_dim)
        self.convs.append(GINConv(nn1, eps=0.0, train_eps=eps_trainable))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            nnk = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.convs.append(GINConv(nnk, eps=0.0, train_eps=eps_trainable))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        hg = global_add_pool(h, batch)

        if return_embedding:
            return hg

        logits = self.classifier(hg)
        return logits
