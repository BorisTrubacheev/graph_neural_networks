import torch
from torch import Tensor, nn


class MyGCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = x.size(0)
        source, target = edge_index

        not_self_loop = source != target
        source = source[not_self_loop]
        target = target[not_self_loop]
        nodes = torch.arange(num_nodes, device=x.device)
        source = torch.cat((source, nodes))
        target = torch.cat((target, nodes))

        degree = x.new_zeros(num_nodes)
        degree.index_add_(0, target, x.new_ones(target.numel()))
        inv_sqrt_degree = degree.clamp_min(1).pow(-0.5)
        norm = inv_sqrt_degree[source] * inv_sqrt_degree[target]

        transformed = self.lin(x)
        messages = transformed[source] * norm.unsqueeze(-1)
        output = transformed.new_zeros((num_nodes, transformed.size(-1)))
        output.index_add_(0, target, messages)
        return output


class MySAGEConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = x.size(0)
        source, target = edge_index

        neighbor_sum = x.new_zeros(x.shape)
        neighbor_sum.index_add_(0, target, x[source])
        degree = x.new_zeros(num_nodes)
        degree.index_add_(0, target, x.new_ones(target.numel()))
        neighbor_mean = neighbor_sum / degree.clamp_min(1).unsqueeze(-1)

        return self.lin_neigh(neighbor_mean) + self.lin_self(x)
