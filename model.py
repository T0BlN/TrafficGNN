import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class TrafficGNN(torch.nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels):
        super(TrafficGNN, self).__init__()
        self.conv1 = GCNConv(node_in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.edge_fc = torch.nn.Linear(hidden_channels * 2 + edge_in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        src, dest = edge_index
        edge_features = torch.cat([x[src], x[dest], edge_attr], dim=1)

        return self.edge_fc(edge_features)