import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import MinMaxScaler
import torch
import random

def create_graph():
    graph = nx.DiGraph()

    num_nodes = 50
    for i in range(num_nodes - 1):
        graph.add_edge(i, i + 1, length=random.uniform(5, 15), speed=random.uniform(30, 60))
    for _ in range(200):
        src = random.randint(0, num_nodes - 1)
        dest = random.randint(0, num_nodes - 1)
        if src != dest:
            graph.add_edge(src, dest, length=random.uniform(5, 15), speed=random.uniform(30, 60))

    for _, _, data in graph.edges(data=True):
        data['eta'] = data['length'] / data['speed']

    degrees = np.array([graph.degree(n) for n in graph.nodes], dtype=np.float32)
    avg_speeds = np.array([
        sum(data['speed'] for _, _, data in graph.edges(n, data=True)) / graph.degree(n) if graph.degree(n) > 0 else 0
        for n in graph.nodes
    ], dtype=np.float32)
    node_features = np.stack([degrees, avg_speeds], axis=1)
    scaler = MinMaxScaler()
    normalized_node_features = scaler.fit_transform(node_features)
    data_x = torch.tensor(normalized_node_features, dtype=torch.float)

    edge_attrs = np.array([
        graph[u][v]['eta'] for u, v in graph.edges
    ], dtype=np.float32)
    edge_scaler = MinMaxScaler()
    normalized_edge_attrs = edge_scaler.fit_transform(edge_attrs.reshape(-1, 1))
    data_edge_attr = torch.tensor(normalized_edge_attrs, dtype=torch.float)

    data = from_networkx(graph)
    data.x = data_x
    data.edge_attr = data_edge_attr

    #debug prints:
    print("Graph created with:")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Number of edges: {len(edge_attrs)}")
    print("  Edge attribute stats:")
    print("    Mean:", data.edge_attr.mean().item())
    print("    Variance:", data.edge_attr.var().item())
    print("    Min:", data.edge_attr.min().item())
    print("    Max:", data.edge_attr.max().item())

    return data, graph