import numpy as np #provides numerical operations
import networkx as nx #used to create an manipulate graphs
from torch_geometric.utils import from_networkx #converts networkx graph into pytorch format
from sklearn.preprocessing import MinMaxScaler #used for scaling values
import torch #used to create tensors compatible with pytorch
import random #random number generation

def create_graph():
    #creates a directed graph
    graph = nx.DiGraph()


    """
    this section creates 50 nodes and assigns 200-250 random edges between the
    nodes with lengths of 5 to 15 units and speeds from 30 to 60 units per hour
    """
    num_nodes = 50
    for i in range(num_nodes - 1):
        graph.add_edge(i, i + 1, length=random.uniform(5, 15), speed=random.uniform(30, 60))
    for _ in range(200):
        src = random.randint(0, num_nodes - 1)
        dest = random.randint(0, num_nodes - 1)
        if src != dest:
            graph.add_edge(src, dest, length=random.uniform(5, 15), speed=random.uniform(30, 60))


    #iterates through all valid edges and adds an ETA attribute which in this case is the actual ETA
    for _, _, data in graph.edges(data=True):
        data['eta'] = data['length'] / data['speed']

    #creates an array of the degrees of each node
    degrees = np.array([graph.degree(n) for n in graph.nodes], dtype=np.float32)
    #creates an array of the average speeds of all edges connected to each node
    avg_speeds = np.array([
        sum(data['speed'] for _, _, data in graph.edges(n, data=True)) / graph.degree(n) if graph.degree(n) > 0 else 0
        for n in graph.nodes
    ], dtype=np.float32)
    #combines these two arrays into a 2D array of degree and average speed for each node
    node_features = np.stack([degrees, avg_speeds], axis=1)


    """
    Uses the scaler to normalize the values on a 0 to 1 scale and converts to a torch.tensor. 
    A tensor is a multidimensional array that has more capabilities for deep learning and
    high-performance computing that is used by pytorch.
    """
    scaler = MinMaxScaler()
    normalized_node_features = scaler.fit_transform(node_features)
    data_x = torch.tensor(normalized_node_features, dtype=torch.float)


    #pulls actual ETA values into array, normalizes them, then converts to a tensor
    edge_attrs = np.array([
        graph[u][v]['eta'] for u, v in graph.edges
    ], dtype=np.float32)
    edge_scaler = MinMaxScaler()
    normalized_edge_attrs = edge_scaler.fit_transform(edge_attrs.reshape(-1, 1))
    data_edge_attr = torch.tensor(normalized_edge_attrs, dtype=torch.float)


    #converts the networkx graph vraiables into pytorch format
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
    print("\n")

    return data, graph