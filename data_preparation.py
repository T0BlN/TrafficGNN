from torch_geometric.utils import from_networkx
import torch

def create_graph():
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_edge(0, 1, length=5, speed=40)
    graph.add_edge(1, 2, length=10, speed=60)
    graph.add_edge(2, 3, length=8, speed=50)
    graph.add_edge(0, 3, length=15, speed=70)

    for _, _, data in graph.edges(data=True):
        data['eta'] = data['length'] / data['speed']

    data = from_networkx(graph)

    num_nodes = graph.number_of_nodes()
    data.x = torch.ones((num_nodes, 1))

    edge_attrs = [data['eta'] for _, _, data in graph.edges(data=True)]
    data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    print(data.edge_attr)

    return data