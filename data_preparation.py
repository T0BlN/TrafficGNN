from torch_geometric.utils import from_networkx
import torch
import random
import networkx as nx

def create_graph():
    import networkx as nx

    graph = nx.DiGraph()

    num_nodes = 20
    for i in range(num_nodes - 1):
        graph.add_edge(i, i + 1, length=random.uniform(5, 15), speed=random.uniform(30, 60))

    for _ in range(30):
        src = random.randint(0, num_nodes - 1)
        dest = random.randint(0, num_nodes - 1)
        if src != dest:
            graph.add_edge(src, dest, length=random.uniform(5, 15), speed=random.uniform(30, 60))

    for _, _, data in graph.edges(data=True):
        data['eta'] = data['length'] / data['speed']

    data = from_networkx(graph)

    degrees = [graph.degree(n) for n in graph.nodes]
    avg_speeds = [sum(data['speed'] for _, _, data in graph.edges(n, data=True)) / graph.degree(n) for n in graph.nodes]
    data.x = torch.tensor([[degrees[n], avg_speeds[n]] for n in graph.nodes], dtype=torch.float)

    edge_attrs = []
    for i in range(data.edge_index.size(1)):
        src, dest = data.edge_index[:, i]
        edge_data = graph.get_edge_data(src.item(), dest.item())
        edge_attrs.append(edge_data['eta'])
    data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)

    print("Aligned edge attributes (data.edge_attr.size):", data.edge_attr.size())
    print("Edge index shape after alignment:", data.edge_index.size())
    print("Graph created with:")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Number of edges: {len(edge_attrs)}")
    print("  Sample edge attributes (ETA):", edge_attrs[:5])

    return data, graph
