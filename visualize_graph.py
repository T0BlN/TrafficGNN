import matplotlib.pyplot as plt
import networkx as nx

#code to display the node graph. Rounds decimals for veiwing purposes

def visualize_graph(graph, round_decimals=2):
    pos = nx.spring_layout(graph)

    plt.figure(figsize=(10, 10))
    nx.draw(
        graph, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=500, font_size=10
    )
    
    edge_labels = nx.get_edge_attributes(graph, "eta")
    rounded_edge_labels = {key: round(value, round_decimals) for key, value in edge_labels.items()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=rounded_edge_labels, font_size=8)

    plt.title("Graph Visualization (Nodes and Edges)")
    plt.show(block=False)
