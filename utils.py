
import networkx as nx
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def draw_graph(G, pos):
    """
    Draws the graph with nodes and edges.

    Parameters:
    G (networkx.Graph): The graph to be drawn.
    pos (dict): A dictionary with nodes as keys and positions as values.
    """
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_labels(G, pos, font_size=7, font_family="sans-serif")
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.show()
