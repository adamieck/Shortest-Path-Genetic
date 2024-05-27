
import networkx as nx
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

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

def create_complete_graph(vertex_count):
    G = nx.complete_graph(vertex_count)
    permutation = np.random.permutation(vertex_count)
    
    for i in range(vertex_count - 1):
        u = permutation[i]
        v = permutation[i + 1]
        weight = np.random.randint(1, 3 * vertex_count)
        G.edges[u, v]['weight'] = weight
        G.edges[v, u]['weight'] = weight
    
    for i in range(vertex_count):
        for j in range(i + 1, vertex_count):
            if 'weight' not in G.edges[i, j]:
                weight = np.random.randint(vertex_count, 5 * vertex_count)
                G.edges[i, j]['weight'] = weight
                G.edges[j, i]['weight'] = weight
    
    return G
                