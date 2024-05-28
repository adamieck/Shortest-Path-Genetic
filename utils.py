
import networkx as nx
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

def draw_graph(G, pos):
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_labels(G, pos, font_size=7, font_family="sans-serif")
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.show()

def create_complete_graph(vertex_count):
    G = nx.complete_graph(vertex_count)
    permutation = np.random.permutation(vertex_count)
    start = permutation[0]
    finish = permutation[-1]
    
    for i in range(vertex_count - 1):
        u = permutation[i]
        v = permutation[i + 1]
        weight = 1
        G.edges[u, v]['weight'] = weight
        G.edges[v, u]['weight'] = weight
    
    for i in range(vertex_count):
        for j in range(i + 1, vertex_count):
            if 'weight' not in G.edges[i, j]:
                weight = np.random.randint(2 * vertex_count, 5 * vertex_count)
                G.edges[i, j]['weight'] = weight
                G.edges[j, i]['weight'] = weight
    
    return (G, start, finish)
                