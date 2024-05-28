
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



def draw_graph(G, pos, genome=None, start=None, finish=None, iteration=None, mutation_rate=None):
    plt.clf()
    plt.title(f"Generation {iteration}")
    nx.draw(G, pos, with_labels=True, node_color='lightpink', node_size=700, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    draw_red_edges = True
    
    if genome is not None and start is not None and finish is not None:
        current_node = start
        best_path = [start]
        visited = [False] * len(G.nodes)
        visited[start] = True
    
        while current_node != finish:
            neighbors = [neighbor for neighbor in G.neighbors(current_node) 
                        if not visited[neighbor]]
            if not neighbors:
                draw_red_edges = False
                break
        
            for next_node in genome:
                if next_node in neighbors:
                    current_node = next_node
                    best_path.append(next_node)
                    visited[next_node] = True
                    break
        
        if draw_red_edges and best_path[-1] != finish:
            best_path.append(finish)
            
        if draw_red_edges:
            edges = [(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', width=2)

    plt.savefig(f"frame_{iteration}.png")
    

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
                