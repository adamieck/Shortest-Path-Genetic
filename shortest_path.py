import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import sys


def draw_graph(G, pos):

    #nodes
    nx.draw_networkx_nodes(G, pos, node_size=300)

    # edges
    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_edges(
        G, pos, width=2, alpha=0.5, edge_color="b", style="dashed"
    )

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    plt.show()


def mutate(lst, rate=0.08):
    for i in range(len(lst)):
        if random.random() <= rate:
            rand_idx = random.randint(0, len(lst) - 1)
            lst[i], lst[rand_idx] = lst[rand_idx], lst[i]
    return lst


def cross(lst1, lst2):
    to_cut = random.choice([lst1, lst2])

    if to_cut == lst1:
        other_list = lst2
    else:
        other_list = lst1

    cutoff = random.randint(1, len(to_cut))
    cut = to_cut[0:cutoff]
    return cut + [x for x in other_list if x not in cut]


def evaluate(G, lst, start, finish):
    v = start
    path_length = 0
    visited = [False] * (len(lst) + 2)
    visited[start] = True
    while v != finish:
        visited[v] = True
        neighbors = [neighbor for neighbor in list(G.neighbors(v)) if visited[neighbor] is False]
        if finish in neighbors:
            return path_length + G.edges[v, finish]['weight']
        if len(neighbors) == 0:
            return math.inf
        for element in lst:
            if element in neighbors:
                visited[element] = True
                path_length = path_length + G.edges[v, element]['weight']
                v = element
                break


def main():
    np.random.seed(5)
    G = nx.tutte_graph()

    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.randint(1, 50)

    pos = nx.spring_layout(G, seed=42)
    start = 31
    finish = 36
    path_length = nx.dijkstra_path_length(G, source=start, target=finish)
    print("Djikstra result: " + str(path_length))
    #draw_graph(G, pos)

    genoms = []
    generation_size = 20
    for _ in range(generation_size):
        genom = list(range(0, 46))
        genom.remove(start)
        genom.remove(finish)
        random.shuffle(genom)
        genoms.append(genom)

    best_path_length = sys.maxsize
    best_genom = []
    no_improvement = 0
    max_no_improvement = 20
    while no_improvement < max_no_improvement:
        evaluated_genoms = [evaluate(G, genom, start, finish) for genom in genoms]
        print(evaluated_genoms)
        print(min(evaluated_genoms))
        # loop condition
        if min(evaluated_genoms) < best_path_length:
            best_path_length = min(evaluated_genoms)
            best_genom = genoms[np.argmin(evaluated_genoms)]
            no_improvement = 0
        else:
            no_improvement += 1

        path_legths_inverted = [1/x for x in evaluated_genoms]
        p = min([x for x in path_legths_inverted if x != 0.0])
        roulette_weights = [np.sqrt(x) if x != 0.0 else np.sqrt(p/2) for x in path_legths_inverted]
        norm_roulette_weights = [x/sum(roulette_weights) for x in roulette_weights]
        new_genoms = [best_genom]
        for _ in range(generation_size - 1):
            parents_idxs = np.random.choice(range(generation_size), size=2, replace=False, p=norm_roulette_weights)
            new_genoms.append(mutate(cross(genoms[parents_idxs[0]], genoms[parents_idxs[1]])))
            #new_genoms.append(mutate(cross(genoms[parents_idxs[0]], genoms[parents_idxs[0]]))) #bez crossa
            #new_genoms.append(cross(genoms[parents_idxs[0]], genoms[parents_idxs[1]])) #bez mutacji
        genoms = new_genoms

if __name__ == "__main__":
    main()
