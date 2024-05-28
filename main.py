import numpy as np
import networkx as nx
from genetic_algorithm import initialize_population, evaluate, selection, crossover, mutate, calculate_roulette_weights
from utils import draw_graph, create_complete_graph
import sys
import random
import imageio.v2 as imageio
import os
import pandas as pd

def create_graph():
    G = nx.Graph()
    edges = [
        (0, 1, 10), (0, 2, 5), (1, 2, 2), (1, 3, 1), (2, 3, 9), (2, 4, 2), 
        (3, 4, 4), (3, 5, 6), (4, 5, 7), (4, 6, 3), (5, 6, 1), (5, 7, 2), 
        (6, 7, 5), (6, 8, 4), (7, 8, 1), (7, 9, 2), (8, 9, 3), (8, 10, 6), 
        (9, 10, 2), (10, 11, 1), (11, 12, 4), (11, 13, 5), (12, 13, 2)
    ]
    G.add_weighted_edges_from(edges)
    return G

def main():
    # print(1 / np.inf)
    np.random.seed(5)
    
    # vertex_count = 46
    
    # G = nx.tutte_graph()
    # # Assign random weights to the edges
    # for (u, v) in G.edges():
    #     G.edges[u, v]['weight'] = np.random.randint(1, 50)
    G = create_graph()

    df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
    for row, data in nx.shortest_path_length(G):
        for col, dist in data.items():
            df.loc[row, col] = dist

    df = df.fillna(df.max().max())

    # Use Kamada-Kawai layout with custom distances
    layout = nx.kamada_kawai_layout(G, dist=df.to_dict())



    # pos = nx.spring_layout(G, seed=1)
    # draw_graph(G, pos)

    np.random.seed()

    start = 0
    finish = 13
    path_length = nx.dijkstra_path_length(G, source=start, target=finish)
    print("Dijkstra result: " + str(path_length))

    generation_size = 5
    elite_size = 1
    num_nodes = len(G.nodes)
    population = initialize_population(generation_size, start, num_nodes)

    best_path_length = sys.maxsize
    best_genome = []
    iteration_count = 0
    no_improvement = 0
    max_no_improvement = 10
    images = []
    
    
    while no_improvement < max_no_improvement:
        iteration_count += 1
        fitness_values = [evaluate(G, genome, start, finish) for genome in population]
        print(fitness_values)
        print(min(fitness_values))
        
        # Update the best solution if a better one is found
        if min(fitness_values) < best_path_length:
            best_path_length = min(fitness_values)
            best_genome = population[np.argmin(fitness_values)]
            no_improvement = 0
        else:
            no_improvement += 1
            
        draw_graph(G, layout, genome=best_genome, start=start, finish=finish, iteration=iteration_count, mutation_rate=0.1)
        images.append(imageio.imread(f"frame_{iteration_count}.png"))

        elite_indices = np.argsort(fitness_values)[:elite_size]
        elite_genomes = [population[idx] for idx in elite_indices]
        new_population = elite_genomes.copy()
        
        norm_roulette_weights = calculate_roulette_weights(fitness_values)
        
        # Version with crossover
        while len(new_population) < generation_size:
            # Select two parents
            selected_population = selection(population, generation_size, 2, norm_roulette_weights)
            offspring = crossover(selected_population[0], selected_population[1])
            offspring = mutate(offspring)
            new_population.append(offspring)
        
        # Version without crossover
        # selected_population = selection(population, generation_size, generation_size - elite_size, norm_roulette_weights)
        # for genome in selected_population:
        #     new_population.append(mutate(genome))
                
        population = new_population

    print("Best path length found: " + str(best_path_length))
    print("Best genome: " + str(best_genome))
    
    if images:
        imageio.mimwrite('genetic_algorithm.gif', images, fps = 4)
    else:
        print("No images to create a GIF.")
    # remove the images
    for i in range(iteration_count):
        os.remove(f"frame_{i + 1}.png")

if __name__ == "__main__":
    main()