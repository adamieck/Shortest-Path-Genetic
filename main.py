import numpy as np
import networkx as nx
from genetic_algorithm import initialize_population, evaluate, selection, crossover, mutate, calculate_roulette_weights
from utils import draw_graph, create_complete_graph
import sys
import random
import imageio.v2 as imageio

def create_graph():
    G = nx.Graph()
    edges = [
        (0, 1, 4), (0, 7, 8), (1, 2, 8), (1, 7, 11), (2, 3, 7), (2, 8, 2),
        (2, 5, 4), (3, 4, 9), (3, 5, 14), (4, 5, 10), (5, 6, 2), (6, 7, 1),
        (6, 8, 6), (7, 8, 7)
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
    pos = nx.spring_layout(G, seed=1)


    # pos = nx.spring_layout(G, seed=1)
    # draw_graph(G, pos)

    np.random.seed()

    start = 0
    finish = 4
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
    max_no_improvement = 20
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
            
        draw_graph(G, pos, genome=best_genome, start=start, finish=finish, iteration=iteration_count, mutation_rate=0.1)
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
        imageio.mimwrite('genetic_algorithm.gif', images, duration=5)
    else:
        print("No images to create a GIF.")

if __name__ == "__main__":
    main()