import numpy as np
import networkx as nx
from genetic_algorithm import initialize_population, evaluate, selection, crossover, mutate
from utils import draw_graph
import sys

def main():
    np.random.seed(5)
    
    G = nx.tutte_graph()


    # Assign random weights to the edges
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.randint(1, 50)

    pos = nx.spring_layout(G, seed=1)
    # draw_graph(G, pos)



    start = 31
    finish = 36
    path_length = nx.dijkstra_path_length(G, source=start, target=finish)
    print("Dijkstra result: " + str(path_length))

    generation_size = 20
    elite_size = 1
    num_nodes = len(G.nodes)
    population = initialize_population(generation_size, start, finish, num_nodes)

    best_path_length = sys.maxsize
    best_genome = []
    no_improvement = 0
    max_no_improvement = 20
    
    
    while no_improvement < max_no_improvement:
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

        elite_indices = np.argsort(fitness_values)[:elite_size]
        elite_genomes = [population[idx] for idx in elite_indices]
        new_population = elite_genomes.copy()
        
        while len(new_population) < generation_size:
            # Select two parents
            selected_population = selection(population, fitness_values, generation_size)
            offspring = crossover(selected_population[0], selected_population[1])
            offspring = mutate(offspring)
            new_population.append(offspring)
        
        population = new_population

    print("Best path length found: " + str(best_path_length))
    print("Best genome: " + str(best_genome))

if __name__ == "__main__":
    main()