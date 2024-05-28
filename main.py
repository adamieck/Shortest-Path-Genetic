import numpy as np
import networkx as nx
from genetic_algorithm import initialize_population, evaluate, selection, crossover, mutate, calculate_roulette_weights
from utils import draw_graph, create_complete_graph, create_arek_graph
import sys
import imageio.v2 as imageio
import random
import pandas as pd
import os


def main():
    np.random.seed(5)
    
    vertex_count = 100
    
    # G = nx.tutte_graph()
    # # Assign random weights to the edges
    # for (u, v) in G.edges():
    #     G.edges[u, v]['weight'] = np.random.randint(1, 50)
    
    G = create_arek_graph()
    start = 23
    finish = 69

    df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
    for row, data in nx.shortest_path_length(G):
        for col, dist in data.items():
            df.loc[row, col] = dist

    df = df.fillna(df.max().max())

    # Use Kamada-Kawai layout with custom distances
    layout = nx.kamada_kawai_layout(G, dist=df.to_dict())


    # draw_graph(G, layout, genome=None, start=start, finish=finish, iteration=0, mutation_rate=0.1)
    # pos = nx.spring_layout(G, seed=1)
    # draw_graph(G, pos)

    path_length = nx.dijkstra_path_length(G, source=start, target=finish)
    print("Dijkstra result: " + str(path_length))

    generation_size = 10
    elite_size = 1
    num_nodes = len(G.nodes)
    population = initialize_population(generation_size, start, num_nodes)

    best_path_length = sys.maxsize
    best_genome = []
    # images = []
    results = []
    iterations = []
    num_trials = 1

    for i in range(num_trials):
        np.random.seed()
        population = initialize_population(generation_size, start, num_nodes)
        best_path_length = sys.maxsize
        no_improvement = 0
        max_no_improvement = 20
        iteration_count = 0

        while no_improvement < max_no_improvement:
            iteration_count += 1
            fitness_values = [evaluate(G, genome, start, finish) for genome in population]
            print(fitness_values)
            min_fitness = min(fitness_values)
            
            if min_fitness < best_path_length:
                best_path_length = min_fitness
                best_genome = population[np.argmin(fitness_values)]
                no_improvement = 0
            else:
                no_improvement += 1

            elite_indices = np.argsort(fitness_values)[:elite_size]
            elite_genomes = [population[idx] for idx in elite_indices]
            new_population = elite_genomes.copy()
            
            # draw_graph(G, layout, genome=best_genome, start=start, finish=finish, iteration=iteration_count, mutation_rate=0.1)
            # images.append(imageio.imread(f"frame_{iteration_count}.png"))
            
            norm_roulette_weights = calculate_roulette_weights(fitness_values)
            
        # Version with crossover
            while len(new_population) < generation_size:
                # Select two parents
                selected_population = selection(population, generation_size, 2, norm_roulette_weights)
                offspring = crossover(selected_population[0], selected_population[1])
                offspring = mutate(offspring)
                new_population.append(offspring)
        
            # # Version without crossover
            # selected_population = selection(population, generation_size, generation_size - elite_size, norm_roulette_weights)
            # for genome in selected_population:
            #     new_population.append(mutate(genome))
                    
            population = new_population
            
        print(f"Trial {i + 1} finished.")
        result_ratio = best_path_length / path_length
        results.append(result_ratio)
        iterations.append(iteration_count)
    
    results = np.array(results)
    iterations = np.array(iterations)
    mean_result = np.mean(results)
    median_result = np.median(results)
    max_5_results = np.sort(results)[-5:]
    max_5_results_str = "[" + "; ".join([f"{x:.2f}".replace('.', ',') for x in max_5_results]) + "]"
    mean_iterations = np.mean(iterations)
    
    print(f"Mean result ratio: {mean_result}")
    print(f"Median result ratio: {median_result}")
    print(f"5 maximum result ratios: {max_5_results_str}")
    print(f"Mean number of iterations: {mean_iterations}")

    # print("Best path length found: " + str(best_path_length))
    # print("Best genome: " + str(best_genome))
    
    # if images:
    #     imageio.mimwrite('genetic_algorithm.gif', images, fps = 4)
   # else:
    #    print("No images to create a GIF.")
    # remove the images
    # for i in range(iteration_count):
        # os.remove(f"frame_{i + 1}.png")

if __name__ == "__main__":
    main()