import numpy as np
import networkx as nx
from genetic_algorithm import initialize_population, evaluate, selection, crossover, mutate, calculate_roulette_weights
from utils import draw_graph, create_arek_graph
import sys
import imageio.v2 as imageio
import random
import pandas as pd
import os


def main():
    np.random.seed(5)
    
    G = create_arek_graph()

    # df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
    # for row, data in nx.shortest_path_length(G):
    #     for col, dist in data.items():
    #         df.loc[row, col] = dist

    # df = df.fillna(df.max().max())

    # Use Kamada-Kawai layout with custom distances
    # layout = nx.kamada_kawai_layout(G, dist=df.to_dict())
    generation_size = 20
    elite_size = 1
    num_nodes = len(G.nodes)

    num_trials = 3
    crossover_results = []
    crossover_iterations = []
    non_crossover_results = []
    non_crossover_iterations = []

    for i in range(num_trials):
        np.random.seed()
        start, finish = random.sample(range(num_nodes), 2)
        path_length = nx.dijkstra_path_length(G, source=start, target=finish)
        print(f"Dijkstra result: {path_length}")

        # Genetic algorithm with crossover
        population = initialize_population(generation_size, start, num_nodes)
        best_path_length = sys.maxsize
        no_improvement = 0
        max_no_improvement = 20
        iteration_count = 0

        while no_improvement < max_no_improvement:
            iteration_count += 1
            fitness_values = [evaluate(G, genome, start, finish) for genome in population]
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

            norm_roulette_weights = calculate_roulette_weights(fitness_values)

            while len(new_population) < generation_size:
                selected_population = selection(population, generation_size, 2, norm_roulette_weights)
                offspring = crossover(selected_population[0], selected_population[1])
                offspring = mutate(offspring)
                new_population.append(offspring)

            population = new_population

        crossover_results.append(best_path_length / path_length)
        crossover_iterations.append(iteration_count)

        # Genetic algorithm without crossover
        population = initialize_population(generation_size, start, num_nodes)
        best_path_length = sys.maxsize
        no_improvement = 0
        iteration_count = 0

        while no_improvement < max_no_improvement:
            iteration_count += 1
            fitness_values = [evaluate(G, genome, start, finish) for genome in population]
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

            norm_roulette_weights = calculate_roulette_weights(fitness_values)

            selected_population = selection(population, generation_size, generation_size - elite_size, norm_roulette_weights)
            for genome in selected_population:
                new_population.append(mutate(genome))

            population = new_population

        non_crossover_results.append(best_path_length / path_length)
        non_crossover_iterations.append(iteration_count)

        print(f"Trial {i + 1} finished.")

    # Calculate and print statistics for crossover
    crossover_results = np.array(crossover_results)
    crossover_iterations = np.array(crossover_iterations)
    mean_crossover_result = np.mean(crossover_results)
    median_crossover_result = np.median(crossover_results)
    max_5_crossover_results = np.sort(crossover_results)[-5:]
    max_5_crossover_results_str = "[" + "; ".join([f"{x:.2f}".replace('.', ',') for x in max_5_crossover_results]) + "]"
    mean_crossover_iterations = np.mean(crossover_iterations)
    
    print(f"Mean crossover result ratio: {mean_crossover_result}")
    print(f"Median crossover result ratio: {median_crossover_result}")
    print(f"5 maximum crossover result ratios: {max_5_crossover_results_str}")
    print(f"Mean number of crossover iterations: {mean_crossover_iterations}")

    # Calculate and print statistics for non-crossover
    non_crossover_results = np.array(non_crossover_results)
    non_crossover_iterations = np.array(non_crossover_iterations)
    mean_non_crossover_result = np.mean(non_crossover_results)
    median_non_crossover_result = np.median(non_crossover_results)
    max_5_non_crossover_results = np.sort(non_crossover_results)[-5:]
    max_5_non_crossover_results_str = "[" + "; ".join([f"{x:.2f}".replace('.', ',') for x in max_5_non_crossover_results]) + "]"
    mean_non_crossover_iterations = np.mean(non_crossover_iterations)
    
    print(f"Mean non-crossover result ratio: {mean_non_crossover_result}")
    print(f"Median non-crossover result ratio: {median_non_crossover_result}")
    print(f"5 maximum non-crossover result ratios: {max_5_non_crossover_results_str}")
    print(f"Mean number of non-crossover iterations: {mean_non_crossover_iterations}")

if __name__ == "__main__":
    main()
