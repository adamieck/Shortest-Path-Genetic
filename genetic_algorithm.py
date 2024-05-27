import networkx as nx
import numpy as np

def initialize_population(size, start, finish, num_nodes):
    population = []
    # Generate random genomes by making permutations of all nodes except start
    for _ in range(size):
        genome = list(range(num_nodes))
        genome.remove(start)
        np.random.shuffle(genome)
        population.append(genome)
    return population

# Fitness function - calculates the length of the path by following the order of nodes in the genome
def evaluate(G, genome, start, finish):
    current_node = start
    path_length = 0
    visited = [False] * len(G.nodes)
    visited[start] = True
    
    while current_node != finish:
        neighbors = [neighbor for neighbor in G.neighbors(current_node) if not visited[neighbor]]
        if not neighbors:
            return np.inf
        
        for next_node in genome:
            if next_node in neighbors:
                path_length += G.edges[current_node, next_node]['weight']
                current_node = next_node
                visited[next_node] = True
                break
            
    return path_length

def calculate_roulette_weights(fitness_values):
    path_lengths_inverted = np.where(fitness_values != 0, 1 / np.array(fitness_values), 0)
    min_nonzero = np.min(path_lengths_inverted[path_lengths_inverted != 0])
    roulette_weights = np.where(path_lengths_inverted != 0, np.sqrt(path_lengths_inverted), np.sqrt(min_nonzero / 2))
    norm_roulette_weights = roulette_weights / np.sum(roulette_weights)
    return norm_roulette_weights

# Selection function - we will use the roulette wheel selection
def selection(genomes, generation_size, num_chosen, norm_roulette_weights):
    selected_genomes = []
    selected_indices = np.random.choice(range(generation_size), size=num_chosen, replace=False, p=norm_roulette_weights)
    for idx in selected_indices:
        selected_genomes.append(genomes[idx])
    return selected_genomes
    

# Crossover function:
# 1. We will keep first k elements from the first parent.
# 2. We will then add the remaining elements from the other parent in the order they appear. We will not add duplicates.
def crossover(parent1, parent2):
    # We want to preserve at least a part of each parent
    cutoff = np.random.randint(1, len(parent1) - 1)
    cut = parent1[:cutoff]
    child_genome = cut + [x for x in parent2 if x not in cut]
    return child_genome
        

# Mutating a list genome with a given rate
def mutate(genome, rate=0.1):
    for i in range(len(genome)):
        if np.random.random() <= rate:
            rand_idx = np.random.randint(0, len(genome))
            genome[i], genome[rand_idx] = genome[rand_idx], genome[i]
    return genome
