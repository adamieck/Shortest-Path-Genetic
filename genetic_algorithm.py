import networkx as nx
import numpy as np

def initialize_population(size, start, finish, num_nodes):
    population = []
    # Generate random genomes by making permutations of all nodes except start and finish
    for _ in range(size):
        genome = list(range(num_nodes))
        genome.remove(start)
        genome.remove(finish)
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
        visited[current_node] = True
        neighbors = [neighbor for neighbor in G.neighbors(current_node) if not visited[neighbor]]
        
        if finish in neighbors:
            return path_length + G.edges[current_node, finish]['weight']
        
        if not neighbors:
            return np.inf
        
        for next_node in genome:
            if next_node in neighbors:
                visited[next_node] = True
                path_length += G.edges[current_node, next_node]['weight']
                current_node = next_node
                break
        else:
            # If no valid next node is found in the genome, return infinity
            return np.inf
    
    return path_length

# Selection function - we will use the roulette wheel selection
def selection(genomes, fitness_values):
    path_lengths_inverted = np.where(fitness_values != 0, 1 / np.array(fitness_values), 0)
    min_nonzero = np.min(path_lengths_inverted[path_lengths_inverted != 0])
    roulette_weights = np.where(path_lengths_inverted != 0, np.sqrt(path_lengths_inverted), np.sqrt(min_nonzero / 2))
    norm_roulette_weights = roulette_weights / np.sum(roulette_weights)
    
    selected_genomes = []
    selected_indices = np.random.choice(range(len(genomes)), size=2, replace=False, p=norm_roulette_weights)
    selected_genomes.append(genomes[selected_indices[0]])
    selected_genomes.append(genomes[selected_indices[1]])
    
    return selected_genomes
    

# Crossover function:
# 1. We will keep first k elements from the first parent.
# 2. We will then add the remaining elements from the other parent in the order they appear. We will not add duplicates.
def crossover(parent1, parent2):
    cutoff = np.random.randint(1, len(parent1))
    cut = parent1[:cutoff]
    child_genome = cut + [x for x in parent2 if x not in cut]
    return child_genome
        

# Mutating a list genome with a given rate
def mutate(genome, rate=0.1):
    for i in range(len(genome)):
        if np.random.random() <= rate:
            rand_idx = np.random.randint(0, len(genome) - 1)
            genome[i], genome[rand_idx] = genome[rand_idx], genome[i]
    return genome
