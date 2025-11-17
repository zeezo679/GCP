# Class PopulationSpace:
# Handles individuals, crossover, and mutation.
import random
from utils.graph_generator import GraphGenerator


class PopulationSpace:
    def __init__(self):
        pass

random_graph = GraphGenerator("data\\sample_graphs\\graph_three")

def create_chromosome(n_nodes):
    chromosome = [random.randint(1, n_nodes) for _ in range(n_nodes)]
    return chromosome


def initialize_population(pop_size):
    pop = [create_chromosome(random_graph.n_nodes) for _ in range(pop_size)]
    return pop

def calculate_fitness(chromosome):
    bad_edges = 0
    for u, adj in random_graph.graph.items():
        u_color = chromosome[u-1]

        for v in adj:
            #condition to check adjacany once
            if(v > u):
                v_color = chromosome[v-1]
                if u_color == v_color:
                    bad_edges += 1
    return bad_edges

def selection(population):
    temp_parents = random.sample(population, 2)
    print(temp_parents)
    parent_one = min(temp_parents, key=calculate_fitness)

    temp_parents = random.sample(population, 2)
    print(temp_parents)
    parent_two = min(temp_parents, key=calculate_fitness)

    print(parent_one)
    print(parent_two)
