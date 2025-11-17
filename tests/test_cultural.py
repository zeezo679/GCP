from algorithms.cultural import population_space
from utils.graph_generator import GraphGenerator


random_graph = GraphGenerator("data\\sample_graphs\\graph_three")

N_NODES = random_graph.n_nodes
print(N_NODES)

test_chrmsm = population_space.create_chromosome(N_NODES)

test_pop = population_space.initialize_population(10)

test_fitness = population_space.calculate_fitness(test_chrmsm)

