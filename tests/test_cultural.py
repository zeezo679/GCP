from algorithms.cultural import population_space
from utils.graph_generator import GraphGenerator


N_NODES = population_space.random_graph.n_nodes
print(N_NODES)

test_chrmsm = population_space.create_chromosome(N_NODES)
# print(test_chrmsm)

test_pop = population_space.initialize_population(10)

test_fitness = population_space.calculate_fitness(test_chrmsm)
# print(test_chrmsm)
# print(test_fitness)

population_space.selection(test_pop)