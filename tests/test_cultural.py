from algorithms.cultural import population_space
from utils.graph_generator import GraphGenerator
from algorithms.cultural.population_space import PopulationSpace
from typing import Dict, List
import pdb
import random


popInitialize = PopulationSpace(100)
N_NODES = popInitialize.random_graph.n_nodes
print(N_NODES)


test_chrmsm = popInitialize.create_chromosome(N_NODES)

test_pop = popInitialize.initialize_population()


test_fitness = popInitialize.calculate_fitness(test_chrmsm)


p1, p2 = popInitialize.selection(test_pop)

popInitialize.crossover(p1,p2)

mutated_chromosome = popInitialize.mutation(test_chrmsm, [random.randint(1, popInitialize.random_graph.n_nodes) for _ in range(popInitialize.random_graph.n_nodes)])


def runGA(pop: List[int], results_tracker: List[int]):
    p1, p2 = popInitialize.selection_two(pop)
    child = popInitialize.crossover(p1,p2)
    child = popInitialize.mutation(child,[random.randint(1, popInitialize.random_graph.n_nodes) for _ in range(popInitialize.random_graph.n_nodes)])
    pop.append(child)


    current_generation_fitness = [popInitialize.calculate_fitness(individual) for individual in pop]
    best_fitness = min(current_generation_fitness)
    print(f" (Best Fitness: {best_fitness}) -> Exploitation")
    results_tracker.append(best_fitness)


popul = popInitialize.initialize_population()
# print(f"this is our population {popul}")
runs = 20000
temp = runs
generation_results = []
while(runs != 0):
    runGA(popul, generation_results)

    # print(f"this is the generational results {generation_results}")
    current_gen = temp - runs + 1
    print(f"Generation {current_gen}: Best Fitness = {generation_results[-1]}")
    if(generation_results[-1] == 0):
        break

    runs -= 1

# --- FINAL RESULTS ---
print("\n--- Final Summary ---")
print("Best Fitness per Generation:", generation_results)


best_overall_fitness = min(generation_results)
print(f"Overall Best Fitness Achieved: {best_overall_fitness}")
