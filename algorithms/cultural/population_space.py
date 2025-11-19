# Class PopulationSpace:
# Handles individuals, crossover, and mutation.
import random
from utils.graph_generator import GraphGenerator


class PopulationSpace:
    def __init__(self, pop_size):
        self.random_graph = GraphGenerator("data\\sample_graphs\\graph_two")
        self.pop_size = pop_size
        
    def create_chromosome(self, n_nodes):
        chromosome = [random.randint(1, n_nodes) for _ in range(n_nodes)]
        return chromosome

    def initialize_population(self):
        pop = [self.create_chromosome(self.random_graph.n_nodes) for _ in range(self.pop_size)]
        return pop
    
    def calculate_fitness(self, chromosome):
        bad_edges = 0
        
        for u, adj in self.random_graph.graph.items():
            u_color = chromosome[u-1]

            for v in adj:
                #condition to check adjacany once
                if(v > u):
                    v_color = chromosome[v-1]
                    if u_color == v_color:
                        bad_edges += 1
        return bad_edges

    def selection(self, population):
        temp_parents = random.sample(population, 2)
        print(f"Temp Parent one {temp_parents}")
        print(f"First parentA fitness: {self.calculate_fitness(temp_parents[0])}")
        print(f"Second parentA fitness: {self.calculate_fitness(temp_parents[1])}")
        parent_one = min(temp_parents, key=self.calculate_fitness)
        print("-" * 50)
        temp_parents = random.sample(population, 2)
        print(f"Temp Parent two {temp_parents}")
        print(f"First parentB fitness: {self.calculate_fitness(temp_parents[0])}")
        print(f"Second parentB fitness: {self.calculate_fitness(temp_parents[1])}")
        print("#"*50)
        parent_two = min(temp_parents, key=self.calculate_fitness)
        print(parent_one)
        print("-" * 50)
        print(parent_two)
        return parent_one, parent_two

    def crossover(self, parent_one, parent_two):
        n = len(parent_one)
        crosspoint = random.randint(0, n-2)
        child = parent_one[:crosspoint+1] + parent_two[crosspoint + 1:]
        # print(crosspoint)
        print(self.calculate_fitness(parent_one))
        print(self.calculate_fitness(parent_two))
        print(self.calculate_fitness(child))
        return child

