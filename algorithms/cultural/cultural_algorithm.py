# Class CulturalAlgorithm:
# Combines population and belief spaces to evolve the best coloring.
from algorithms.cultural.population_space import PopulationSpace
from algorithms.cultural.belief_space import BeliefSpace
import time


class CulturalAlgorithm:
    def __init__(self, pop_size=100, max_stagnation_tries=50, 
                 max_k=10,mutation_rate=0.1, mutation_increase_factor=2.0, graph_path="data\\sample_graphs\\graph_two"):
       
        self.pop_size = pop_size
        self.max_stagnation_tries = max_stagnation_tries
        self.initial_muation_rate = mutation_rate

        self.pop_space = PopulationSpace(self.pop_size, graph_path)
        self.initial_upper_bound = self.pop_space.n_nodes # safe upper bound initialization
        self.belief_space = BeliefSpace(self.initial_upper_bound, max_k, mutation_increase_factor)
        self.population = []


    def run_ca(self):
       start_time = time.time()

        # With EP
       initial_bound, initial_pop = self.pop_space.run_estimation_phase()
       self.belief_space.general_belief = initial_bound # set B
       self.population = initial_pop
       
       # Without EP
       # self.population = self.pop_space.initialize_population(self.initial_upper_bound)
       

       best_initial = min(self.population, key=lambda x: x.fitness)
       print("--- CA Initialization ---")
       print(f"Graph Nodes: {self.pop_space.n_nodes}")
       print(f"Initial Pop Size: {self.pop_size}, General Belief (EP): {self.belief_space.general_belief}")
       print(f"Initial Best Fitness: {best_initial.fitness}")

       self.run_improvement_phase()
        
       end_time = time.time()
       total_time = end_time - start_time

      # --- Final Metrics Report ---
       final_best = min(self.population, key=lambda x: x.fitness)
       print("\n--- CA Optimization Finished ---")
       print(f"1. Final Chromatic Number (Belief): {final_best.belief}")
       print(f"2. Final Minimum Conflicts (Fitness): {final_best.fitness}")
       print(f"3. Total Computational Time: {total_time:.4f} seconds")
        
       if final_best.fitness == 0:
           print("Status: SUCCESS - A valid coloring was found!")
       else:
           print("Status: Local optimum found - Conflicts remain.")
       
       # Return results for web interface
       return {
           'best_fitness': final_best.fitness,
           'best_chromosome': final_best.chromosome,
           'colors_used': final_best.belief,
           'execution_time': total_time,
           'iterations': self.max_stagnation_tries
       } 
    
    def run_improvement_phase(self):
        T = 0     # Stagnation count
        S = self.max_stagnation_tries 

        while T < S:
            best_of_gen = self.pop_space.evaluate_and_get_best(self.population)
            old_general_belief = self.belief_space.general_belief

            belief_changed = self.belief_space.update_belief(best_of_gen)
            group_improved = self.belief_space.process_groups(self.population, self.pop_space)

            if not group_improved:
                T += 1
            else:
                T = 0
            
            # influencing population
            if belief_changed:
                self.population = [ind for ind in self.population if ind.belief <= self.belief_space.general_belief]
                print(f"  > Gen {T}: General Belief reduced to {self.belief_space.general_belief}. Population restricted.")

            current_mutation_rate = self.belief_space.get_mutation_rate(
                self.initial_muation_rate,
                best_of_gen.belief
            )
            new_individuals = self.pop_space.perform_variation(
                self.population,  #to alter the population
                self.pop_size, #number of children to generate
                self.belief_space.general_belief, # for mutation
                current_mutation_rate #mutation factor
            )

            self.population.extend(new_individuals)
            self.population.sort(key=lambda x: x.fitness) 
            self.population = self.population[:self.pop_size]

            # Monitoring 
            if T > 0:
                 current_best_fit = self.population[0].fitness
                 current_best_belief = self.population[0].belief
                
                 print(f"  > T={T} (Stagnant Gens). Conflicts={current_best_fit}, Colors={current_best_belief}")
            