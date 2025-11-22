# Class BeliefSpace: Stores knowledge and guides the population
from algorithms.cultural.individual import Individual
import math
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from algorithms.cultural.individual import Individual, PopulationSpace

class BeliefSpace:
    def __init__(self, initial_upper_bound, max_k, mutation_rate):
        self.general_belief = initial_upper_bound
        self.group_metrics = {}  
        self.MAX_K = max_k # maximum number of individuals to inject when rg is 0 / stagnants are recurring
        self.MUTATION_RATE = mutation_rate

    def update_belief(self, best_indvidual: Individual) -> bool:
        new_belief = min(self.general_belief, best_indvidual.belief)
        if(new_belief < self.general_belief):
            self.general_belief = new_belief
            return True
        return False
    
    def calculate_rg(self, current_best_fitness, previous_best_fitness):
        if previous_best_fitness == 0:
            return 0.0
        # r_g = 1 - (current / previous)
        return 1.0 - (current_best_fitness / previous_best_fitness)
    
    def calculate_Kg(self, group_belief, rg):
        if rg <= 0.9 and group_belief >= self.general_belief - 2:
            return math.floor(self.MAX_K * (rg + 0.1))
        else:
            return math.floor(self.MAX_K * rg)
        
    def process_groups(self, population: List['Individual'], pop_space_ref: 'PopulationSpace'):
        group_improved = False
        groups_to_update = {}

        for individual in population:
            color_count = individual.belief
            if color_count not in groups_to_update:
                groups_to_update[color_count] = []
            groups_to_update[color_count].append(individual)

        for belief_group, individuals in groups_to_update.items():

            best_individual = min(individuals, key=lambda x: x.fitness)
            current_best_fitness = best_individual.fitness

            if belief_group not in self.group_metrics:
                self.group_metrics[belief_group] = {
                    'prev_fit': current_best_fitness,
                    'stagnant_count': 0,
                    'is_adapted': False
                }
            
            metrics = self.group_metrics[belief_group]
            prev_best_fitness = metrics['prev_fit']

            if(current_best_fitness < prev_best_fitness):
                metrics['is_adapted'] = False
                metrics['stagnant_count'] = 0
                metrics['prev_fit'] = current_best_fitness
                group_improved = True
            else:
                metrics['stagnant_count'] += 1

                if metrics['stagnant_count'] > 5 and not metrics['is_adapted']:
                    rg = self.calculate_rg(current_best_fitness, prev_best_fitness)
                    Kg = self.calculate_Kg(rg, belief_group)

                    for _ in range(Kg):
                        new_chromo = pop_space_ref.create_chromosome(belief_group)
                        new_ind = Individual(new_chromo)
                        pop_space_ref.calculate_fitness(new_ind)
                        population.append(new_ind)
                    
                    metrics['is_adapted'] = True
                    
            return group_improved

