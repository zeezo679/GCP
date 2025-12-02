# # Class BeliefSpace: Stores knowledge and guides the population
# from algorithms.cultural.individual import Individual
# import math
# from typing import TYPE_CHECKING, List

# if TYPE_CHECKING:
#     from algorithms.cultural.individual import Individual, PopulationSpace

# class BeliefSpace:
#     def __init__(self, initial_upper_bound, max_k, mutation_rate):
#         self.general_belief = initial_upper_bound
#         self.group_metrics = {}  
#         self.MAX_K = max_k # maximum number of individuals to inject when rg is 0 / stagnants are recurring
#         self.MUTATION_RATE = mutation_rate

#     def update_belief(self, best_indvidual: Individual) -> bool:
#         new_belief = min(self.general_belief, best_indvidual.belief)
#         if(new_belief < self.general_belief):
#             self.general_belief = new_belief
#             return True
#         return False
    
#     def calculate_rg(self, current_best_fitness, previous_best_fitness):
#         if previous_best_fitness == 0:
#             return 0.0
#         # r_g = 1 - (current / previous)
#         return 1.0 - (current_best_fitness / previous_best_fitness)
    
#     def calculate_Kg(self, group_belief, rg):
#         if rg <= 0.9 and group_belief >= self.general_belief - 2:
#             return math.floor(self.MAX_K * (rg + 0.1))
#         else:
#             return math.floor(self.MAX_K * rg)
        
#     def process_groups(self, population: List['Individual'], pop_space_ref: 'PopulationSpace'):
#         group_improved = False
#         groups_to_update = {}

#         for individual in population:
#             color_count = individual.belief
#             if color_count not in groups_to_update:
#                 groups_to_update[color_count] = []
#             groups_to_update[color_count].append(individual)

#         for belief_group, individuals in groups_to_update.items():

#             best_individual = min(individuals, key=lambda x: x.fitness)
#             current_best_fitness = best_individual.fitness

#             if belief_group not in self.group_metrics:
#                 self.group_metrics[belief_group] = {
#                     'prev_fit': current_best_fitness,
#                     'stagnant_count': 0,
#                     'is_adapted': False
#                 }
            
#             metrics = self.group_metrics[belief_group]
#             prev_best_fitness = metrics['prev_fit']

#             if(current_best_fitness < prev_best_fitness):
#                 metrics['is_adapted'] = False
#                 metrics['stagnant_count'] = 0
#                 metrics['prev_fit'] = current_best_fitness
#                 group_improved = True
#             else:
#                 metrics['stagnant_count'] += 1

#                 if metrics['stagnant_count'] > 5 and not metrics['is_adapted']:
#                     rg = self.calculate_rg(current_best_fitness, prev_best_fitness)
#                     Kg = self.calculate_Kg(rg, belief_group)

#                     for _ in range(Kg):
#                         new_chromo = pop_space_ref.create_chromosome(belief_group)
#                         new_ind = Individual(new_chromo)
#                         pop_space_ref.calculate_fitness(new_ind)
#                         population.append(new_ind)
                    
#                     metrics['is_adapted'] = True
                    
#             return group_improved












import random
import math
import time
from typing import TYPE_CHECKING # Import for forward reference typing

# Use TYPE_CHECKING guard for complex circular imports or late-defined classes
if TYPE_CHECKING:
    from .cultural_algorithm import Individual # Assuming the file structure for clarity

# --- 1. Helper Classes ---

class GraphGenerator:
# ... (rest of GraphGenerator class remains unchanged)
    """
    Mock class to simulate loading an adjacency list graph.
    Nodes are 1-indexed for simplicity to align with common array usage (index 0 is node 1).
    """
    def __init__(self, path):
        # Example graph (4 nodes, 4 edges): C4 cycle graph with an extra edge
        # 1-2, 2-3, 3-4, 4-1, 1-3
        self.graph = {
            1: [2, 3, 4],
            2: [1, 3],
            3: [1, 2, 4],
            4: [1, 3]
        }
        self.n_nodes = len(self.graph)
        print(f"Graph loaded with {self.n_nodes} nodes.")

class Individual:
# ... (rest of Individual class remains unchanged)
    """
    Represents a single solution (chromosome) and its associated metrics.
    This links the micro-level search to the macro-level belief system.
    """
    def __init__(self, chromosome, fitness=None):
        self.chromosome = chromosome  # The coloring array (e.g., [1, 2, 1, 3])
        self.fitness = fitness        # Number of conflicts (0 is optimal)
        # The number of unique colors used (this is the Individual's 'Belief')
        self.belief = len(set(chromosome))

    def __lt__(self, other):
        # Comparison operator for selection (lower fitness is better)
        return self.fitness < other.fitness

# --- 2. BeliefSpace Class (The Macro-Level Knowledge) ---

class BeliefSpace:
    """
    Manages the collective knowledge, including the General Belief and group metrics
    for self-adaptiveness (r_g, K_g).
    """
    def __init__(self, initial_upper_bound, max_k, mutation_increase_factor):
        # General Belief (B) is the overall minimum color count found
        self.general_belief = initial_upper_bound 
        
        # Stores group performance metrics. Key: belief (color count)
        # Value: {'prev_fit': int, 'current_fit': int, 'stagnant_count': int, 'is_adapted': bool}
        self.group_metrics = {}
        self.MAX_K = max_k # K: Max individuals to inject
        self.MUTATION_INCREASE_FACTOR = mutation_increase_factor

    def update_general_belief(self, best_individual: 'Individual') -> bool:
        """Updates the general belief if a better (fewer colors) individual is found."""
        new_belief = min(self.general_belief, best_individual.belief)
        if new_belief < self.general_belief:
            self.general_belief = new_belief
            return True
        return False

    def calculate_rg(self, current_best_fitness, previous_best_fitness):
# ... (rest of BeliefSpace class remains unchanged)
        """Calculates the Ratio of Improvement (r_g)."""
        if previous_best_fitness == 0:
            return 0.0 # Already perfect
        # r_g = 1 - (current / previous)
        return 1.0 - (current_best_fitness / previous_best_fitness)

    def calculate_Kg(self, rg, group_belief):
        """Calculates the number of random individuals to inject using the piecewise formula."""
        
        # Condition Check: (r_g <= 0.9 AND belief_g >= general_belief - 2)
        if rg <= 0.9 and group_belief >= self.general_belief - 2:
            # Case 1: Promising, stalled group gets bonus (+0.1)
            return math.floor(self.MAX_K * (rg + 0.1))
        else:
            # Case 2: Standard scaling
            return math.floor(self.MAX_K * rg)

    def process_groups(self, population, pop_space_ref):
        """
        Partitions the population, tracks progress (Step 5), and applies K_g injection.
        Returns True if any group improved its fitness.
        """
        group_improved = False
        groups_to_update = {}
        
        # 1. Partition Population by Belief (group_g)
        for individual in population:
            color_count = individual.belief
            if color_count not in groups_to_update:
                groups_to_update[color_count] = []
            groups_to_update[color_count].append(individual)

        # 2. Iterate and apply Self-Adaptiveness (Step 5)
        for belief_g, individuals in groups_to_update.items():
            
            # Find the best fitness within this group
            best_individual = min(individuals, key=lambda x: x.fitness)
            current_best_fitness = best_individual.fitness

            # Initialize metrics if group is new
            if belief_g not in self.group_metrics:
                self.group_metrics[belief_g] = {
                    'prev_fit': current_best_fitness, 
                    'stagnant_count': 0,
                    'is_adapted': False # Tracks if mutation rate is currently increased
                }

            metrics = self.group_metrics[belief_g]
            previous_best_fitness = metrics['prev_fit']

            # Check for improvement (lower fitness is better)
            if current_best_fitness < previous_best_fitness:
                # Progress was made (Step 5, Otherwise clause)
                
                # Reset adaptive changes
                metrics['is_adapted'] = False 
                metrics['stagnant_count'] = 0
                metrics['prev_fit'] = current_best_fitness
                group_improved = True
                
            else:
                # No progress was made (Step 5, If clause)
                metrics['stagnant_count'] += 1
                
                # Check for stagnation threshold (Example threshold: 5 generations)
                if metrics['stagnant_count'] > 5 and not metrics['is_adapted']:
                    
                    rg = self.calculate_rg(current_best_fitness, previous_best_fitness)
                    Kg = self.calculate_Kg(rg, belief_g)
                    
                    # Inject K_g random individuals (new chromosomes)
                    for _ in range(Kg):
                        # Use the group's belief as the maximum color count for the new individual
                        new_chromo = pop_space_ref.create_chromosome(belief_g)
                        new_ind = Individual(new_chromo)
                        pop_space_ref.calculate_fitness(new_ind)
                        population.append(new_ind)
                        
                    # Increase the mutation chance for this group
                    metrics['is_adapted'] = True
                    # The actual mutation rate is fetched by the Cultural class later

        return group_improved
    
    def get_mutation_rate(self, default_rate, belief_g=None):
        """Returns the base rate or an increased rate if the group is adapted."""
        if belief_g is not None and belief_g in self.group_metrics and self.group_metrics[belief_g]['is_adapted']:
            # Use the increased factor
            return min(1.0, default_rate * self.MUTATION_INCREASE_FACTOR)
        return default_rate

# --- 3. PopulationSpace Class (The Micro-Level Search) ---

class PopulationSpace:
# ... (rest of PopulationSpace class remains unchanged)
    """
    Handles individuals, graph operations, and genetic operators.
    """
    def __init__(self, pop_size, graph_path="data\\sample_graphs\\graph_one"):
        self.random_graph = GraphGenerator(graph_path)
        self.pop_size = pop_size
        self.n_nodes = self.random_graph.n_nodes

    def create_chromosome(self, max_colors):
        """Creates a chromosome using colors from 1 up to max_colors."""
        return [random.randint(1, max_colors) for _ in range(self.n_nodes)]

    def initialize_population(self, max_colors):
        """Initializes a population for the EP."""
        pop = [Individual(self.create_chromosome(max_colors)) for _ in range(self.pop_size)]
        return pop
    
    def calculate_fitness(self, individual):
        """Calculates fitness (number of bad_edges/conflicts)."""
        chromosome = individual.chromosome
        bad_edges = 0
        
        for u, adj in self.random_graph.graph.items():
            u_color = chromosome[u-1]
            for v in adj:
                # Check each edge once by ensuring v > u (to avoid double counting)
                if (v > u) and (u_color == chromosome[v-1]):
                    bad_edges += 1
        
        individual.fitness = bad_edges
        return bad_edges

    def evaluate_and_get_best(self, population):
        """Evaluates fitness for all individuals and returns the best overall."""
        for ind in population:
            self.calculate_fitness(ind)
        return min(population, key=lambda x: x.fitness)

    def run_estimation_phase(self):
        """Executes the EP: Finds the initial upper bound (General Belief)."""
        initial_pop = self.initialize_population(self.n_nodes)
        
        # Evaluate to calculate fitness and belief for all
        for ind in initial_pop:
            self.calculate_fitness(ind)
        
        # Find the individual with the lowest 'belief' (fewest colors used)
        best_ep_individual = min(initial_pop, key=lambda x: x.belief)
        
        return best_ep_individual.belief, initial_pop

    def selection(self, population):
        """Tournament Selection (pick best from random groups of 2)."""
        parent_one = min(random.sample(population, 2), key=lambda x: x.fitness)
        parent_two = min(random.sample(population, 2), key=lambda x: x.fitness)
        return parent_one, parent_two

    def crossover(self, parent_one, parent_two):
        """One-point crossover."""
        n = self.n_nodes
        crosspoint = random.randint(0, n-2)
        
        child_chromo = parent_one.chromosome[:crosspoint+1] + parent_two.chromosome[crosspoint + 1:]
        return Individual(child_chromo)

    def mutation(self, child_individual, general_belief):
        """
        Implements Stochastic Color Change mutation, constrained by the General Belief.
        """
        chromo = child_individual.chromosome
        # Create the available color pool based on the General Belief
        available_colors = list(range(1, general_belief + 1)) 
        
        # Select a random vertex to mutate
        vertex_index = random.randint(0, self.n_nodes - 1)
        
        # Change the color randomly using the constrained colors
        chromo[vertex_index] = random.choice(available_colors)
        
        return child_individual
    
    def perform_variation(self, population, pop_size, general_belief, mutation_rate):
        """
        Performs selection, crossover, and mutation to generate a new population,
        which will then be merged with the old population for selection.
        """
        new_population = []
        
        # Generate enough children to fill up the difference if combined population is less than 2*pop_size
        num_children_to_generate = pop_size 

        for _ in range(num_children_to_generate):
            # 1. Selection
            p1, p2 = self.selection(population)
            
            # 2. Crossover
            child = self.crossover(p1, p2)
            
            # 3. Mutation (Influenced by General Belief and Rate)
            if random.random() < mutation_rate:
                self.mutation(child, general_belief)
            
            # 4. Evaluate
            self.calculate_fitness(child)
            new_population.append(child)
            
        return new_population


# --- 4. Cultural Class (The Orchestrator and Metric Tracker) ---

class Cultural:
# ... (rest of Cultural class remains unchanged)
    """
    Manages the overall CA flow, including EP, IP, and metrics tracking.
    Parameters (for tuning) are set here.
    """
    def __init__(self, pop_size=100, max_stagnation_tries=50, max_k=10, 
                mutation_rate=0.1, mutation_increase_factor=2.0, graph_path="data\\sample_graphs\\graph_one"):
        
        # Parameters
        self.pop_size = pop_size
        self.max_stagnation_tries = max_stagnation_tries # S from the paper's stopping criteria
        self.initial_mutation_rate = mutation_rate
        
        # Handlers
        self.pop_space = PopulationSpace(pop_size, graph_path)
        self.belief_space = BeliefSpace(
            initial_upper_bound=self.pop_space.n_nodes, # Initial high safe value
            max_k=max_k,
            mutation_increase_factor=mutation_increase_factor
        )
        self.population = []

    def run_ca(self):
        """Runs the entire Cultural Algorithm process and measures metrics."""
        start_time = time.time()
        
        # 1. ESTIMATION PHASE (EP)
        initial_bound, initial_pop = self.pop_space.run_estimation_phase()
        self.belief_space.general_belief = initial_bound # Set B from Step 2
        self.population = initial_pop
        
        # Initial status print
        best_initial = min(self.population, key=lambda x: x.fitness)
        print("--- CA Initialization ---")
        print(f"Graph Nodes: {self.pop_space.n_nodes}")
        print(f"Initial Pop Size: {self.pop_size}, General Belief (EP): {self.belief_space.general_belief}")
        print(f"Initial Best Fitness: {best_initial.fitness}")
        
        # 2. IMPROVEMENT PHASE (IP)
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


    def run_improvement_phase(self):
        
        T = 0 # T: Current number of consecutive unsuccessful tries (Step 3/5)
        S = self.max_stagnation_tries # S: Maximum number of tries (Stopping Criteria)
        
        # IP Loop: Continue until T >= S
        while T < S:
            
            # 4. Update Belief Space (Accept) - General Belief
            # Evaluation ensures fitness is current for all individuals
            best_of_gen = self.pop_space.evaluate_and_get_best(self.population)
            old_general_belief = self.belief_space.general_belief
            
            # Update General Belief and check if a new, better color count was found
            # (Implicitly handles part of Step 4)
            belief_changed = self.belief_space.update_general_belief(best_of_gen)
            
            # 5. Check fitness progress, run self-adaptive mechanism, and update T
            # This handles partitioning, r_g, K_g injection, and stagnation counting
            group_improved = self.belief_space.process_groups(self.population, self.pop_space)
            
            if not group_improved:
                T += 1 # Advance T if no progress was made in any group
            else:
                T = 0 # Reset T if progress was made

            # 6. Influence Population (Restriction)
            if belief_changed:
                # Restriction: Remove individuals from groups with belief > new general belief
                self.population = [ind for ind in self.population if ind.belief <= self.belief_space.general_belief]
                print(f"  > Gen {T}: General Belief reduced to {self.belief_space.general_belief}. Population restricted.")
            
            # 7. Perform Reproduction and Mutation
            # Get the current, possibly adapted, mutation rate
            current_mutation_rate = self.belief_space.get_mutation_rate(self.initial_mutation_rate)
            
            # Generate new population
            new_individuals = self.pop_space.perform_variation(
                self.population, 
                self.pop_size, 
                self.belief_space.general_belief,
                current_mutation_rate
            )
            
            # Selection: Merge current and new, then truncate back to pop_size
            self.population.extend(new_individuals)
            self.population.sort(key=lambda x: x.fitness)
            self.population = self.population[:self.pop_size]

            # Monitoring
            if T % 10 == 0 and T > 0:
                current_best_fit = self.population[0].fitness
                current_best_belief = self.population[0].belief
                print(f"  > T={T} (Stagnant Gens). Conflicts={current_best_fit}, Colors={current_best_belief}")

            # 3. Stopping Criteria Check is handled by the while T < S loop condition

# --- Execution ---
if __name__ == '__main__':
    # Parameter Tuning Example: Define a configuration to test
    config = {
        "pop_size": 100,
        "max_stagnation_tries": 50, # Algorithm stops after 50 consecutive generations with no group improvement
        "max_k": 10,                 # Max individuals injected (K)
        "mutation_rate": 0.05,       # Initial P_m
        "mutation_increase_factor": 3.0 # Factor to boost P_m during stagnation
    }
    
    ca_solver = Cultural(**config) 
    ca_solver.run_ca()