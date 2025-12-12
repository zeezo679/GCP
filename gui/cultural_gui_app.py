"""
Cultural Algorithm GUI for Graph Coloring Problem
Visualizes the algorithm execution with real-time updates
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
from pathlib import Path
import threading
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx

from algorithms.cultural.cultural_algorithm import CulturalAlgorithm
from utils.graph_generator import GraphGenerator


class CulturalAlgorithmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Graph Coloring - Cultural Algorithm Visualizer")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1e293b")
        
        # Algorithm state
        self.ca = None
        self.graph_path = "data/sample_graphs/graph_two"
        self.is_running = False
        self.generation_data = []
        self.current_generation = 0
        self.is_large_graph = False  # Track if graph is too large for detailed viz
        
        # Colors for nodes
        self.node_colors = [
            '#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6',
            '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
            '#14b8a6', '#a855f7', '#f43f5e', '#22d3ee', '#facc15'
        ]
        
        self.setup_ui()
        self.load_default_graph()
        
    def setup_ui(self):
        # Main container with padding
        main_frame = tk.Frame(self.root, bg="#1e293b")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls and Parameters
        left_panel = tk.Frame(main_frame, bg="#334155", relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.config(width=350)
        
        self.setup_control_panel(left_panel)
        
        # Right panel - Visualization
        right_panel = tk.Frame(main_frame, bg="#1e293b")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.setup_visualization_panel(right_panel)
        
    def setup_control_panel(self, parent):
        # Title
        title_label = tk.Label(parent, text="🎨 Algorithm Controls", 
                               font=("Arial", 16, "bold"),
                               bg="#334155", fg="#f1f5f9")
        title_label.pack(pady=15)
        
        # Graph Selection
        graph_frame = tk.LabelFrame(parent, text="Graph Selection", 
                                    bg="#334155", fg="#f1f5f9",
                                    font=("Arial", 10, "bold"))
        graph_frame.pack(fill=tk.X, padx=15, pady=10)
        
        self.graph_var = tk.StringVar(value="graph_two")
        graphs = ["graph_one", "graph_two", "graph_three", "graph_four", "graph_five"]
        
        for graph in graphs:
            rb = tk.Radiobutton(graph_frame, text=graph.replace("_", " ").title(),
                               variable=self.graph_var, value=graph,
                               bg="#334155", fg="#f1f5f9", 
                               selectcolor="#1e293b",
                               font=("Arial", 9),
                               command=self.on_graph_change)
            rb.pack(anchor=tk.W, padx=10, pady=3)
        
        # Parameters Frame
        params_frame = tk.LabelFrame(parent, text="Algorithm Parameters",
                                     bg="#334155", fg="#f1f5f9",
                                     font=("Arial", 10, "bold"))
        params_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Population Size
        self.create_parameter_slider(params_frame, "Population Size:", 
                                     10, 500, 100, self.on_pop_size_change)
        
        # Mutation Rate
        self.create_parameter_slider(params_frame, "Mutation Rate:", 
                                     0.01, 1.0, 0.1, self.on_mutation_rate_change,
                                     resolution=0.01)
        
        # Max Stagnation
        self.create_parameter_slider(params_frame, "Max Stagnation:", 
                                     10, 200, 50, self.on_stagnation_change)
        
        # Max Colors
        self.create_parameter_slider(params_frame, "Max Colors (k):", 
                                     2, 20, 10, self.on_max_k_change)
        
        # Action Buttons
        button_frame = tk.Frame(parent, bg="#334155")
        button_frame.pack(fill=tk.X, padx=15, pady=15)
        
        self.run_button = tk.Button(button_frame, text="▶ Run Algorithm",
                                    command=self.run_algorithm,
                                    bg="#6366f1", fg="white",
                                    font=("Arial", 11, "bold"),
                                    relief=tk.FLAT, cursor="hand2",
                                    height=2)
        self.run_button.pack(fill=tk.X, pady=5)
        
        self.reset_button = tk.Button(button_frame, text="⟲ Reset",
                                      command=self.reset,
                                      bg="#475569", fg="white",
                                      font=("Arial", 11, "bold"),
                                      relief=tk.FLAT, cursor="hand2",
                                      height=2)
        self.reset_button.pack(fill=tk.X, pady=5)
        
        # Statistics Frame
        stats_frame = tk.LabelFrame(parent, text="Statistics",
                                   bg="#334155", fg="#f1f5f9",
                                   font=("Arial", 10, "bold"))
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.stats_labels = {}
        stats = [
            ("Nodes:", "nodes"),
            ("Edges:", "edges"),
            ("Generation:", "generation"),
            ("Best Fitness:", "fitness"),
            ("Colors Used:", "colors"),
            ("Stagnation:", "stagnation"),
            ("Time Elapsed:", "time"),
            ("Status:", "status")
        ]
        
        for i, (label, key) in enumerate(stats):
            lbl = tk.Label(stats_frame, text=label, 
                          bg="#334155", fg="#94a3b8",
                          font=("Arial", 9, "bold"), anchor=tk.W)
            lbl.grid(row=i, column=0, sticky=tk.W, padx=10, pady=5)
            
            value_lbl = tk.Label(stats_frame, text="-",
                                bg="#334155", fg="#f1f5f9",
                                font=("Arial", 9), anchor=tk.E)
            value_lbl.grid(row=i, column=1, sticky=tk.E, padx=10, pady=5)
            self.stats_labels[key] = value_lbl
        
    def create_parameter_slider(self, parent, label, from_, to, default, command, resolution=1):
        frame = tk.Frame(parent, bg="#334155")
        frame.pack(fill=tk.X, padx=10, pady=8)
        
        lbl = tk.Label(frame, text=label, bg="#334155", fg="#f1f5f9",
                      font=("Arial", 9))
        lbl.pack(side=tk.LEFT)
        
        value_lbl = tk.Label(frame, text=str(default), bg="#334155", fg="#6366f1",
                            font=("Arial", 9, "bold"), width=6)
        value_lbl.pack(side=tk.RIGHT)
        
        slider = tk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL,
                         bg="#334155", fg="#f1f5f9", troughcolor="#1e293b",
                         highlightthickness=0, resolution=resolution,
                         command=lambda v: [value_lbl.config(text=str(float(v) if resolution < 1 else int(float(v)))), command(v)])
        slider.set(default)
        slider.pack(fill=tk.X, padx=(0, 10))
        
        return slider
    
    def setup_visualization_panel(self, parent):
        # Top section - Graph Visualization
        graph_frame = tk.LabelFrame(parent, text="Graph Visualization",
                                   bg="#334155", fg="#f1f5f9",
                                   font=("Arial", 12, "bold"))
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create matplotlib figure for graph
        self.graph_fig = Figure(figsize=(8, 6), facecolor="#1e293b")
        self.graph_ax = self.graph_fig.add_subplot(111)
        self.graph_ax.set_facecolor("#0f172a")
        
        self.graph_canvas = FigureCanvasTkAgg(self.graph_fig, graph_frame)
        self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom section - Fitness Chart
        chart_frame = tk.LabelFrame(parent, text="Fitness Evolution",
                                   bg="#334155", fg="#f1f5f9",
                                   font=("Arial", 12, "bold"))
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure for fitness chart
        self.chart_fig = Figure(figsize=(8, 3), facecolor="#1e293b")
        self.chart_ax = self.chart_fig.add_subplot(111)
        self.chart_ax.set_facecolor("#0f172a")
        self.chart_ax.set_xlabel("Generation", color="#94a3b8")
        self.chart_ax.set_ylabel("Fitness (Conflicts)", color="#94a3b8")
        self.chart_ax.tick_params(colors="#94a3b8")
        self.chart_ax.spines['bottom'].set_color("#334155")
        self.chart_ax.spines['left'].set_color("#334155")
        self.chart_ax.spines['top'].set_visible(False)
        self.chart_ax.spines['right'].set_visible(False)
        
        self.chart_canvas = FigureCanvasTkAgg(self.chart_fig, chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def load_default_graph(self):
        self.load_graph(self.graph_path)
        
    def load_graph(self, path):
        try:
            self.graph_gen = GraphGenerator(path)
            
            # Update stats
            self.stats_labels["nodes"].config(text=str(self.graph_gen.n_nodes))
            
            # Count edges
            edges = []
            seen = set()
            for node, neighbors in self.graph_gen.graph.items():
                for neighbor in neighbors:
                    edge = tuple(sorted([node, neighbor]))
                    if edge not in seen:
                        edges.append(edge)
                        seen.add(edge)
            
            self.stats_labels["edges"].config(text=str(len(edges)))
            
            # Visualize initial graph
            self.visualize_graph()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load graph: {str(e)}")
    
    def visualize_graph(self, chromosome=None):
        self.graph_ax.clear()
        self.graph_ax.set_facecolor("#0f172a")
        self.graph_ax.axis('off')
        
        # Create NetworkX graph
        G = nx.Graph()
        for node, neighbors in self.graph_gen.graph.items():
            for neighbor in neighbors:
                if node < neighbor:  # Avoid duplicate edges
                    G.add_edge(node, neighbor)
        
        # Performance optimization: Use faster layout for large graphs
        num_nodes = len(G.nodes())
        self.is_large_graph = num_nodes > 100
        
        if self.is_large_graph:
            # Use faster circular layout for large graphs
            pos = nx.circular_layout(G)
        else:
            # Use spring layout with reduced iterations for better performance
            pos = nx.spring_layout(G, seed=42, k=1, iterations=20)
        
        # Determine node colors
        if chromosome:
            node_colors_list = []
            # Create a mapping from graph nodes to chromosome indices
            node_list = sorted(G.nodes())
            node_to_index = {node: i for i, node in enumerate(node_list)}
            
            for node in G.nodes():
                # Use the mapping to get the correct chromosome index
                idx = node_to_index[node]
                if idx < len(chromosome):
                    color_idx = chromosome[idx] - 1
                    node_colors_list.append(self.node_colors[color_idx % len(self.node_colors)])
                else:
                    node_colors_list.append("#334155")
        else:
            node_colors_list = ["#334155"] * len(G.nodes())
        
        # Determine edge colors (red if conflict)
        edge_colors = []
        edge_widths = []
        if chromosome:
            node_list = sorted(G.nodes())
            node_to_index = {node: i for i, node in enumerate(node_list)}
            
            for u, v in G.edges():
                u_idx = node_to_index[u]
                v_idx = node_to_index[v]
                if u_idx < len(chromosome) and v_idx < len(chromosome):
                    if chromosome[u_idx] == chromosome[v_idx]:
                        edge_colors.append("#ef4444")
                        edge_widths.append(3)
                    else:
                        edge_colors.append("#475569")
                        edge_widths.append(1)
                else:
                    edge_colors.append("#475569")
                    edge_widths.append(1)
        else:
            edge_colors = ["#475569"] * len(G.edges())
            edge_widths = [1] * len(G.edges())
        
        # Draw with performance optimization
        if self.is_large_graph:
            # Simplified rendering for large graphs
            nx.draw_networkx_edges(G, pos, ax=self.graph_ax, 
                                  edge_color=edge_colors, width=0.5,
                                  alpha=0.3)
            nx.draw_networkx_nodes(G, pos, ax=self.graph_ax,
                                  node_color=node_colors_list, 
                                  node_size=50, edgecolors="#1e293b",
                                  linewidths=0.5)
            # No labels for large graphs - too cluttered
        else:
            # Full detail for small graphs
            nx.draw_networkx_edges(G, pos, ax=self.graph_ax, 
                                  edge_color=edge_colors, width=edge_widths,
                                  alpha=0.6)
            nx.draw_networkx_nodes(G, pos, ax=self.graph_ax,
                                  node_color=node_colors_list, 
                                  node_size=500, edgecolors="#1e293b",
                                  linewidths=2)
            nx.draw_networkx_labels(G, pos, ax=self.graph_ax,
                                   font_color="white", font_weight="bold")
        
        self.graph_canvas.draw()
    
    def update_fitness_chart(self):
        self.chart_ax.clear()
        self.chart_ax.set_facecolor("#0f172a")
        self.chart_ax.set_xlabel("Generation", color="#94a3b8")
        self.chart_ax.set_ylabel("Fitness (Conflicts)", color="#94a3b8")
        self.chart_ax.tick_params(colors="#94a3b8")
        self.chart_ax.spines['bottom'].set_color("#334155")
        self.chart_ax.spines['left'].set_color("#334155")
        self.chart_ax.spines['top'].set_visible(False)
        self.chart_ax.spines['right'].set_visible(False)
        
        if self.generation_data:
            generations = [d['generation'] for d in self.generation_data]
            fitness_values = [d['fitness'] for d in self.generation_data]
            
            self.chart_ax.plot(generations, fitness_values, 
                             color="#6366f1", linewidth=2, marker='o',
                             markersize=4)
            self.chart_ax.fill_between(generations, fitness_values, 
                                      alpha=0.3, color="#6366f1")
            self.chart_ax.grid(True, alpha=0.2, color="#334155")
        
        self.chart_canvas.draw()
        self.chart_canvas.flush_events()  # Force immediate update
        self.root.update_idletasks()  # Process pending GUI events
    
    def run_algorithm(self):
        if self.is_running:
            messagebox.showinfo("Info", "Algorithm is already running")
            return
        
        self.is_running = True
        self.run_button.config(state=tk.DISABLED, bg="#475569")
        self.generation_data = []
        self.current_generation = 0
        
        # Run in separate thread
        thread = threading.Thread(target=self.algorithm_thread, daemon=True)
        thread.start()
    
    def algorithm_thread(self):
        try:
            # Get parameters
            pop_size = int(self.pop_size_value)
            mutation_rate = float(self.mutation_rate_value)
            max_stagnation = int(self.max_stagnation_value)
            max_k = int(self.max_k_value)
            
            # Initialize CA with callback
            self.ca = CulturalAlgorithmWithTracking(
                pop_size=pop_size,
                max_stagnation_tries=max_stagnation,
                max_k=max_k,
                mutation_rate=mutation_rate,
                graph_path=self.graph_path,
                callback=self.generation_callback
            )
            
            # Run algorithm
            start_time = time.time()
            result = self.ca.run_ca()
            end_time = time.time()
            
            # Final update
            self.root.after(0, lambda: self.update_stats(
                generation=len(self.generation_data),
                fitness=result['best_fitness'],
                colors=result['colors_used'],
                status="Success ✓" if result['best_fitness'] == 0 else "Completed",
                time_elapsed=end_time - start_time
            ))
            
            self.root.after(0, lambda: messagebox.showinfo(
                "Completed", 
                f"Algorithm completed!\nFitness: {result['best_fitness']}\n"
                f"Colors: {result['colors_used']}\nTime: {end_time - start_time:.2f}s"
            ))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.run_button.config(state=tk.NORMAL, bg="#6366f1"))
    
    def generation_callback(self, gen, fitness, chromosome, colors, stagnation):
        """Called after each generation"""
        self.generation_data.append({
            'generation': gen,
            'fitness': fitness,
            'chromosome': chromosome,
            'colors': colors,
            'stagnation': stagnation
        })
        
        # Update UI in main thread
        self.root.after(0, lambda: self.update_generation_ui(gen, fitness, chromosome, colors, stagnation))
    
    def update_generation_ui(self, gen, fitness, chromosome, colors, stagnation):
        # Update stats
        self.stats_labels["generation"].config(text=str(gen))
        self.stats_labels["fitness"].config(text=str(fitness))
        self.stats_labels["colors"].config(text=str(colors))
        self.stats_labels["stagnation"].config(text=str(stagnation))
        self.stats_labels["status"].config(text="Running...")
        
        # Update visualizations with throttling for large graphs
        if self.is_large_graph:
            # Only update graph viz every 5 generations for large graphs
            if gen % 5 == 0 or fitness == 0:
                self.visualize_graph(chromosome)
        else:
            # Update graph every generation for small graphs
            self.visualize_graph(chromosome)
        
        # Always update fitness chart (lightweight operation)
        self.update_fitness_chart()
    
    def update_stats(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.stats_labels:
                if key == "time_elapsed":
                    self.stats_labels["time"].config(text=f"{value:.2f}s")
                else:
                    self.stats_labels[key].config(text=str(value))
    
    def reset(self):
        self.generation_data = []
        self.current_generation = 0
        
        # Reset stats
        self.stats_labels["generation"].config(text="0")
        self.stats_labels["fitness"].config(text="-")
        self.stats_labels["colors"].config(text="-")
        self.stats_labels["stagnation"].config(text="0")
        self.stats_labels["time"].config(text="-")
        self.stats_labels["status"].config(text="Ready")
        
        # Reset visualizations
        self.visualize_graph()
        self.update_fitness_chart()
    
    # Parameter change handlers
    def on_pop_size_change(self, value):
        self.pop_size_value = int(float(value))
    
    def on_mutation_rate_change(self, value):
        self.mutation_rate_value = float(value)
    
    def on_stagnation_change(self, value):
        self.max_stagnation_value = int(float(value))
    
    def on_max_k_change(self, value):
        self.max_k_value = int(float(value))
    
    def on_graph_change(self):
        graph_name = self.graph_var.get()
        self.graph_path = f"data/sample_graphs/{graph_name}"
        self.load_graph(self.graph_path)
        self.reset()


class CulturalAlgorithmWithTracking(CulturalAlgorithm):
    """Extended CA that calls callback after each generation"""
    
    def __init__(self, pop_size=100, max_stagnation_tries=50, max_k=10,
                 mutation_rate=0.1, mutation_increase_factor=2.0, 
                 graph_path="data\\sample_graphs\\graph_two", callback=None):
        super().__init__(pop_size, max_stagnation_tries, max_k, 
                        mutation_rate, mutation_increase_factor, graph_path)
        self.callback = callback
    
    def run_improvement_phase(self):
        T = 0
        S = self.max_stagnation_tries
        generation = 1

        while T < S:
            best_of_gen = self.pop_space.evaluate_and_get_best(self.population)
            old_general_belief = self.belief_space.general_belief

            belief_changed = self.belief_space.update_belief(best_of_gen)
            group_improved = self.belief_space.process_groups(self.population, self.pop_space)

            if not group_improved:
                T += 1
            else:
                T = 0
            
            if belief_changed:
                self.population = [ind for ind in self.population 
                                 if ind.belief <= self.belief_space.general_belief]

            current_mutation_rate = self.belief_space.get_mutation_rate(
                self.initial_muation_rate, best_of_gen.belief
            )
            
            new_individuals = self.pop_space.perform_variation(
                self.population, self.pop_size,
                self.belief_space.general_belief, current_mutation_rate
            )

            self.population.extend(new_individuals)
            self.population.sort(key=lambda x: x.fitness)
            self.population = self.population[:self.pop_size]

            # Callback with generation data
            if self.callback:
                current_best = self.population[0]
                self.callback(generation, current_best.fitness, 
                            current_best.chromosome, current_best.belief, T)
            
            # Stop if optimal solution found
            if self.population[0].fitness == 0:
                break
            
            generation += 1
            
            # Adaptive sleep - faster for large graphs
            if self.pop_space.n_nodes > 100:
                time.sleep(0.01)  # Minimal delay for large graphs
            else:
                time.sleep(0.1)  # Visible animation for small graphs


def main():
    root = tk.Tk()
    app = CulturalAlgorithmGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
