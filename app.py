"""
Flask + Socket.IO Web Application for Cultural Algorithm
Real-time visualization of Graph Coloring Problem solving
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import sys
from pathlib import Path
import time
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from algorithms.cultural.cultural_algorithm import CulturalAlgorithm
from algorithms.cultural.population_space import PopulationSpace
from algorithms.cultural.belief_space import BeliefSpace
from utils.graph_generator import GraphGenerator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cultural_algorithm_secret_2025'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
current_simulation = None
simulation_running = False


class CulturalAlgorithmWebSocket(CulturalAlgorithm):
    """
    Extended Cultural Algorithm that emits real-time updates via Socket.IO
    """
    
    def __init__(self, pop_size=100, max_stagnation_tries=50, max_k=10,
                 mutation_rate=0.1, mutation_increase_factor=2.0, 
                 graph_path="data/sample_graphs/graph_two"):
        super().__init__(pop_size, max_stagnation_tries, max_k, 
                        mutation_rate, mutation_increase_factor, graph_path)
        self.generation = 0
        self.should_stop = False
        
    def run_ca_with_events(self):
        """Run CA with real-time Socket.IO events"""
        start_time = time.time()
        
        # Random initialization instead of estimation phase
        self.belief_space.general_belief = self.initial_upper_bound
        self.population = self.pop_space.initialize_population(self.initial_upper_bound)
        
        # Calculate fitness for initial population
        for ind in self.population:
            self.pop_space.calculate_fitness(ind)
        
        best_initial = min(self.population, key=lambda x: x.fitness)
        
        # Emit initialization event
        socketio.emit('generation_update', {
            'generation': 0,
            'best_fitness': best_initial.fitness,
            'general_belief': self.belief_space.general_belief,
            'chromosome': best_initial.chromosome,
            'stagnation': 0,
            'colors_used': best_initial.belief,
            'population_size': len(self.population),
            'status': 'running'
        })
        
        # Small delay to ensure client receives initial state
        socketio.sleep(0.1)
        
        # Run improvement phase with events
        self.run_improvement_phase_with_events()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        final_best = min(self.population, key=lambda x: x.fitness)
        
        # Emit completion event
        socketio.emit('simulation_complete', {
            'best_fitness': final_best.fitness,
            'best_chromosome': final_best.chromosome,
            'colors_used': final_best.belief,
            'execution_time': total_time,
            'total_generations': self.generation,
            'status': 'success' if final_best.fitness == 0 else 'completed'
        })
        
        return {
            'best_fitness': final_best.fitness,
            'best_chromosome': final_best.chromosome,
            'colors_used': final_best.belief,
            'execution_time': total_time
        }
    
    def run_improvement_phase_with_events(self):
        """Improvement phase that emits updates after each generation"""
        T = 0
        S = self.max_stagnation_tries
        self.generation = 1
        
        while T < S and not self.should_stop:
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
            
            current_best = self.population[0]
            
            # Emit generation update via Socket.IO with throttling for large graphs
            # For large graphs, emit less frequently to reduce network overhead
            is_large_graph = self.pop_space.n_nodes > 100
            should_emit = (not is_large_graph) or (self.generation % 3 == 0) or belief_changed or current_best.fitness == 0
            
            if should_emit:
                socketio.emit('generation_update', {
                    'generation': self.generation,
                    'best_fitness': current_best.fitness,
                    'general_belief': self.belief_space.general_belief,
                    'chromosome': current_best.chromosome,
                    'stagnation': T,
                    'colors_used': current_best.belief,
                    'population_size': len(self.population),
                    'belief_changed': belief_changed,
                    'status': 'running'
                })
            
            # Stop if optimal solution found
            if current_best.fitness == 0:
                break
            
            self.generation += 1
            
            # Adaptive sleep for visualization
            if is_large_graph:
                time.sleep(0.01)  # Minimal delay for large graphs
            else:
                time.sleep(0.15)  # Smooth animation for small graphs
    
    def stop(self):
        """Stop the simulation"""
        self.should_stop = True


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/graphs', methods=['GET'])
def get_available_graphs():
    """Return list of available graph files"""
    graphs_dir = Path('data/sample_graphs')
    graphs = []
    
    if graphs_dir.exists():
        for graph_file in graphs_dir.iterdir(): 
            if graph_file.is_file():
                try:
                    gen = GraphGenerator(str(graph_file))
                    graphs.append({
                        'name': graph_file.name,
                        'path': str(graph_file),
                        'nodes': gen.n_nodes,
                        'edges': sum(len(neighbors) for neighbors in gen.graph.values()) // 2
                    })
                except:
                    pass
    
    return jsonify(graphs)


@app.route('/api/graph/<graph_name>', methods=['GET'])
def get_graph_data(graph_name):
    """Return graph structure (nodes and edges) for visualization"""
    try:
        graph_path = f"data/sample_graphs/{graph_name}"
        gen = GraphGenerator(graph_path)
        
        # Get unique edges
        edges = []
        seen = set()
        for node, neighbors in gen.graph.items():
            for neighbor in neighbors:
                edge = tuple(sorted([node, neighbor]))
                if edge not in seen:
                    edges.append({'from': edge[0], 'to': edge[1]})
                    seen.add(edge)
        
        # Get nodes
        nodes = [{'id': node, 'label': str(node)} for node in sorted(gen.graph.keys())]
        
        return jsonify({
            'nodes': nodes,
            'edges': edges,
            'n_nodes': gen.n_nodes
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/save-graph', methods=['POST'])
def save_custom_graph():
    """Save a custom graph from user input"""
    try:
        data = request.get_json()
        edges = data.get('edges', [])
        
        if not edges:
            return jsonify({'error': 'No edges provided'}), 400
        
        # Find next available graph number
        graphs_dir = Path('data/sample_graphs')
        existing_nums = []
        for graph_file in graphs_dir.iterdir():
            if graph_file.is_file() and graph_file.name.startswith('graph_'):
                try:
                    # Extract number from graph_X or graph_name
                    name_parts = graph_file.name.split('_')
                    if len(name_parts) >= 2:
                        num_str = name_parts[-1]
                        if num_str.isdigit():
                            existing_nums.append(int(num_str))
                except:
                    pass
        
        # Get next number (or start at 8 if no numbered graphs found)
        next_num = max(existing_nums) + 1 if existing_nums else 8
        graph_name = f"graph_{next_num}"
        graph_path = graphs_dir / graph_name
        
        # Write edges to file
        with open(graph_path, 'w', encoding='utf-8') as f:
            for u, v in edges:
                f.write(f"{u} {v}\n")
        
        print(f"Saved custom graph as {graph_name} with {len(edges)} edges")
        
        return jsonify({
            'success': True,
            'graph_name': graph_name,
            'num_edges': len(edges)
        })
        
    except Exception as e:
        print(f"Error saving custom graph: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/delete-graph', methods=['DELETE'])
def delete_graph():
    """Delete a graph file"""
    try:
        data = request.get_json()
        graph_name = data.get('graph_name', '')
        
        if not graph_name:
            return jsonify({'error': 'No graph name provided'}), 400
        
        # Prevent deleting default graphs
        default_graphs = ['graph_one', 'graph_two', 'graph_three', 'graph_four', 
                         'graph_five', 'graph_six', 'graph_seven']
        
        if graph_name in default_graphs:
            return jsonify({'error': 'Cannot delete default graphs'}), 403
        
        # Delete the file
        graph_path = Path(f"data/sample_graphs/{graph_name}")
        
        if not graph_path.exists():
            return jsonify({'error': 'Graph file not found'}), 404
        
        graph_path.unlink()
        print(f"Deleted graph: {graph_name}")
        
        return jsonify({
            'success': True,
            'graph_name': graph_name
        })
        
    except Exception as e:
        print(f"Error deleting graph: {e}")
        return jsonify({'error': str(e)}), 400


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    global simulation_running, current_simulation
    print('Client disconnected')
    if current_simulation:
        current_simulation.stop()
    simulation_running = False


@socketio.on('start_simulation')
def handle_start_simulation(data):
    """Start the Cultural Algorithm simulation in background thread"""
    global current_simulation, simulation_running
    
    if simulation_running:
        emit('error', {'message': 'Simulation already running'})
        return
    
    try:
        # Extract parameters
        pop_size = int(data.get('pop_size', 100))
        max_stagnation = int(data.get('max_stagnation', 50))
        mutation_rate = float(data.get('mutation_rate', 0.1))
        max_k = int(data.get('max_k', 10))
        graph_name = data.get('graph_name', 'graph_two')
        fixed_seed = data.get('fixed_seed', False)
        graph_path = f"data/sample_graphs/{graph_name}"
        
        # Set random seed if fixed_seed is enabled
        if fixed_seed:
            import random
            random.seed(42)
            print(f"Using fixed seed (42) for reproducible results")
        
        print(f"Starting simulation with: pop_size={pop_size}, max_stagnation={max_stagnation}, "
              f"mutation_rate={mutation_rate}, max_k={max_k}, graph={graph_name}, fixed_seed={fixed_seed}")
        
        # Create algorithm instance
        current_simulation = CulturalAlgorithmWebSocket(
            pop_size=pop_size,
            max_stagnation_tries=max_stagnation,
            max_k=max_k,
            mutation_rate=mutation_rate,
            graph_path=graph_path
        )
        
        simulation_running = True
        
        # Run in background thread
        def run_simulation():
            global simulation_running
            try:
                current_simulation.run_ca_with_events()
            except Exception as e:
                socketio.emit('error', {'message': str(e)})
                print(f"Simulation error: {e}")
            finally:
                simulation_running = False
        
        socketio.start_background_task(run_simulation)
        
        emit('simulation_started', {'status': 'started'})
        
    except Exception as e:
        emit('error', {'message': str(e)})
        print(f"Error starting simulation: {e}")


@socketio.on('stop_simulation')
def handle_stop_simulation():
    """Stop the running simulation"""
    global current_simulation, simulation_running
    
    if current_simulation:
        current_simulation.stop()
        simulation_running = False
        emit('simulation_stopped', {'status': 'stopped'})
    else:
        emit('error', {'message': 'No simulation running'})


if __name__ == '__main__':
    print("Starting Cultural Algorithm Web Application...")
    print("Open http://localhost:5000 in your browser")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
