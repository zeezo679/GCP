# Class GraphGenerator:
# Creates random graphs for testing.

from typing import Dict, List
import os 


class GraphGenerator:
    def __init__(self, filename):
        self.graph: Dict[int, List[int]] = {}
        self.n_nodes = 0

        try:
            abs_path = os.path.abspath(filename)

            with open(abs_path, 'r') as file:
                for line in file:
                    cleaned_line = line.strip()

                    u, v = map(int, cleaned_line.split())

                    if u not in self.graph:
                        self.graph[u] = []
                    if v not in self.graph:
                        self.graph[v] = []

                    self.graph[u].append(v)
                    self.graph[v].append(u)

                self.n_nodes = len(self.graph)

            print(f"Successfully generated graph with {self.n_nodes} nodes")

        except FileNotFoundError:
            print(f"File not found: {abs_path}")
        except Exception as e:
            print(f"Error occurred: {e}")
        

    def show(self):
        for node, neighbors in self.graph.items():
            print(f"Node {node:2}: {sorted(neighbors)}")

#Example usage
# random_graph = GraphGenerator("data\\sample_graphs\\graph_two")
# random_graph.show()