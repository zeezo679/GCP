class Backtracking:
    def __init__(self, graph, colors_list, analytics=None):
        self.graph = graph
        self.colors = colors_list
        self.nodes = list(graph.keys())
        self.solution = {}
        self.analytics = analytics

    # ! Standard Backtracking
    def is_safe(self, node, color):
        for neighbor in self.graph.get(node, []):
            if neighbor in self.solution and self.solution[neighbor] == color:
                return False
        return True

    def solve_standard(self, node_index=0):
        if self.analytics:
          self.analytics.increment_visited_nodes()
        
        if node_index == len(self.nodes):
          return True
        
        current_node = self.nodes[node_index]
        
        #! Brute-force approach within backtracking
        for color in self.colors:
            if self.is_safe(current_node, color):
                self.solution[current_node] = color
                if self.solve_standard(node_index + 1):
                  return True
                
                if self.analytics: 
                  self.analytics.increment_backtracks()
                
                del self.solution[current_node]

        return False

    #! Optimized Backtracking
    
    def get_valid_colors(self, node):
        neighbors = self.graph.get(node, [])
        forbidden = {self.solution[n] for n in neighbors if n in self.solution}
        #? Return only colors that are not forbidden
        return [c for c in self.colors if c not in forbidden]

    def solve_optimized(self, node_index=0):
        if self.analytics: 
            self.analytics.increment_visited_nodes()
        
        if node_index == len(self.nodes):
          return True
        
        current_node = self.nodes[node_index]
        
        #? Domain Reduction
        valid_colors = self.get_valid_colors(current_node)
        
        for color in valid_colors:
            self.solution[current_node] = color
            if self.solve_optimized(node_index + 1):
              return True
            
            if self.analytics:
              self.analytics.increment_backtracks()
              
            del self.solution[current_node]

        return False

    #! Entry Points
    def start_standard_solve(self):
        if self.analytics:
          self.analytics.start_timer()
        self.solution = {}

        result = self.solve_standard(0)
        if self.analytics:
          self.analytics.stop_timer(success=result)

        return self.solution if result else None

    def start_optimal_solve(self):
        if self.analytics:
            self.analytics.start_timer()

        original_colors = self.colors.copy()
        
        for k in range(1, len(original_colors) + 1):
            self.colors = original_colors[:k]
            self.solution = {}

            if self.solve_optimized(0):
                if self.analytics:
                    self.analytics.stop_timer(success=True)
                return k, self.solution
        
        if self.analytics:
            self.analytics.stop_timer(success=False)
            
        return -1, None