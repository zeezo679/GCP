# 📘 Graph Coloring Project

This project implements **Graph Coloring** using multiple algorithms, a GUI for visualization, and tools for generating and testing graphs.  
It is structured into clean, modular components to support team collaboration.

---

# 1. 🔧 core/

Contains the core system components used across all algorithms and modules.

### **graph.py**
- Class **Graph**
- Stores nodes, edges, colors, and adjacency information.

### **json_handler.py**
- Class **JSONHandler**
- Handles saving and loading JSON files for algorithm I/O.

### **performance.py**
- Class **PerformanceTracker**
- Tracks execution time, steps, visited nodes, and other performance metrics.

---

# 2. 🧠 algorithms/

Contains all algorithm implementations, each in its own subfolder.

---

## 🔹 backtracking/

### **backtracking_solver.py**
- Class **BacktrackingSolver**
- Implements the classic Backtracking algorithm for graph coloring.
- Returns coloring results in JSON format.

---

## 🔹 cultural/

### **population_space.py**
- Class **PopulationSpace**
- Manages individuals, crossover, mutation, and evolutionary population operations.

### **belief_space.py**
- Class **BeliefSpace**
- Stores and updates global knowledge influencing the population.

### **cultural_algorithm.py**
- Class **CulturalAlgorithm**
- Main controller combining **PopulationSpace** and **BeliefSpace** to evolve optimal colorings.

---

# 3. 🖥️ gui/

Everything related to the Graphical User Interface.

### **main_gui.py**
- Class **MainGUI**
- Main application window for selecting algorithms, uploading graphs, and running tests.

### **gui_backtracking.py**
- Class **BacktrackingWindow**
- Visualizes the Backtracking algorithm step-by-step.

### **gui_cultural.py**
- Class **CulturalWindow**
- Visualizes the Cultural Algorithm’s evolutionary process.

### **visualization.py**
- Class **GraphVisualizer**
- Draws nodes, edges, and colors from JSON results.
- Shared visualizer used by both Backtracking and Cultural GUIs.

---

# 4. 🛠️ utils/

Helper utilities to support development.

### **graph_generator.py**
- Class **GraphGenerator**
- Generates random test graphs of variable sizes and edge densities.

### **file_utils.py**
- Class **FileUtils**
- Handles file reading/writing safely and efficiently.

---

# 5. 📁 data/

Contains input files, sample graphs, and algorithm outputs.

### **sample_graphs/**
- Example graph files for testing.

### **outputs/**
- Stores generated algorithm results.
    - `backtracking_output.json`
    - `cultural_output.json`

### **dummy.json**
- A placeholder JSON file for GUI testing and debugging.

---

# 6. 🧪 tests/

Unit test files for verifying functionality of each module.

### **test_backtracking.py**
- Tests the Backtracking algorithm.

### **test_cultural.py**
- Tests the Cultural Algorithm.

### **test_gui.py**
- Tests GUI components and interactions.

---
