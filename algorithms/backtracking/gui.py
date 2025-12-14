import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import math
import random
import os
import sys
import json
import time
# Imports for Charts
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk 

# Import Performance Analytics
try:
    from BackTracking.Performance.PerformanceAnalytics import PerformanceAnalytics
except ImportError:
    print("Analytics module not found")
# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# إنشاء فولدر لحفظ الجرافات لو مش موجود
SAVES_DIR = os.path.join(current_dir, 'SavedGraphs')
os.makedirs(SAVES_DIR, exist_ok=True)

try:
    from BackTracking.Algo.backtracking import Backtracking
    from BackTracking.Helper.helper import Helper
except ImportError:
    pass

# --- Config ---
THEME_BG = "#2C3E50"
CANVAS_BG = "#FDFFE6"
NODE_COLOR = "#FFFFFF"
NODE_BORDER = "#34495E"
ACCENT_COLOR = "#E67E22"

class GraphColoringUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Graph Coloring Solver - Project Manager Edition")
        self.root.geometry("1200x800")

        # --- Data ---
        self.nodes = {} 
        self.edges = [] 
        self.adj_list = {}
        self.node_radius = 20 
        self.node_counter = 0
        self.current_mode = "NODE" 
        self.selected_node = None
        self.available_colors = self.load_real_colors()

        # --- Layout ---
        self.sidebar = tk.Frame(root, bg=THEME_BG, width=320)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        self.canvas = tk.Canvas(root, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.setup_sidebar()
        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<Configure>", self.on_resize)

        # تحميل قائمة الملفات المحفوظة عند البدء
        self.refresh_saved_files()

    def load_real_colors(self):
        try:
            path = os.path.join(current_dir, 'BackTracking', 'Colors', 'color.json')
            with open(path, 'r') as f:
                data = json.load(f)
                return data.get("all_colors", [])
        except:
            return ["Red", "Green", "Blue", "Yellow", "Orange", "Purple"]

    def setup_sidebar(self):
        # 1. Title (Top)
        tk.Label(self.sidebar, text="🎮 CONTROL PANEL", bg=THEME_BG, fg="white", font=("Segoe UI", 14, "bold")).pack(pady=(15, 10))

        # 2. Generator (Compact)
        gen_frame = tk.LabelFrame(self.sidebar, text="⚡ Generator", bg=THEME_BG, fg=ACCENT_COLOR, font=("Segoe UI", 9, "bold"))
        gen_frame.pack(fill=tk.X, padx=10, pady=2)
        
        tk.Label(gen_frame, text="N:", bg=THEME_BG, fg="white").pack(side=tk.LEFT, padx=2)
        self.entry_nodes = tk.Entry(gen_frame, width=3); self.entry_nodes.pack(side=tk.LEFT, padx=2); self.entry_nodes.insert(0, "15")
        
        tk.Label(gen_frame, text="E:", bg=THEME_BG, fg="white").pack(side=tk.LEFT, padx=2)
        self.entry_edges = tk.Entry(gen_frame, width=3); self.entry_edges.pack(side=tk.LEFT, padx=2); self.entry_edges.insert(0, "25")
        
        tk.Button(gen_frame, text="Planets", command=self.generate_random_graph, bg="#27AE60", fg="white", relief="flat", font=("Arial", 8)).pack(side=tk.RIGHT, padx=5, pady=2)

        # 3. Project Manager (Save/Load)
        save_frame = tk.Frame(self.sidebar, bg=THEME_BG)
        save_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(save_frame, text="💾 Save", command=self.save_graph_to_file, bg="#2980B9", fg="white", width=8).pack(side=tk.LEFT, padx=1)
        
        self.combo_files = ttk.Combobox(save_frame, state="readonly", width=12)
        self.combo_files.pack(side=tk.LEFT, padx=1)
        tk.Button(save_frame, text="📂 Load", command=self.load_graph_from_file, bg="#34495E", fg="white", width=6).pack(side=tk.LEFT, padx=1)


        # 4. Tools (Compact)
        tool_frame = tk.Frame(self.sidebar, bg=THEME_BG)
        tool_frame.pack(fill=tk.X, padx=10, pady=5)
        self.btn_node = self.create_button(tool_frame, "📍 Node", "NODE")
        self.btn_edge = self.create_button(tool_frame, "🔗 Link", "EDGE")
        self.btn_del = self.create_button(tool_frame, "🗑️ Del", "DELETE", color="#C0392B")

        # 5. Palette (Compact Grid)
        tk.Label(self.sidebar, text=f"🎨 PALETTE", bg=THEME_BG, fg="#BDC3C7", font=("Arial", 8)).pack(pady=(5, 0))
        palette_frame = tk.Frame(self.sidebar, bg=THEME_BG)
        palette_frame.pack(pady=2, padx=10)
        row, col = 0, 0
        for color_name in self.available_colors:
            try:
                lbl = tk.Label(palette_frame, bg=color_name.lower(), width=2, height=1, relief="ridge", borderwidth=1)
                lbl.grid(row=row, column=col, padx=1, pady=1)
                col += 1
                if col > 7: col = 0; row += 1
            except: pass

        # ----------------------------------------------------
        # 🔥 BOTTOM SECTION (Fixed at bottom) 🔥
        # ----------------------------------------------------
        bottom_frame = tk.Frame(self.sidebar, bg=THEME_BG)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Log (Last item)
        self.log_lbl = tk.Label(bottom_frame, text="Ready.", bg="#34495E", fg="#2ECC71", anchor="w", padx=5)
        self.log_lbl.pack(side=tk.BOTTOM, fill=tk.X)

        # Clear/Reset Buttons (Above Log)
        action_frame = tk.Frame(bottom_frame, bg=THEME_BG)
        action_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        tk.Button(action_frame, text="Reset Colors", command=self.reset_colors, bg="#8E44AD", fg="white", width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(action_frame, text="Clear All", command=self.clear_canvas, bg="#7F8C8D", fg="white", width=12).pack(side=tk.RIGHT, padx=2)

        # ----------------------------------------------------
        # 🔥 SOLVING SECTION (Between Palette and Bottom) 🔥
        # ----------------------------------------------------
        solve_frame = tk.Frame(self.sidebar, bg=THEME_BG)
        solve_frame.pack(fill=tk.X, padx=10, pady=10) # Pack normally to take remaining space

        tk.Frame(solve_frame, height=2, bg="grey").pack(fill=tk.X, pady=5)
        
        # Mode 1
        tk.Label(solve_frame, text="MODE 1: Standard", bg=THEME_BG, fg="#BDC3C7", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.limit_scale = tk.Scale(solve_frame, from_=1, to=len(self.available_colors), orient=tk.HORIZONTAL, bg=THEME_BG, fg="white", highlightthickness=0)
        self.limit_scale.set(4)
        self.limit_scale.pack(fill=tk.X)
        tk.Button(solve_frame, text="⚡ Run Standard", command=self.run_standard_mode, bg="#E67E22", fg="white", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=2)

        # Mode 2
        tk.Frame(solve_frame, height=1, bg="grey").pack(fill=tk.X, pady=5)
        tk.Label(solve_frame, text="MODE 2: Scientific", bg=THEME_BG, fg="#BDC3C7", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Button(solve_frame, text="🏆 Find Min Colors", command=self.run_optimized_mode, bg="#2980B9", fg="white", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=2)

    def create_button(self, parent, text, mode, color="#34495E"):
        btn = tk.Button(parent, text=text, bg=color, fg="white", relief="flat", padx=5, pady=2, font=("Arial", 9),
                        command=lambda: self.set_mode(mode, btn))
        btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        if mode == "NODE": self.active_btn = btn 
        return btn

    def refresh_saved_files(self):
        """ تحديث القائمة بأسماء الملفات الموجودة """
        files = [f for f in os.listdir(SAVES_DIR) if f.endswith('.json')]
        self.combo_files['values'] = files
        if files:
            self.combo_files.current(0) # اختار اول واحد اوتوماتيك

    def save_graph_to_file(self):
        """ حفظ الجراف الحالي في ملف JSON """
        if not self.nodes:
            messagebox.showwarning("Save Error", "Canvas is empty!")
            return

        # نطلب اسم الملف من اليوزر
        filename = simpledialog.askstring("Save Graph", "Enter graph name (without .json):")
        if not filename: return # لو داس Cancel

        # تجهيز الداتا للحفظ
        save_data = {
            "nodes": {},
            "adj_list": self.adj_list
        }
        
        # لازم نحفظ الاحداثيات (x, y) عشان لما نرجعه يترسم في نفس المكان
        for nid, data in self.nodes.items():
            save_data["nodes"][nid] = {
                "x": data['x'],
                "y": data['y']
            }

        try:
            full_path = os.path.join(SAVES_DIR, f"{filename}.json")
            with open(full_path, 'w') as f:
                json.dump(save_data, f, indent=4)
            
            messagebox.showinfo("Success", f"Graph saved as {filename}.json")
            self.refresh_saved_files() # تحديث القائمة
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def load_graph_from_file(self):
        """ استرجاع الجراف من الملف ورسمه """
        selected_file = self.combo_files.get()
        if not selected_file: return

        try:
            full_path = os.path.join(SAVES_DIR, selected_file)
            with open(full_path, 'r') as f:
                data = json.load(f)

            self.clear_canvas() # امسح القديم الاول

            # 1. Recreate Nodes
            saved_nodes = data.get("nodes", {})
            for nid_str, coords in saved_nodes.items():
                x, y = coords['x'], coords['y']
                # بنستخدم الدالة القديمة بس بنعدل ال node_counter عشان يظبط ال ID
                # (تريكه بسيطة: بنرسم ونعدل ال ID يدويا لضمان التطابق)
                self.add_node_from_load(int(nid_str), x, y)
            
            # تحديث العداد عشان لو ضاف نودز جديدة مياخدوش ID موجود
            if saved_nodes:
                self.node_counter = max([int(k) for k in saved_nodes.keys()]) + 1
            else:
                self.node_counter = 0

            # 2. Recreate Edges
            # ال adj_list المحفوظة ممكن يكون فيها تكرار (undirected)، فلازم ناخد بالنا
            saved_adj = data.get("adj_list", {})
            for u_str, neighbors in saved_adj.items():
                u = int(u_str)
                for v_str in neighbors:
                    v = int(v_str)
                    # عشان منرسمش الخط مرتين، نرسم بس لو u < v
                    if u < v: 
                        # نتأكد ان النودز موجودة
                        if u in self.nodes and v in self.nodes:
                            self.add_edge_visual(u, v)

            self.log_status(f"Loaded {selected_file}")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load file: {e}")

    def add_node_from_load(self, nid, x, y):
        """ دالة مساعدة خاصة بال Load عشان تحافظ على ال IDs القديمة """
        r = 20 if len(self.nodes) < 25 else 15
        oval = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=NODE_COLOR, outline=NODE_BORDER, width=2)
        font_size = 10 if r >= 20 else 8
        text = self.canvas.create_text(x, y, text=str(nid), font=("Arial", font_size, "bold"))
        
        self.nodes[nid] = {'x': x, 'y': y, 'oval': oval, 'text': text}
        self.adj_list[str(nid)] = []

    def generate_random_graph(self):
        self.clear_canvas()
        try:
            total_n = int(self.entry_nodes.get())
            total_e = int(self.entry_edges.get())
        except: return
        
        main_cluster_size = int(total_n * 0.6)
        side_cluster_1_size = int(total_n * 0.2)
        side_cluster_2_size = total_n - main_cluster_size - side_cluster_1_size
        clusters = [main_cluster_size, side_cluster_1_size, side_cluster_2_size]
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        padding = 100
        safe_w = w - (2 * padding)
        safe_h = h - (2 * padding)
        min_dim = min(safe_w, safe_h)

        main_radius = min_dim * 0.35 
        center_main = (w//2, h//2, main_radius)
        side_radius = min_dim * 0.15
        center_side1 = (padding + side_radius, padding + side_radius, side_radius)
        center_side2 = (w - padding - side_radius, h - padding - side_radius, side_radius)
        centers = [center_main, center_side1, center_side2]
        
        cluster_nodes_ids = []
        for i, size in enumerate(clusters):
            cx, cy, radius = centers[i]
            current_ids = []
            if size == 0: continue
            for j in range(size):
                angle = 2 * math.pi * j / size
                spread = random.uniform(0.85, 1.15) 
                nx = cx + (radius * spread) * math.cos(angle)
                ny = cy + (radius * spread) * math.sin(angle)
                nx = max(20, min(w-20, nx))
                ny = max(20, min(h-20, ny))
                self.add_node_visual(nx, ny)
                current_ids.append(self.node_counter - 1)
            cluster_nodes_ids.append(current_ids)
        
        edges_per_cluster = [int(total_e * 0.7), int(total_e * 0.15), int(total_e * 0.15)]
        for i, cluster_ids in enumerate(cluster_nodes_ids):
            if len(cluster_ids) < 2: continue
            target_edges = min(edges_per_cluster[i], len(cluster_ids)*(len(cluster_ids)-1)//2)
            count = 0; attempts = 0
            while count < target_edges and attempts < 2000:
                u, v = random.sample(cluster_ids, 2)
                if str(v) not in self.adj_list.get(str(u), []):
                    self.add_edge_visual(u, v)
                    count += 1
                attempts += 1
        self.log_status(f"Generated Planetary Graph ({total_n} Nodes)")

    def animate_coloring(self, solution):
        total_nodes = len(solution)
        used_colors = set()
        delay = 0.2
        for node_str_id, color_name in solution.items():
            try:
                node_id = int(node_str_id)
                if node_id in self.nodes:
                    oval = self.nodes[node_id]['oval']
                    self.canvas.itemconfig(oval, fill=color_name.lower(), width=3) 
                    used_colors.add(color_name)
                    self.root.update()
                    time.sleep(delay) 
            except: pass
        messagebox.showinfo("Done", f"Coloring Finished! 🎨\nUsed {len(used_colors)} colors.")

    def add_node_visual(self, x, y):
        node_id = self.node_counter
        self.node_counter += 1
        r = 20 if self.node_counter < 30 else 15
        oval = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=NODE_COLOR, outline=NODE_BORDER, width=2)
        font_size = 10 if r >= 20 else 8
        text = self.canvas.create_text(x, y, text=str(node_id), font=("Arial", font_size, "bold"))
        self.nodes[node_id] = {'x': x, 'y': y, 'oval': oval, 'text': text}
        self.adj_list[str(node_id)] = []

    def add_edge_visual(self, u, v):
        u_str, v_str = str(u), str(v)
        if u == v: return
        x1, y1 = self.nodes[u]['x'], self.nodes[u]['y']
        x2, y2 = self.nodes[v]['x'], self.nodes[v]['y']
        line = self.canvas.create_line(x1, y1, x2, y2, fill="#95A5A6", width=2)
        self.canvas.tag_lower(line)
        self.edges.append((u, v, line))
        if u_str not in self.adj_list: self.adj_list[u_str] = []
        if v_str not in self.adj_list: self.adj_list[v_str] = []
        if v_str not in self.adj_list[u_str]: self.adj_list[u_str].append(v_str)
        if u_str not in self.adj_list[v_str]: self.adj_list[v_str].append(u_str)

    def delete_node(self, nid):
        self.canvas.delete(self.nodes[nid]['oval'])
        self.canvas.delete(self.nodes[nid]['text'])
        del self.nodes[nid]
        if str(nid) in self.adj_list: del self.adj_list[str(nid)]
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.nodes = {}
        self.edges = []
        self.adj_list = {}
        self.node_counter = 0

    def reset_colors(self):
        for nid, data in self.nodes.items():
            self.canvas.itemconfig(data['oval'], fill=NODE_COLOR, width=2)

    def get_clicked_node(self, x, y):
        for nid, data in self.nodes.items():
            if math.hypot(x-data['x'], y-data['y']) <= 25: return nid
        return None

    def handle_click(self, event):
        x, y = event.x, event.y
        clicked = self.get_clicked_node(x, y)
        if self.current_mode == "NODE" and clicked is None: self.add_node_visual(x, y)
        elif self.current_mode == "EDGE" and clicked is not None:
            if self.selected_node is None:
                self.selected_node = clicked
                self.canvas.itemconfig(self.nodes[clicked]['oval'], outline=ACCENT_COLOR, width=3)
            else:
                self.add_edge_visual(self.selected_node, clicked)
                self.canvas.itemconfig(self.nodes[self.selected_node]['oval'], outline=NODE_BORDER, width=2)
                self.selected_node = None
        elif self.current_mode == "DELETE" and clicked is not None:
            self.delete_node(clicked)

    def solve_graph_action(self):
        if not self.nodes: return
        self.reset_colors() 
        
        graph_data = self.adj_list.copy()
        for nid in self.nodes:
            if str(nid) not in graph_data: graph_data[str(nid)] = []

        limit = self.limit_scale.get()
        selected_colors = self.available_colors[:limit]
        
        # Check Connectivity
        if not Helper.is_graph_connected(graph_data):
            messagebox.showinfo("Note", "Disconnected Graph Detected!")

        # Analytics
        analytics = PerformanceAnalytics()
        
        # Create Solver
        solver = Backtracking(graph_data, selected_colors, analytics=analytics)
        
        solution = solver.start_standard_solve()

        if solution:
            self.log_status("Standard Solution Found! Animating...")
            self.animate_coloring(solution)
            self.show_dashboard(analytics) # اعرض الداشبورد
        else:
            messagebox.showerror("Failed", f"Standard Backtracking failed with {limit} colors!")
            self.show_dashboard(analytics)
    
    def find_min_colors_action(self):
        if not self.nodes: return
        self.reset_colors()
        
        graph_data = self.adj_list.copy()
        for nid in self.nodes:
            if str(nid) not in graph_data: graph_data[str(nid)] = []
            
        all_colors = self.available_colors
        
        # Analytics عشان نحسب التكلفة الكلية
        analytics = PerformanceAnalytics()
        solver = Backtracking(graph_data, all_colors, analytics=analytics)
        
        # 🔥 التعديل هنا: نداء الدالة الـ Optimal 🔥
        min_k, solution = solver.start_optimal_solve()
        
        if solution:
            self.animate_coloring(solution)
            self.show_dashboard(analytics) # اعرض الداشبورد برضه
            
            messagebox.showinfo("Optimal Solution Found! 🌟", 
                                f"Minimum Colors Needed: {min_k}\n"
                                f"This is the scientifically minimal number (Chromatic Number).")
        else:
            messagebox.showerror("Error", "Could not find a solution!")
    
    def set_mode(self, mode, btn_ref):
        self.current_mode = mode
        self.btn_node.config(bg="#34495E"); self.btn_edge.config(bg="#34495E"); self.btn_del.config(bg="#C0392B")
        if mode != "DELETE": btn_ref.config(bg=ACCENT_COLOR)
        else: btn_ref.config(bg="#E74C3C")
        self.selected_node = None
    
    def log_status(self, msg): self.log_lbl.config(text=f">> {msg}")
    
    def on_resize(self, event): pass

    def show_dashboard(self, analytics, mode_title, result_info):
        dash = tk.Toplevel(self.root)
        dash.title(f"📊 Analytics - {mode_title}")
        dash.geometry("800x650") # كبرنا النافذة شوية
        dash.configure(bg="white")

        # --- Header Section (ثابت فوق) ---
        header_frame = tk.Frame(dash, bg="#ECF0F1", pady=15)
        header_frame.pack(fill=tk.X)
        
        # عنوان الـ Mode بخط كبير
        tk.Label(header_frame, text=mode_title, font=("Segoe UI", 16, "bold"), bg="#ECF0F1", fg="#2C3E50").pack()
        # نتيجة الحل (أخضر لو نجح، أحمر لو فشل)
        fg_color = "#27AE60" if analytics.solution_found else "#C0392B"
        tk.Label(header_frame, text=result_info, font=("Segoe UI", 12, "bold"), bg="#ECF0F1", fg=fg_color).pack(pady=(5,0))

        # --- NOTEBOOK (TABS SYSTEM) --- 🔥 التعديل الجوهري 🔥
        notebook = ttk.Notebook(dash)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # تاب 1: الرسوم البيانية
        chart_tab = tk.Frame(notebook, bg="white")
        notebook.add(chart_tab, text="   📈 Charts & Graphs   ")

        # تاب 2: الجدول
        table_tab = tk.Frame(notebook, bg="white")
        notebook.add(table_tab, text="   📋 Detailed Data Table   ")

        # ---------------------------
        # تصميم تاب 1 (الرسم البياني)
        # ---------------------------
        fig = Figure(figsize=(6, 4), dpi=100) # رسمة أكبر وأوضح
        ax = fig.add_subplot(111)
        
        metrics = ['Search Space\n(Nodes Visited)', 'Pruning Operations\n(Backtracks)']
        values = [analytics.nodes_visited, analytics.backtracks_count]
        colors = ['#3498DB', '#E74C3C']

        bars = ax.bar(metrics, values, color=colors, width=0.5)
        ax.set_title('Backtracking Algorithm Efficiency', fontsize=12, fontweight='bold', pad=15)
        ax.set_ylabel('Count')
        ax.grid(axis='y', linestyle='--', alpha=0.7) # خطوط شبكة خفيفة
        
        # كتابة الأرقام فوق العواميد بخط واضح
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (height*0.01), # رفعنا الرقم سنة فوق العمود
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        canvas_widget = FigureCanvasTkAgg(fig, master=chart_tab)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(pady=20, fill=tk.BOTH, expand=True)

        # ---------------------------
        # تصميم تاب 2 (الجدول)
        # ---------------------------
        # ستايل للجدول عشان يبقى شكله حديث
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Segoe UI", 11, "bold"), background="#BDC3C7", foreground="#2C3E50")
        style.configure("Treeview", font=("Segoe UI", 11), rowheight=30)

        columns = ("Metric", "Value")
        tree = ttk.Treeview(table_tab, columns=columns, show="headings", height=8, style="Treeview")
        tree.heading("Metric", text="Performance Metric"); tree.heading("Value", text="Measured Value")
        tree.column("Metric", anchor="w", width=300); tree.column("Value", anchor="center", width=300)

        # تعبئة البيانات
        status_icon = "✅ Success" if analytics.solution_found else "❌ Failed"
        tree.insert("", tk.END, values=("Algorithm Status", status_icon))
        tree.insert("", tk.END, values=("Total Execution Time", f"{analytics.execution_time_ms:.4f} ms"))
        tree.insert("", tk.END, values=("Search Space (Nodes Visited)", analytics.nodes_visited))
        tree.insert("", tk.END, values=("Pruning Operations (Backtracks)", analytics.backtracks_count))
        
        # سطر فاضي فاصل
        tree.insert("", tk.END, values=("", "")) 
        
        # تفاصيل النتيجة
        tree.insert("", tk.END, values=("Final Result Details", result_info.replace("\n", ". ")))

        # Scrollbar للجدول (احتياطي لو البيانات كتير)
        scrollbar = ttk.Scrollbar(table_tab, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # --- Footer (زرار الإغلاق) ---
        tk.Button(dash, text="Close Analytics Report", command=dash.destroy, bg="#34495E", fg="white", font=("Segoe UI", 10)).pack(pady=15)
    
    #! Standard Backtracking
    def run_standard_mode(self):
        if not self.nodes:
            messagebox.showwarning("Warning", "Draw a graph first!")
            return
        
        self.reset_colors()
        graph_data = self.adj_list.copy()
        for nid in self.nodes:
            if str(nid) not in graph_data: graph_data[str(nid)] = []
            
        limit = self.limit_scale.get()
        selected_colors = self.available_colors[:limit]
        
        analytics = PerformanceAnalytics()
        solver = Backtracking(graph_data, selected_colors, analytics=analytics)
        
        self.log_status(f"Running Standard Backtracking ({limit} colors)...")
        solution = solver.start_standard_solve()
        
        if solution:
            self.animate_coloring(solution)
            
            # حساب عدد الألوان المستخدمة عشان نعرضه
            used = set(solution.values())
            info_text = f"Solved using {len(used)} colors (Limit: {limit})"
            
            # نفتح الداشبورد بالبيانات الجديدة
            self.show_dashboard(analytics, "Standard Mode (Fast)", info_text)
        else:
            self.show_dashboard(analytics, "Standard Mode (Failed)", f"Failed to solve with {limit} colors.")
    
    # !Optimized Minimization
    def run_optimized_mode(self):
        if not self.nodes:
            messagebox.showwarning("Warning", "Draw a graph first!")
            return
        
        self.reset_colors()
        graph_data = self.adj_list.copy()
        for nid in self.nodes:
            if str(nid) not in graph_data: graph_data[str(nid)] = []
        
        analytics = PerformanceAnalytics()
        solver = Backtracking(graph_data, self.available_colors, analytics=analytics)
        
        self.log_status("Calculating Minimum Chromatic Number...")
        min_k, solution = solver.start_optimal_solve()
        
        if solution:
            self.animate_coloring(solution)
            
            info_text = f"✅ Optimal Solution Found!\nMinimum Colors Needed: {min_k}"
            
            self.show_dashboard(analytics, "Scientific Mode (Optimized)", info_text)
        else:
            self.show_dashboard(analytics, "Scientific Mode (Failed)", "Could not solve even with all colors.")

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphColoringUI(root)
    root.mainloop()