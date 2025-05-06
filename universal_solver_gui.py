import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import os

# Ensure CTk theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class StageDescription(ctk.CTkFrame):
    def __init__(self, master, stage_title, description, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.expanded = False
        self.stage_title = stage_title
        self.description = description
        self.label_title = ctk.CTkLabel(self, text=stage_title + " ▼", font=("Segoe UI", 14, "bold"), anchor="w", justify="left", cursor="hand2")
        self.label_title.grid(row=0, column=0, sticky="ew", padx=8, pady=(2,0))
        self.label_title.bind("<Button-1>", self.toggle_desc)
        # Description label, initially hidden
        self.desc_label = ctk.CTkLabel(self, text=description, font=("Segoe UI", 11), anchor="w", justify="left")
        self.bind("<Configure>", self._update_wraplength)
        self._show_desc(False)
    def _update_wraplength(self, event=None):
        wrap = max(100, int(self.winfo_width() * 0.95) - 32)
        self.desc_label.configure(wraplength=wrap)
    def toggle_desc(self, event=None):
        self.expanded = not self.expanded
        self._show_desc(self.expanded)
    def _show_desc(self, show):
        if show:
            self.label_title.configure(text=self.stage_title + " ▲")
            self.desc_label.grid(row=1, column=0, sticky="ew", padx=16, pady=(0,10))
        else:
            self.label_title.configure(text=self.stage_title + " ▼")
            self.desc_label.grid_forget()

class OptionPanel(ctk.CTkFrame):
    def __init__(self, master, options, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        ctk.CTkLabel(self, text="Selected Options", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(4,0), padx=8)
        for key, value in options.items():
            row = ctk.CTkFrame(self)
            row.pack(anchor="w", fill="x", padx=12, pady=1)
            ctk.CTkLabel(row, text=f"{key}: ", font=("Segoe UI", 11, "bold"), width=60).pack(side="left")
            ctk.CTkLabel(row, text=f"{value}", font=("Segoe UI", 11)).pack(side="left")

class ProjectStructurePanel(ctk.CTkFrame):
    def __init__(self, master, structure, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        ctk.CTkLabel(self, text="Project Structure", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(4,0), padx=8)
        tree_frame = ctk.CTkFrame(self)
        tree_frame.pack(expand=True, fill="both", padx=8, pady=4)
        tree = ttk.Treeview(tree_frame, height=8)
        tree.pack(expand=True, fill="both")
        self.insert_tree(tree, '', structure)
    def insert_tree(self, tree, parent, struct):
        for k, v in struct.items():
            node = tree.insert(parent, 'end', text=k)
            if isinstance(v, dict):
                self.insert_tree(tree, node, v)

class ProblemProcessingPanel(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        ctk.CTkLabel(self, text="Problem Processing", font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=16, pady=(10,4))
        self.text = ctk.CTkTextbox(self, font=("Consolas", 13), wrap="word")
        self.text.pack(expand=True, fill="both", padx=16, pady=8)
        self.text.insert("end", "[Processing log will appear here]")

class VotingPanel(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        ctk.CTkLabel(self, text="Voting & Results", font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=16, pady=(10,4))
        self.results = ctk.CTkTextbox(self, font=("Consolas", 13), wrap="word")
        self.results.pack(expand=True, fill="both", padx=16, pady=8)
        self.results.insert("end", "[Voting results will appear here]")

class DebuggingPanel(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        ctk.CTkLabel(self, text="Debugging / Logs", font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=8, pady=(6,2))
        self.log = ctk.CTkTextbox(self, font=("Consolas", 11), width=400, height=80)
        self.log.pack(expand=True, fill="both", padx=8, pady=4)
        self.log.insert("end", "[Debug logs will appear here]")

class UniversalSolverGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Universal Solver - Dev View")
        # Launch maximized and adapt to screen size
        try:
            self.state('zoomed')  # Windows, Linux
        except Exception:
            self.attributes('-zoomed', True)  # macOS fallback
        self.update_idletasks()
        w = self.winfo_screenwidth()
        h = self.winfo_screenheight()
        self.geometry(f"{int(w*0.98)}x{int(h*0.96)}+0+0")
        self.minsize(int(w*0.7), int(h*0.6))
        self.grid_columnconfigure(0, weight=2, uniform="col")  # Make left panel wider
        self.grid_columnconfigure(1, weight=3, uniform="col")
        self.grid_columnconfigure(2, weight=3, uniform="col")
        self.grid_rowconfigure(0, weight=1)
        # Left: Project info
        left_panel = ctk.CTkFrame(self)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        left_panel.grid_propagate(True)
        # Project structure (dummy for now)
        struct = {"adv_resolver_math": {"solver_registry.py": {}, "universal_math_solver.py": {}}, "tests": {"test_math_ensemble.py": {}}}
        ProjectStructurePanel(left_panel, struct).pack(fill="x", pady=(0,8))
        # Dynamic solver options
        from showcase_advanced_math import solvers
        self.solver_names = [name for name, _ in solvers]
        self.solver_map = {name: solver for name, solver in solvers}
        ctk.CTkLabel(left_panel, text="Processing Option", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(4,0), padx=8)
        self.selected_solver = tk.StringVar(value=self.solver_names[0])
        solver_dropdown = ctk.CTkOptionMenu(left_panel, variable=self.selected_solver, values=self.solver_names)
        solver_dropdown.pack(fill="x", padx=12, pady=(0,8))
        # Optionally, show details of selected solver
        self.solver_detail_label = ctk.CTkLabel(left_panel, text=f"Selected: {self.solver_names[0]}", font=("Segoe UI", 11))
        self.solver_detail_label.pack(anchor="w", padx=16, pady=(0,6))
        def update_solver_detail(choice):
            self.solver_detail_label.configure(text=f"Selected: {choice}")
        solver_dropdown.configure(command=update_solver_detail)
        # Stage descriptions
        ctk.CTkLabel(left_panel, text="Stages", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(8,0), padx=8)
        stages = [
            ("1. Problem Processing", "Input is parsed and preprocessed. System identifies problem type and requirements."),
            ("2. Voting", "Multiple solver agents propose solutions. Voting mechanism selects the most promising result."),
            ("3. Debugging", "Detailed logs and traces are available for inspection. Useful for dev and troubleshooting."),
        ]
        for title, desc in stages:
            StageDescription(left_panel, title, desc).pack(fill="x", pady=(0,2))
        # Center: Problem processing and voting
        center_panel = ctk.CTkFrame(self)
        center_panel.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        # --- Input Field & Flow Control Buttons ---
        input_frame = ctk.CTkFrame(center_panel)
        input_frame.pack(fill="x", pady=(0,4), padx=4)
        self.input_var = tk.StringVar()
        self.input_entry = ctk.CTkEntry(input_frame, textvariable=self.input_var, font=("Consolas", 13), width=400, placeholder_text="Enter problem (supports LaTeX, e.g. x^2 + y^2 = 1)")
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0,8))
        # Button group
        send_btn = ctk.CTkButton(input_frame, text="Send Message", command=self.on_send, fg_color="#2a8cff")
        send_btn.pack(side="left", padx=(0, 4))
        hard_stop_btn = ctk.CTkButton(input_frame, text="Hard Stop (Scram)", command=self.on_hard_stop, fg_color="#ff3c3c")
        hard_stop_btn.pack(side="left", padx=(0, 4))
        soft_stop_btn = ctk.CTkButton(input_frame, text="Soft Stop (Summary)", command=self.on_soft_stop, fg_color="#ffd633", text_color="#222")
        soft_stop_btn.pack(side="left")
        # --- Panels ---
        self.proc_panel = ProblemProcessingPanel(center_panel)
        self.proc_panel.pack(fill="both", expand=True, pady=4)
        self.vote_panel = VotingPanel(center_panel)
        self.vote_panel.pack(fill="both", expand=True, pady=4)

    def on_send(self):
        query = self.input_var.get()
        solver_name = self.selected_solver.get()
        solver = self.solver_map[solver_name]
        self.proc_panel.text.delete("1.0", "end")
        self.vote_panel.results.delete("1.0", "end")
        if hasattr(self, 'debug_panel'):
            self.debug_panel.log.delete("1.0", "end")
        # Confirmation message (label at top of center panel)
        if hasattr(self, 'confirmation_label') and self.confirmation_label.winfo_exists():
            self.confirmation_label.destroy()
        self.confirmation_label = ctk.CTkLabel(self, text=f"Message sent to {solver_name} at your request.", font=("Segoe UI", 12, "bold"), text_color="#1e8c3a")
        self.confirmation_label.place(relx=0.5, rely=0.04, anchor="n")
        self.after(2500, lambda: self.confirmation_label.destroy() if self.confirmation_label.winfo_exists() else None)
        # Run in background thread to keep GUI responsive
        import threading
        def run_solver():
            try:
                # Unified interface: EnhancedMathSolver, MemorySharingMathSolver, LatentSpaceMathSolver all use get_solution/vote_on_solutions; RStarMathSolver uses solve
                from showcase_advanced_math import agents
                if solver_name == "RStarMathSolver":
                    result = solver.solve(query)
                    self.proc_panel.text.insert("end", f"[RStarMathSolver Result]\n{result}\n")
                    self.vote_panel.results.insert("end", f"Final Answer: {result.get('answer', result)}\nConfidence: {result.get('confidence', '')}\n")
                else:
                    agent_solutions = [solver.get_solution(agent, query) for agent in agents]
                    result = solver.vote_on_solutions(agent_solutions)
                    self.proc_panel.text.insert("end", "\n".join([f"{s.agent_name}: {s.answer}\n{s.explanation}\nConfidence: {s.confidence}" for s in agent_solutions]))
                    self.vote_panel.results.insert("end", f"Final Answer: {result.answer}\nConfidence: {result.confidence}\nAgents in agreement: {', '.join(result.agents_in_agreement) if hasattr(result, 'agents_in_agreement') else ''}\n")
                if hasattr(self, 'debug_panel'):
                    self.debug_panel.log.insert("end", "[Processing completed successfully]\n")
            except Exception as e:
                if hasattr(self, 'debug_panel'):
                    self.debug_panel.log.insert("end", f"[ERROR] {e}\n")
                self.proc_panel.text.insert("end", f"[ERROR] {e}\n")
        threading.Thread(target=run_solver, daemon=True).start()

    def on_hard_stop(self):
        """Triggered when Hard Stop (Scram) is pressed. Simulate immediate stop."""
        self.proc_panel.text.insert("end", "\n[Hard Stop initiated: Processing interrupted!]")
        self.vote_panel.results.insert("end", "\n[Hard Stop: Voting interrupted!]")
        if hasattr(self, 'debug_panel'):
            self.debug_panel.log.insert("end", "\n[Hard Stop: Debugging interrupted!]")

    def on_soft_stop(self):
        """Triggered when Soft Stop (Summary) is pressed. Simulate graceful stop with summary."""
        self.proc_panel.text.insert("end", "\n[Soft Stop: Finishing current step and summarizing...]")
        self.vote_panel.results.insert("end", "\n[Soft Stop: Step finished, summary generated.]")
        if hasattr(self, 'debug_panel'):
            self.debug_panel.log.insert("end", "\n[Soft Stop: Step finished, summary generated.]")
        
        # Right: Debugging/logs
        right_panel = ctk.CTkFrame(self)
        right_panel.grid(row=0, column=2, sticky="nsew", padx=8, pady=8)
        # Save reference for debug panel refresh
        self.debug_panel = DebuggingPanel(right_panel)
        self.debug_panel.pack(fill="both", expand=True, pady=4)
        # Menu bar
        self.create_menu()
    def create_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

def main():
    app = UniversalSolverGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
