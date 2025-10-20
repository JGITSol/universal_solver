import threading
import tkinter as tk
from tkinter import ttk

# Types for clarity when manipulating solver collections
from typing import Any, Dict, List, Tuple

import customtkinter as ctk

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
        self.label_title = ctk.CTkLabel(
            self,
            text=stage_title + " ▼",
            font=("Segoe UI", 14, "bold"),
            anchor="w",
            justify="left",
            cursor="hand2",
        )
        self.label_title.grid(row=0, column=0, sticky="ew", padx=8, pady=(2, 0))
        self.label_title.bind("<Button-1>", self.toggle_desc)
        # Description label, initially hidden
        self.desc_label = ctk.CTkLabel(
            self, text=description, font=("Segoe UI", 11), anchor="w", justify="left"
        )
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
            self.desc_label.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))
        else:
            self.label_title.configure(text=self.stage_title + " ▼")
            self.desc_label.grid_forget()


class OptionPanel(ctk.CTkFrame):
    def __init__(self, master, options, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        ctk.CTkLabel(self, text="Selected Options", font=("Segoe UI", 13, "bold")).pack(
            anchor="w", pady=(4, 0), padx=8
        )
        for key, value in options.items():
            row = ctk.CTkFrame(self)
            row.pack(anchor="w", fill="x", padx=12, pady=1)
            ctk.CTkLabel(
                row, text=f"{key}: ", font=("Segoe UI", 11, "bold"), width=60
            ).pack(side="left")
            ctk.CTkLabel(row, text=f"{value}", font=("Segoe UI", 11)).pack(side="left")


class ProjectStructurePanel(ctk.CTkFrame):
    def __init__(self, master, structure, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        ctk.CTkLabel(
            self, text="Project Structure", font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", pady=(4, 0), padx=8)
        tree_frame = ctk.CTkFrame(self)
        tree_frame.pack(expand=True, fill="both", padx=8, pady=4)
        tree = ttk.Treeview(tree_frame, height=8)
        tree.pack(expand=True, fill="both")
        self.insert_tree(tree, "", structure)

    def insert_tree(self, tree, parent, struct):
        for k, v in struct.items():
            node = tree.insert(parent, "end", text=k)
            if isinstance(v, dict):
                self.insert_tree(tree, node, v)


class ProblemProcessingPanel(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        ctk.CTkLabel(
            self, text="Problem Processing", font=("Segoe UI", 16, "bold")
        ).pack(anchor="w", padx=16, pady=(10, 4))
        self.text = ctk.CTkTextbox(self, font=("Consolas", 13), wrap="word")
        self.text.pack(expand=True, fill="both", padx=16, pady=8)
        self.text.insert("end", "[Processing log will appear here]")


class VotingPanel(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        ctk.CTkLabel(self, text="Voting & Results", font=("Segoe UI", 16, "bold")).pack(
            anchor="w", padx=16, pady=(10, 4)
        )
        self.results = ctk.CTkTextbox(self, font=("Consolas", 13), wrap="word")
        self.results.pack(expand=True, fill="both", padx=16, pady=8)
        self.results.insert("end", "[Voting results will appear here]")


class DebuggingPanel(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        ctk.CTkLabel(self, text="Debugging / Logs", font=("Segoe UI", 13, "bold")).pack(
            anchor="w", padx=8, pady=(6, 2)
        )
        self.log = ctk.CTkTextbox(self, font=("Consolas", 11), width=400, height=80)
        self.log.pack(expand=True, fill="both", padx=8, pady=4)
        self.log.insert("end", "[Debug logs will appear here]")


class UniversalSolverGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Universal Solver - Dev View")
        # Launch maximized and adapt to screen size
        try:
            self.state("zoomed")  # Windows, Linux
        except Exception:
            self.attributes("-zoomed", True)  # macOS fallback
        self.update_idletasks()
        w = self.winfo_screenwidth()
        h = self.winfo_screenheight()
        self.geometry(f"{int(w*0.98)}x{int(h*0.96)}+0+0")
        self.minsize(int(w * 0.7), int(h * 0.6))
        self.grid_columnconfigure(0, weight=2, uniform="col")  # Make left panel wider
        self.grid_columnconfigure(1, weight=3, uniform="col")
        self.grid_columnconfigure(2, weight=3, uniform="col")
        self.grid_rowconfigure(0, weight=1)
        # Left: Project info
        left_panel = ctk.CTkFrame(self)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        left_panel.grid_propagate(True)
        # Project structure (dummy for now)
        struct = {
            "adv_resolver_math": {
                "solver_registry.py": {},
                "universal_math_solver.py": {},
            },
            "tests": {"test_math_ensemble.py": {}},
        }
        ProjectStructurePanel(left_panel, struct).pack(fill="x", pady=(0, 8))
        # Dynamic solver options
        from showcase_advanced_math import solvers

        solver_pairs: List[Tuple[str, Any]] = list(solvers)
        self.solver_names: List[str] = [name for name, _ in solver_pairs]
        self.solver_map: Dict[str, Any] = {
            name: solver for name, solver in solver_pairs
        }
        ctk.CTkLabel(
            left_panel, text="Processing Option", font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", pady=(4, 0), padx=8)
        default_solver = self.solver_names[0] if self.solver_names else ""
        self.selected_solver = tk.StringVar(value=default_solver)
        solver_dropdown = ctk.CTkOptionMenu(
            left_panel, variable=self.selected_solver, values=self.solver_names
        )
        solver_dropdown.pack(fill="x", padx=12, pady=(0, 8))
        # Optionally, show details of selected solver
        self.solver_detail_label = ctk.CTkLabel(
            left_panel,
            text=f"Selected: {default_solver}"
            if default_solver
            else "No solvers registered",
            font=("Segoe UI", 11),
        )
        self.solver_detail_label.pack(anchor="w", padx=16, pady=(0, 6))

        def update_solver_detail(choice):
            self.solver_detail_label.configure(text=f"Selected: {choice}")

        solver_dropdown.configure(command=update_solver_detail)
        # Stage descriptions
        ctk.CTkLabel(left_panel, text="Stages", font=("Segoe UI", 12, "bold")).pack(
            anchor="w", pady=(8, 0), padx=8
        )
        stages = [
            (
                "1. Problem Processing",
                (
                    "Input is parsed and preprocessed. System identifies problem "
                    "type and requirements."
                ),
            ),
            (
                "2. Voting",
                (
                    "Multiple solver agents propose solutions. Voting mechanism "
                    "selects the most promising result."
                ),
            ),
            (
                "3. Debugging",
                (
                    "Detailed logs and traces are available for inspection. "
                    "Useful for dev and troubleshooting."
                ),
            ),
        ]
        for title, desc in stages:
            StageDescription(left_panel, title, desc).pack(fill="x", pady=(0, 2))
        # Center: Problem processing and voting
        center_panel = ctk.CTkFrame(self)
        center_panel.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        # --- Input Field & Flow Control Buttons ---
        input_frame = ctk.CTkFrame(center_panel)
        input_frame.pack(fill="x", pady=(0, 4), padx=4)
        self.input_var = tk.StringVar()
        self.input_entry = ctk.CTkEntry(
            input_frame,
            textvariable=self.input_var,
            font=("Consolas", 13),
            width=400,
            placeholder_text="Enter problem (supports LaTeX, e.g. x^2 + y^2 = 1)",
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        # Button group
        self.send_btn = ctk.CTkButton(
            input_frame, text="Send Message", command=self.on_send, fg_color="#2a8cff"
        )
        self.send_btn.pack(side="left", padx=(0, 4))
        self.hard_stop_btn = ctk.CTkButton(
            input_frame,
            text="Hard Stop (Scram)",
            command=self.on_hard_stop,
            fg_color="#ff3c3c",
        )
        self.hard_stop_btn.pack(side="left", padx=(0, 4))
        self.soft_stop_btn = ctk.CTkButton(
            input_frame,
            text="Soft Stop (Summary)",
            command=self.on_soft_stop,
            fg_color="#ffd633",
            text_color="#222",
        )
        self.soft_stop_btn.pack(side="left")
        if not self.solver_names:
            self.send_btn.configure(state="disabled")
            self.hard_stop_btn.configure(state="disabled")
            self.soft_stop_btn.configure(state="disabled")
        # --- Panels ---
        self.proc_panel = ProblemProcessingPanel(center_panel)
        self.proc_panel.pack(fill="both", expand=True, pady=4)
        self.vote_panel = VotingPanel(center_panel)
        self.vote_panel.pack(fill="both", expand=True, pady=4)
        # Right: Debugging/logs
        right_panel = ctk.CTkFrame(self)
        right_panel.grid(row=0, column=2, sticky="nsew", padx=8, pady=8)
        self.debug_panel = DebuggingPanel(right_panel)
        self.debug_panel.pack(fill="both", expand=True, pady=4)
        # Menu bar
        self.create_menu()

    def on_send(self):
        query = self.input_var.get()
        solver_name = self.selected_solver.get()
        solver = self.solver_map.get(solver_name)
        if not solver:
            self._handle_solver_error(ValueError("No solver selected."))
            return
        self.proc_panel.text.delete("1.0", "end")
        self.vote_panel.results.delete("1.0", "end")
        self.debug_panel.log.delete("1.0", "end")
        self._set_controls_state("disabled")
        # Confirmation message (label at top of center panel)
        if (
            hasattr(self, "confirmation_label")
            and self.confirmation_label.winfo_exists()
        ):
            self.confirmation_label.destroy()
        self.confirmation_label = ctk.CTkLabel(
            self,
            text=f"Message sent to {solver_name} at your request.",
            font=("Segoe UI", 12, "bold"),
            text_color="#1e8c3a",
        )
        self.confirmation_label.place(relx=0.5, rely=0.04, anchor="n")
        self.after(
            2500,
            lambda: (
                self.confirmation_label.destroy()
                if self.confirmation_label.winfo_exists()
                else None
            ),
        )

        # Run in background thread to keep GUI responsive
        def run_solver():
            try:
                # Unified interface: EnhancedMathSolver, MemorySharingMathSolver,
                # and LatentSpaceMathSolver call get_solution/vote_on_solutions;
                # RStarMathSolver exposes solve instead.
                from showcase_advanced_math import agents

                processing_output = ""
                voting_output = ""
                debug_output = "[Processing completed successfully]\n"

                if solver_name == "RStarMathSolver":
                    result = solver.solve(query)
                    processing_output = f"[RStarMathSolver Result]\n{result}\n"
                    voting_output = "Final Answer: {}\nConfidence: {}\n".format(
                        result.get("answer", result),
                        result.get("confidence", ""),
                    )
                else:
                    agent_solutions = [
                        solver.get_solution(agent, query) for agent in agents
                    ]
                    processing_output = "\n".join(
                        [
                            (
                                f"{s.agent_name}: {s.answer}\n{s.explanation}\n"
                                f"Confidence: {s.confidence}"
                            )
                            for s in agent_solutions
                        ]
                    )
                    vote_result = solver.vote_on_solutions(agent_solutions)
                    agent_list = (
                        ", ".join(vote_result.agents_in_agreement)
                        if hasattr(vote_result, "agents_in_agreement")
                        else ""
                    )
                    voting_output = (
                        "Final Answer: {}\n"
                        "Confidence: {}\n"
                        "Agents in agreement: {}\n"
                    ).format(
                        vote_result.answer,
                        vote_result.confidence,
                        agent_list,
                    )

                self.after(
                    0,
                    lambda: self._handle_solver_success(
                        processing_output, voting_output, debug_output
                    ),
                )
            except Exception as exc:
                self.after(0, lambda err=exc: self._handle_solver_error(err))

        threading.Thread(target=run_solver, daemon=True).start()

    def on_hard_stop(self):
        """Handle the Hard Stop button by simulating an immediate halt."""
        self.proc_panel.text.insert(
            "end", "\n[Hard Stop initiated: Processing interrupted!]"
        )
        self.vote_panel.results.insert("end", "\n[Hard Stop: Voting interrupted!]")
        self.debug_panel.log.insert("end", "\n[Hard Stop: Debugging interrupted!]")

    def on_soft_stop(self):
        """Handle the Soft Stop button by simulating a graceful shutdown."""
        self.proc_panel.text.insert(
            "end", "\n[Soft Stop: Finishing current step and summarizing...]"
        )
        self.vote_panel.results.insert(
            "end", "\n[Soft Stop: Step finished, summary generated.]"
        )
        self.debug_panel.log.insert(
            "end", "\n[Soft Stop: Step finished, summary generated.]"
        )

    def _handle_solver_success(
        self, processing_output: str, voting_output: str, debug_output: str
    ) -> None:
        """Update UI after successful solver execution."""
        self.proc_panel.text.insert("end", processing_output)
        self.vote_panel.results.insert("end", voting_output)
        if debug_output:
            self.debug_panel.log.insert("end", debug_output)
        self._set_controls_state("normal")

    def _handle_solver_error(self, error: Exception) -> None:
        """Display solver error details to the user."""
        message = f"[ERROR] {error}\n"
        self.proc_panel.text.insert("end", message)
        self.vote_panel.results.insert("end", message)
        self.debug_panel.log.insert("end", message)
        self._set_controls_state("normal")

    def _set_controls_state(self, state: str) -> None:
        """Enable or disable main action controls."""
        self.send_btn.configure(state=state)
        if state == "disabled":
            return
        self.hard_stop_btn.configure(state="normal")
        self.soft_stop_btn.configure(state="normal")

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
