"""
Showcase script for advanced mathematics problem solving using all advanced ensemble solvers.
Demonstrates nonlinear equations, systems, calculus, optimization, and geometry/combinatorics.
"""
import sys
import os
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from adv_resolver_math.ensemble_iterations.enhanced_solver import EnhancedMathSolver
from adv_resolver_math.ensemble_iterations.memory_sharing_solver import MemorySharingMathSolver
from adv_resolver_math.ensemble_iterations.latent_space_solver import LatentSpaceMathSolver
from adv_resolver_math.ensemble_iterations.rstar_math_solver import RStarMathSolver
from adv_resolver_math.math_ensemble_adv_ms_hackaton import Agent

# Setup rich console
theme_console = Console()

# Output directory for results
OUTPUT_DIR = Path("showcase_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define a pool of diverse agents (can adjust models as available on your Ollama server)
agents = [
    Agent(name="Cogito", model="cogito:3b", system_prompt="You are a rigorous mathematician.", temperature=0.1, max_tokens=512),
    Agent(name="Gemma", model="gemma3:1b", system_prompt="Solve math problems step by step with detailed reasoning.", temperature=0.2, max_tokens=512),
    Agent(name="Phi4", model="phi4-mini:latest", system_prompt="Be concise and accurate in your math solutions.", temperature=0.1, max_tokens=512)
]

# Instantiate solvers
solvers = [
    ("EnhancedMathSolver", EnhancedMathSolver(agents=agents)),
    ("MemorySharingMathSolver", MemorySharingMathSolver(agents=agents)),
    ("LatentSpaceMathSolver", LatentSpaceMathSolver(agents=agents)),
    ("RStarMathSolver", RStarMathSolver(agents=agents))
]

# Define complex math problems
test_problems = [
    ("Nonlinear Equation", "Solve for x: x^3 - 6x^2 + 11x - 6 = 0"),
    ("System of Equations", "Solve for x and y: 2x + 3y = 13, x - y = 1"),
    ("Calculus - Derivative", "Find the derivative of f(x) = x^4 * sin(x) with respect to x"),
    ("Calculus - Integral", "Compute the definite integral of x^2 * e^x from x=0 to x=2"),
    ("Optimization", "Find the maximum value of f(x) = -x^2 + 4x + 5"),
    ("Geometry", "What is the area of a triangle with sides 7, 8, and 9?"),
    ("Combinatorics", "How many ways can you arrange the letters in the word 'ALGEBRA'?")
]

def log_result_rich(solver_name, problem_type, problem, result):
    # Use rich to print a colored, structured summary
    panel_title = f"[bold blue]{solver_name}[/bold blue] | [magenta]{problem_type}[/magenta]"
    if isinstance(result, dict):
        answer = result.get('final_answer', 'N/A')
        confidence = result.get('final_confidence', 0.0)
        agents = result.get('supporting_agents', [])
        table = Table(title=panel_title, box=box.ROUNDED, highlight=True)
        table.add_column("Agent", style="cyan", no_wrap=True)
        table.add_column("Answer", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("Explanation", style="white")
        for sol in result.get('solutions', []):
            table.add_row(
                sol['agent_name'],
                sol['answer'],
                f"{sol['confidence']:.2f}",
                sol['explanation'][:80] + ("..." if len(sol['explanation']) > 80 else "")
            )
        theme_console.print(Panel(table, title=f"Final: [bold green]{answer}[/bold green] | Confidence: [yellow]{confidence:.2f}[/yellow] | Supporting: {agents}", border_style="bold blue"))
    else:
        answer = getattr(result, 'answer', 'N/A')
        confidence = getattr(result, 'confidence', 0.0)
        agents = getattr(result, 'agents_in_agreement', [])
        table = Table(title=panel_title, box=box.ROUNDED, highlight=True)
        table.add_column("Agent(s)", style="cyan", no_wrap=True)
        table.add_column("Answer", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_row(str(agents), str(answer), f"{confidence:.2f}")
        theme_console.print(Panel(table, title=f"Final: [bold green]{answer}[/bold green] | Confidence: [yellow]{confidence:.2f}[/yellow] | Supporting: {agents}", border_style="bold blue"))

def collect_decisions(solver_name, problem_type, problem, result, agent_solutions=None):
    # Returns a list of dicts, one per agent/decision, plus a consensus row if applicable
    rows = []
    if isinstance(result, dict):
        # Standard: all agent solutions are in result['solutions']
        for sol in result.get('solutions', []):
            rows.append({
                'solver': solver_name,
                'problem_type': problem_type,
                'problem': problem,
                'agent': sol.get('agent_name', 'UNKNOWN'),
                'answer': sol.get('answer'),
                'confidence': sol.get('confidence'),
                'explanation': sol.get('explanation'),
                'final_answer': result.get('final_answer'),
                'final_confidence': result.get('final_confidence'),
                'supporting_agents': ','.join(result.get('supporting_agents', [])),
                'row_type': 'agent',
            })
        # Add consensus row
        rows.append({
            'solver': solver_name,
            'problem_type': problem_type,
            'problem': problem,
            'agent': 'CONSENSUS',
            'answer': result.get('final_answer'),
            'confidence': result.get('final_confidence'),
            'explanation': '',
            'final_answer': result.get('final_answer'),
            'final_confidence': result.get('final_confidence'),
            'supporting_agents': ','.join(result.get('supporting_agents', [])),
            'row_type': 'consensus',
        })
    else:
        # For solvers returning only a consensus (e.g. LatentSpaceMathSolver), log all agent solutions if available
        if agent_solutions is not None:
            for sol in agent_solutions:
                rows.append({
                    'solver': solver_name,
                    'problem_type': problem_type,
                    'problem': problem,
                    'agent': getattr(sol, 'agent_name', 'UNKNOWN'),
                    'answer': getattr(sol, 'answer', None),
                    'confidence': getattr(sol, 'confidence', None),
                    'explanation': getattr(sol, 'explanation', ''),
                    'final_answer': getattr(result, 'answer', None),
                    'final_confidence': getattr(result, 'confidence', None),
                    'supporting_agents': ','.join(getattr(result, 'agents_in_agreement', [])),
                    'row_type': 'agent',
                })
        # Always add consensus row
        rows.append({
            'solver': solver_name,
            'problem_type': problem_type,
            'problem': problem,
            'agent': 'CONSENSUS',
            'answer': getattr(result, 'answer', None),
            'confidence': getattr(result, 'confidence', None),
            'explanation': '',
            'final_answer': getattr(result, 'answer', None),
            'final_confidence': getattr(result, 'confidence', None),
            'supporting_agents': ','.join(getattr(result, 'agents_in_agreement', [])),
            'row_type': 'consensus',
        })
    return rows

def export_results(df: pd.DataFrame, out_dir: Path):
    excel_path = out_dir / "math_showcase_results.xlsx"
    parquet_path = out_dir / "math_showcase_results.parquet"
    df.to_excel(excel_path, index=False)
    df.to_parquet(parquet_path, index=False)
    theme_console.print(f"[bold green]Exported results to:[/bold green] [cyan]{excel_path}[/cyan] and [cyan]{parquet_path}[/cyan]")

if __name__ == "__main__":
    all_rows = []
    for problem_type, problem in test_problems:
        for solver_name, solver in solvers:
            try:
                agent_solutions = None
                if solver_name == "RStarMathSolver":
                    result = solver.solve(problem)
                else:
                    agent_solutions = [solver.get_solution(agent, problem) for agent in agents]
                    result = solver.vote_on_solutions(agent_solutions)
                log_result_rich(solver_name, problem_type, problem, result)
                all_rows.extend(collect_decisions(solver_name, problem_type, problem, result, agent_solutions=agent_solutions))
            except Exception as e:
                theme_console.print(f"[bold red][ERROR][/bold red] {solver_name} failed on {problem_type}: {e}")
    if all_rows:
        df = pd.DataFrame(all_rows)
        export_results(df, OUTPUT_DIR)
    else:
        theme_console.print("[bold yellow]No results to export.[/bold yellow]")
