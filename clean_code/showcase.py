import sys
from pathlib import Path
from .solver_adapter import SolverAdapter
from .neuro_symbolic_system import NeuroSymbolicMathSystem
from .visualization import MathVisualizer
from .tool_integration import ToolIntegrationManager
from .knowledge_management import KnowledgeManagementSystem
from .symbolic_regression_pipeline import SymbolicRegressionPipeline

# Example imports for ensemble and KAN solvers (update paths as needed)
# from adv_resolver_math.math_ensemble_adv_ms_hackaton import MathProblemSolver, Agent
# from adv_resolver_math.ensemble_iterations.enhanced_solver import EnhancedMathSolver
# from kan import KAN

# === REAL AGENT AND SOLVER IMPORTS ===
from adv_resolver_math.math_ensemble_adv_ms_hackaton import MathProblemSolver, Agent
from adv_resolver_math.ensemble_iterations.enhanced_solver import EnhancedMathSolver
from adv_resolver_math.ensemble_iterations.memory_sharing_solver import MemorySharingMathSolver
from adv_resolver_math.ensemble_iterations.latent_space_solver import LatentSpaceMathSolver
from adv_resolver_math.ensemble_iterations.rstar_math_solver import RStarMathSolver
from kan import KAN
from .symbolic_regression_pipeline import SymbolicRegressionPipeline

# === AGENT POOL ===
agents = [
    Agent(name="Cogito", model="cogito:3b", system_prompt="You are a rigorous mathematician.", temperature=0.1, max_tokens=512, provider="ollama"),
    Agent(name="Gemma", model="gemma3:1b", system_prompt="Solve math problems step by step with detailed reasoning.", temperature=0.2, max_tokens=512, provider="ollama"),
    Agent(name="Phi4", model="phi4-mini:latest", system_prompt="Be concise and accurate in your math solutions.", temperature=0.1, max_tokens=512, provider="ollama"),
    Agent(name="LlamaNemotronUltra", model="nvidia/llama-3.1-nemotron-ultra-253b-v1:free", system_prompt="You are a SOTA LLM. Solve with detailed reasoning.", temperature=0.1, max_tokens=512, provider="openrouter"),
    Agent(name="Llama4Maverick", model="meta-llama/llama-4-maverick:free", system_prompt="You are a SOTA LLM. Solve with detailed reasoning.", temperature=0.1, max_tokens=512, provider="openrouter"),
    Agent(name="GeminiFlash", model="gemini-2.5-flash-preview-04-17", system_prompt="You are Google Gemini. Solve with step-by-step logic.", temperature=0.1, max_tokens=512, provider="gemini"),
    Agent(name="GeminiLite", model="gemini-2.0-flash-lite", system_prompt="You are Google Gemini. Solve with step-by-step logic.", temperature=0.1, max_tokens=512, provider="gemini"),
]

# === ENSEMBLE SOLVERS ===
solvers = [
    ("MathProblemSolver", MathProblemSolver(agents=agents)),
    ("EnhancedMathSolver", EnhancedMathSolver(agents=agents)),
    ("MemorySharingMathSolver", MemorySharingMathSolver(agents=agents)),
    ("LatentSpaceMathSolver", LatentSpaceMathSolver(agents=agents)),
    ("RStarMathSolver", RStarMathSolver(agents=agents)),
]

# === SYMBOLIC REGRESSION SOLVER ===
kan_model = KAN(width=[1, 3, 1], grid=5, k=3)
symbolic_pipeline = SymbolicRegressionPipeline(kan_model)

# Setup modules
visualizer = MathVisualizer()
tool_manager = ToolIntegrationManager()
knowledge_manager = KnowledgeManagementSystem(db_path="showcase_knowledge_db.json")

# === ENSEMBLE SHOWCASE ===
from .solver_adapter import SolverAdapter
from .neuro_symbolic_system import NeuroSymbolicMathSystem

problems = [
    ("Nonlinear Equation", "Solve for x: x^3 - 6x^2 + 11x - 6 = 0"),
    ("System of Equations", "Solve for x and y: 2x + 3y = 13, x - y = 1"),
    ("Calculus - Derivative", "Find the derivative of f(x) = x^4 * sin(x) with respect to x"),
    ("Calculus - Integral", "Compute the definite integral of x^2 * e^x from x=0 to x=2"),
    ("Optimization", "Find the maximum value of f(x) = -x^2 + 4x + 5"),
    ("Geometry", "What is the area of a triangle with sides 7, 8, and 9?"),
    ("Combinatorics", "How many ways can you arrange the letters in the word 'ALGEBRA'?")
]

for solver_name, solver in solvers:
    print(f"\n=== {solver_name} ===")
    adapter = SolverAdapter(solver, solver_type="ensemble")
    system = NeuroSymbolicMathSystem(
        solver_adapter=adapter,
        visualizer=visualizer,
        tool_manager=tool_manager,
        knowledge_manager=knowledge_manager
    )
    for problem_type, problem in problems:
        print(f"\n[{problem_type}] {problem}")
        result = system.solve_problem(problem, visualize=False)
        print("Result:", result)

# === SYMBOLIC REGRESSION SHOWCASE ===
symbolic_result = symbolic_pipeline.run(steps=500)
kan_adapter = SolverAdapter(kan_model, solver_type="symbolic_regression")
symbolic_system = NeuroSymbolicMathSystem(
    solver_adapter=kan_adapter,
    visualizer=visualizer,
    tool_manager=tool_manager,
    knowledge_manager=knowledge_manager
)
print("\n=== Symbolic Regression (KAN) ===")
print("Discovered formula:", symbolic_result["formula"])
visualizer.visualize_symbolic_regression(
    symbolic_result["x_train"], symbolic_result["y_train"],
    symbolic_result["x_test"], symbolic_result["y_test"], symbolic_result["y_pred"], symbolic_result["formula"]
)
