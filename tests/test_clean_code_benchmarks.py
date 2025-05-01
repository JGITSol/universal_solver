import pytest
import time
from clean_code.solver_adapter import SolverAdapter
from clean_code.neuro_symbolic_system import NeuroSymbolicMathSystem
from clean_code.visualization import MathVisualizer
from clean_code.tool_integration import ToolIntegrationManager
from clean_code.knowledge_management import KnowledgeManagementSystem
from clean_code.symbolic_regression_pipeline import SymbolicRegressionPipeline
from kan import KAN
from adv_resolver_math.math_ensemble_adv_ms_hackaton import MathProblemSolver, Agent

@pytest.mark.benchmark
def test_ensemble_solver_benchmark():
    agents = [Agent(name="Phi4", model="phi4-mini:latest", system_prompt="Solve.", temperature=0.1, max_tokens=128)]
    solver = MathProblemSolver(agents=agents)
    adapter = SolverAdapter(solver, solver_type="ensemble")
    system = NeuroSymbolicMathSystem(adapter)
    problem = "Solve for x: x^2 - 4 = 0"
    start = time.time()
    result = system.solve_problem(problem)
    elapsed = time.time() - start
    assert "answer" in result
    assert elapsed < 15  # Should solve within 15 seconds

@pytest.mark.benchmark
def test_symbolic_regression_benchmark():
    kan = KAN(width=[1, 3, 1], grid=5, k=3)
    pipeline = SymbolicRegressionPipeline(kan)
    result = pipeline.run(steps=100)
    assert "formula" in result
    assert len(str(result["formula"])) > 0
