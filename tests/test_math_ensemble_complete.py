import pytest
from adv_resolver_math.math_ensemble_adv_ms_hackaton import MathProblemSolver, Agent, Solution

def test_refine_solutions():
    agents = [
        Agent("A", "phi4-mini:latest", "Prompt A", 0.2, 1000),
        Agent("B", "phi4-mini:latest", "Prompt B", 0.3, 900)
    ]
    solver = MathProblemSolver(agents=agents)
    solutions = [
        Solution("A", "4", "Explanation 1", 0.8),
        Solution("B", "4", "Explanation 2", 0.6)
    ]
    discussion = "Both agents agree on the answer."
    refined = solver.refine_solutions("x = 2 + 2", solutions, discussion)
    assert isinstance(refined, list)
    assert all(isinstance(r, Solution) for r in refined)

def test_solve():
    agents = [
        Agent("A", "phi4-mini:latest", "Prompt A", 0.2, 1000),
        Agent("B", "phi4-mini:latest", "Prompt B", 0.3, 900)
    ]
    solver = MathProblemSolver(agents=agents)
    result = solver.solve("x = 2 + 2")
    assert isinstance(result, dict)
    assert "solutions" in result
