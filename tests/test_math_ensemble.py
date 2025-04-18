import pytest
from adv_resolver_math.math_ensemble_adv_ms_hackaton import MathProblemSolver, Agent, Solution

def test_basic_solution():
    agents = [
        Agent("A", "phi4-mini:latest", "Prompt A", 0.2, 1000),
        Agent("B", "phi4-mini:latest", "Prompt B", 0.3, 900)
    ]
    solver = MathProblemSolver(agents=agents)
    sol = solver.get_solution(agents[0], "x = 2 + 2")
    assert isinstance(sol, Solution)
    assert hasattr(sol, 'answer')
    assert hasattr(sol, 'explanation')
    assert hasattr(sol, 'confidence')

def test_vote_on_solutions():
    agents = [
        Agent("A", "phi4-mini:latest", "Prompt A", 0.2, 1000),
        Agent("B", "phi4-mini:latest", "Prompt B", 0.3, 900)
    ]
    solver = MathProblemSolver(agents=agents)
    solutions = [
        Solution("A", "4", "Explanation 1", 0.8),
        Solution("B", "4", "Explanation 2", 0.6)
    ]
    result = solver.vote_on_solutions(solutions)
    assert hasattr(result, 'answer')
    assert hasattr(result, 'confidence')
    assert hasattr(result, 'agents_in_agreement')
