import pytest
from adv_resolver_math.ensemble_iterations.enhanced_solver import EnhancedMathSolver
from adv_resolver_math.math_ensemble_adv_ms_hackaton import Agent, Solution

@pytest.fixture
def agents():
    return [
        Agent("A", "phi4-mini:latest", "Prompt A", 0.2, 1000),
        Agent("B", "phi4-mini:latest", "Prompt B", 0.3, 900)
    ]

@pytest.fixture
def solver(agents):
    return EnhancedMathSolver(agents=agents)

def test_semantic_voting(solver):
    solutions = [
        Solution("A", "x=5", "Explanation 1", 0.8),
        Solution("B", "5", "Explanation 2", 0.6)
    ]
    result = solver.vote_on_solutions(solutions)
    assert result.answer in ["x=5", "5"]
    assert set(result.agents_in_agreement) in [{"A"}, {"A", "B"}]
    assert 0 < result.confidence <= 1

def test_select_representative_answer(solver):
    solutions = [
        Solution("A", "5", "Explanation 1", 0.8),
        Solution("B", "5", "Explanation 2", 0.6)
    ]
    assert solver._select_representative_answer(solutions) == "5"
