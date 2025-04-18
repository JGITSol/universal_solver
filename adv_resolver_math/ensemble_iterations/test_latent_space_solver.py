import pytest
from adv_resolver_math.ensemble_iterations.latent_space_solver import LatentSpaceMathSolver
from adv_resolver_math.math_ensemble_adv_ms_hackaton import Agent, Solution

@pytest.fixture
def agents():
    return [
        Agent("A", "phi4-mini:latest", "Prompt A", 0.2, 1000),
        Agent("B", "phi4-mini:latest", "Prompt B", 0.3, 900)
    ]

@pytest.fixture
def solver(agents):
    return LatentSpaceMathSolver(agents=agents)

def test_latent_voting(solver):
    solutions = [
        Solution("A", "x=5", "Explanation 1", 0.8),
        Solution("B", "5", "Explanation 2", 0.6)
    ]
    result = solver.vote_on_solutions(solutions)
    assert result in solutions
