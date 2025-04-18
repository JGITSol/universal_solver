import pytest
from adv_resolver_math.ensemble_iterations.rstar_math_solver import RStarMathSolver
from adv_resolver_math.math_ensemble_adv_ms_hackaton import Agent, Solution

@pytest.fixture
def agents():
    return [
        Agent("A", "phi4-mini:latest", "Prompt A", 0.2, 1000),
        Agent("B", "phi4-mini:latest", "Prompt B", 0.3, 900)
    ]

@pytest.fixture
def solver(agents):
    return RStarMathSolver(agents=agents)

def test_symbolic_verification(solver):
    assert solver.symbolic_verification("x = 2 + 2") in [True, False]

def test_code_verification(solver):
    sol = Solution("A", "4", "x = 2 + 2\n", 0.8)
    score = solver.code_verification("x = 2 + 2", sol)
    assert 0.0 <= score <= 1.0

def test_process_reward(solver):
    sol = Solution("A", "4", "x = 2 + 2\n", 0.8)
    reward = solver.calculate_process_reward(sol)
    assert 0.0 <= reward <= 1.0

def test_solve_integration(solver):
    problem = "x = 2 + 2"
    result = solver.solve(problem)
    assert "solutions" in result
    for sol in result["solutions"]:
        assert "verification_score" in sol
        assert "process_reward" in sol
