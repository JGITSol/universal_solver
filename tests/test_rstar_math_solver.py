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
    assert "final_answer" in result
    assert "final_confidence" in result
    assert "supporting_agents" in result

def test_analyze_step_coherence_constant(solver):
    steps = ["1+1", "1+1"]
    score = solver.analyze_step_coherence(steps)
    assert isinstance(score, float)
    assert pytest.approx(1.0, rel=1e-2) == score

def test_analyze_conceptual_consistency(solver):
    sol = Solution("A", "ans", "This solves an equation step", 0.5)
    score = solver.analyze_conceptual_consistency(sol)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_analyze_computational_efficiency(solver):
    assert solver.analyze_computational_efficiency(["nested loop"]) == 0.6
    assert solver.analyze_computational_efficiency(["loop here"]) == 0.8
    assert solver.analyze_computational_efficiency(["simple step"]) == 1.0

def test_mcts_rollout_and_refine(monkeypatch, solver):
    sol = Solution("A", "x", "exp", 0.5)
    def fake_get(agent, problem, previous_solutions=None):
        return sol
    monkeypatch.setattr(solver, "get_solution", fake_get)
    agent = Agent("A", "m", "p", 0.1, 10)
    result = solver.mcts_rollout(agent, "p")
    assert isinstance(result, Solution)
    refined = solver.refine_with_feedback(sol, "p")
    assert isinstance(refined, Solution)

def test_verification_aware_vote(solver):
    sol_a = Solution("A", "ans", "exp", 0.4)
    sol_b = Solution("B", "ans", "exp", 0.6)
    sol_a.verification_score = 0.5; sol_a.process_reward = 0.5
    sol_b.verification_score = 0.7; sol_b.process_reward = 0.3
    vr = solver.verification_aware_vote([sol_a, sol_b])
    assert vr.answer == "ans"
    assert set(vr.agents_in_agreement) == {"A", "B"}
    assert isinstance(vr.confidence, float)
    assert 0.0 <= vr.confidence <= 1.0
