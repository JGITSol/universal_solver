import pytest
from adv_resolver_math.ensemble_iterations.memory_sharing_solver import MemorySharingMathSolver
from adv_resolver_math.math_ensemble_adv_ms_hackaton import Agent, Solution
import numpy as np

@pytest.fixture
def agents():
    return [
        Agent("A", "phi4-mini:latest", "Prompt A", 0.2, 1000),
        Agent("B", "phi4-mini:latest", "Prompt B", 0.3, 900)
    ]

@pytest.fixture
def solver(agents):
    return MemorySharingMathSolver(agents=agents, embedding_dim=384)

def test_memory_update_and_aggregation(solver):
    emb = np.ones(384)
    solver.update_memory("A", emb)
    pooled = solver.aggregate_memories()
    assert pooled.shape[-1] == 384
    assert pooled.shape[-1] == solver.embedding_dim

def test_vote_on_solutions_with_memory(solver):
    solutions = [
        Solution("A", "x=5", "Explanation 1", 0.8),
        Solution("B", "5", "Explanation 2", 0.6)
    ]
    result = solver.vote_on_solutions(solutions)
    assert hasattr(result, "answer")
    assert hasattr(result, "confidence")
    assert hasattr(result, "agents_in_agreement")
