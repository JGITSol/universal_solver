import pytest
from unittest.mock import MagicMock, patch
from math_ensemble_adv_ms_hackaton import Agent, MathProblemSolver, Solution, VotingResult

@pytest.fixture
def mock_ollama_client():
    client = MagicMock()
    client.invoke.return_value = "Mocked explanation"
    return client

@pytest.fixture
def sample_agents():
    return [
        Agent("Expert", "llama2", "You are a math expert", 0.2, 1000),
        Agent("Creative", "mistral", "Think creatively", 0.7, 500)
    ]

@patch('langchain_ollama.OllamaLLM')
def test_initial_solution(mock_ollama, mock_ollama_client, sample_agents):
    mock_ollama.return_value = mock_ollama_client
    solver = MathProblemSolver(sample_agents)
    
    problem = "Solve 2x + 5 = 15"
    solutions = [solver.get_solution(agent, problem) for agent in sample_agents]
    
    assert len(solutions) == 2
    assert all(isinstance(s, Solution) for s in solutions)

@patch('langchain_ollama.OllamaLLM')
def test_voting_mechanism(mock_ollama, mock_ollama_client, sample_agents):
    mock_ollama.return_value = mock_ollama_client
    solver = MathProblemSolver(sample_agents)
    
    solutions = [
        Solution("Expert", "x=5", "Explanation", 0.8),
        Solution("Creative", "5", "Different approach", 0.6)
    ]
    result = solver.vote_on_solutions(solutions)
    
    assert isinstance(result, VotingResult)
    assert result.confidence >= 0

@pytest.mark.parametrize("answers,expected", [
    (["x=5", "x = 5"], "x=5"),
    (["5", "five"], "5"),
    (["Error", "x=5"], "x=5")
])
def test_answer_normalization(answers, expected):
    solver = MathProblemSolver([])
    solutions = [Solution(f"Agent{i}", ans, "", 0.8) for i, ans in enumerate(answers)]
    result = solver.vote_on_solutions(solutions)
    assert solver._normalize_answer(result.answer) == solver._normalize_answer(expected)

@patch('langchain_ollama.OllamaLLM')
def test_error_handling(mock_ollama, sample_agents):
    mock_client = MagicMock()
    mock_client.invoke.side_effect = Exception("API Error")
    mock_ollama.return_value = mock_client
    
    solver = MathProblemSolver(sample_agents)
    solution = solver.get_solution(sample_agents[0], "test")
    
    assert solution.answer == "Error"
    assert solution.confidence == 0.0  # Should remain 0.0 for error cases





@pytest.mark.parametrize('solutions,expected', [
    (
        [
            Solution('A', 'x=5', '', 0.49),
            Solution('B', '5', '', 0.51)
        ],
        'No consensus'
    ),
    (
        [
            Solution('A', 'x=5', '', 0.4),
            Solution('B', '5', '', 0.3)
        ],
        'No consensus'
    ),
    (
        [
            Solution('A', 'Invalid', '', 0.8),
            Solution('B', 'x=5', '', 0.8)
        ],
        'No consensus'
    ),
    (
        [
            Solution('A', 'Error', '', 0.0),
            Solution('B', 'Error', '', 0.0)
        ],
        'Error'
    )
])
def test_confidence_threshold_validation(solutions, expected):
    solver = MathProblemSolver([])
    result = solver.vote_on_solutions(solutions)
    assert result.answer == expected




@pytest.mark.parametrize('solutions,expected', [
    (
        [
            Solution('A', 'x=5', '', 0.49),
            Solution('B', '5', '', 0.51)
        ],
        '5'
    ),
    (
        [
            Solution('A', 'x=5', '', 0.8),
            Solution('B', 'x = 5', '', 0.7),
            Solution('C', 'five', '', 0.6)
        ],
        'x=5'
    )
])
def test_voting_thresholds(solutions, expected):
    solver = MathProblemSolver([])
    result = solver.vote_on_solutions(solutions)
    assert solver._normalize_answer(result.answer) == solver._normalize_answer(expected)