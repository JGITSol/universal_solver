import pytest
from unittest.mock import MagicMock
from clean_code.solver_adapter import SolverAdapter
from clean_code.neuro_symbolic_system import NeuroSymbolicMathSystem
from clean_code.visualization import MathVisualizer
from clean_code.tool_integration import ToolIntegrationManager
from clean_code.knowledge_management import KnowledgeManagementSystem

class DummySolver:
    def __init__(self):
        self.agents = [MagicMock(name='AgentA'), MagicMock(name='AgentB')]
        for a in self.agents:
            a.name = a._mock_name
    def get_solution(self, agent, problem):
        sol = MagicMock()
        sol.answer = "42"
        sol.confidence = 1.0
        sol.explanation = "Mocked explanation"
        sol.agent_name = agent.name
        return sol
    def vote_on_solutions(self, solutions):
        res = MagicMock()
        res.answer = "42"
        res.confidence = 1.0
        res.agents_in_agreement = [s.agent_name for s in solutions]
        return res

def test_solver_adapter_and_system():
    dummy_solver = DummySolver()
    adapter = SolverAdapter(dummy_solver, solver_type="ensemble")
    visualizer = MathVisualizer()
    tool_manager = ToolIntegrationManager()
    import tempfile, os
    tf = tempfile.NamedTemporaryFile(delete=False)
    tf.close()  # Close so KnowledgeManagementSystem can open it on Windows
    try:
        knowledge_manager = KnowledgeManagementSystem(db_path=tf.name)
        system = NeuroSymbolicMathSystem(adapter, visualizer, tool_manager, knowledge_manager)
        result = system.solve_problem("Mock problem", visualize=False)
        assert result["answer"] == "42"
        assert result["confidence"] == 1.0
        assert "Mocked explanation" in str(result["explanation"])
    finally:
        os.unlink(tf.name)
