from typing import Any, Dict, Optional
from .solver_adapter import SolverAdapter
from .visualization import MathVisualizer
from .tool_integration import ToolIntegrationManager
from .knowledge_management import KnowledgeManagementSystem
from clean_code.logger import get_logger

logger = get_logger(__name__)

class NeuroSymbolicMathSystem:
    """
    Orchestrator for modular neuro-symbolic mathematical reasoning.
    Integrates solver, visualization, tool integration, and knowledge management.
    """
    def __init__(
        self,
        solver_adapter: SolverAdapter,
        visualizer: Optional[MathVisualizer] = None,
        tool_manager: Optional[ToolIntegrationManager] = None,
        knowledge_manager: Optional[KnowledgeManagementSystem] = None
    ):
        self.solver_adapter = solver_adapter
        self.visualizer = visualizer
        self.tool_manager = tool_manager
        self.knowledge_manager = knowledge_manager
        logger.info("NeuroSymbolicMathSystem initialized with provided components.")

    def solve_problem(self, problem: str, visualize: bool = False, **kwargs) -> Dict[str, Any]:
        logger.info(f"Solving problem: {problem}")
        result = self.solver_adapter.solve(problem, **kwargs)
        if visualize and self.visualizer:
            logger.info("Visualizing result.")
            if 'x_train' in result and 'y_train' in result:
                # Symbolic regression visualization
                self.visualizer.visualize_symbolic_regression(
                    result['x_train'], result['y_train'], result['x_test'], result['y_test'], result['y_pred'], result['formula']
                )
            elif 'formalization' in result:
                self.visualizer.visualize_geometry_problem(result['formalization'])
        if self.knowledge_manager:
            logger.info("Storing solution in knowledge manager.")
            self.knowledge_manager.store_solution(problem, result)
        logger.info(f"Problem solved: {problem}")
        return result

    def call_tool(self, tool_name: str, function_name: str, *args, **kwargs) -> Any:
        if not self.tool_manager:
            raise RuntimeError("Tool integration module not provided.")
        return self.tool_manager.call_tool(tool_name, function_name, *args, **kwargs)

    def query_knowledge(self, query_str: str):
        if not self.knowledge_manager:
            raise RuntimeError("Knowledge management module not provided.")
        return self.knowledge_manager.query(query_str)
