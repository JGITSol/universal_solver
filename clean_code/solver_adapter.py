from typing import Any, Dict, List
from clean_code.logger import get_logger

logger = get_logger(__name__)

class SolverAdapter:
    """
    Adapter to unify the interface of symbolic regression (KAN) and ensemble math solvers
    for use in the neuro-symbolic system.
    """
    def __init__(self, solver_instance, solver_type: str = "ensemble"):
        self.solver = solver_instance
        self.solver_type = solver_type
        logger.info(f"Initialized SolverAdapter with solver_type={solver_type}")

    def solve(self, problem: str, **kwargs) -> Dict[str, Any]:
        """
        Solve a problem using the underlying solver.
        Args:
            problem: The problem statement (str)
            kwargs: Additional arguments (e.g., context, agents, data)
        Returns:
            Dict with keys: 'answer', 'confidence', etc.
        """
        logger.info(f"Solving problem: {problem}")
        if self.solver_type == "ensemble":
            # Ensemble solver expects agents and uses get_solution/vote_on_solutions
            solutions = []
            for agent in self.solver.agents:
                sol = self.solver.get_solution(agent, problem)
                solutions.append(sol)
            result = self.solver.vote_on_solutions(solutions)
            explanations = [s.explanation for s in solutions]
            return {
                "answer": result.answer,
                "confidence": result.confidence,
                "agents_in_agreement": result.agents_in_agreement,
                "explanation": explanations
            }
        elif self.solver_type == "symbolic_regression":
            # KAN or similar expects data, not a text problem
            if "x_train" in kwargs and "y_train" in kwargs:
                self.solver.train(kwargs["x_train"], kwargs["y_train"], steps=kwargs.get("steps", 500))
                symbolic_formula = self.solver.to_symbolic()
                return {"formula": symbolic_formula}
            else:
                raise ValueError("Symbolic regression requires x_train and y_train.")
        else:
            raise NotImplementedError("Unknown solver type")

    def get_agents(self) -> List[str]:
        if hasattr(self.solver, "agents"):
            return [a.name for a in self.solver.agents]
        return []
