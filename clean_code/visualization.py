import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional
from clean_code.logger import get_logger

logger = get_logger(__name__)

class MathVisualizer:
    """
    Advanced visualization system for mathematical proofs, constructions, and symbolic regression.
    """
    def __init__(self, figsize=(10, 8), style="modern", interactive=True):
        self.figsize = figsize
        self.style = style
        self.interactive = interactive
        self.colors = self._get_color_scheme(style)
        plt.rcParams["figure.figsize"] = figsize
        if style == "modern":
            plt.style.use("seaborn-v0_8-whitegrid")
        elif style == "classic":
            plt.style.use("classic")
        elif style == "minimal":
            plt.style.use("seaborn-v0_8-white")

    def _get_color_scheme(self, style):
        if style == "modern":
            return {
                "points": "#3498db",
                "lines": "#2c3e50",
                "circles": "#e74c3c",
                "angles": "#9b59b6",
                "highlight": "#f1c40f",
                "background": "#ecf0f1"
            }
        elif style == "classic":
            return {
                "points": "blue",
                "lines": "black",
                "circles": "red",
                "angles": "green",
                "highlight": "orange",
                "background": "white"
            }
        else:
            return {
                "points": "#555555",
                "lines": "#333333",
                "circles": "#777777",
                "angles": "#999999",
                "highlight": "#000000",
                "background": "#ffffff"
            }

    def visualize_geometry_problem(self, formalization: Dict[str, Any], highlight_steps: Optional[List[Any]] = None):
        """
        Visualize a geometric problem from formalized data.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        entities = formalization.get("entities", [])
        # ... (drawing logic omitted for brevity)
        plt.show()
        logger.info("Geometry problem visualization completed")
        plt.savefig("geometry_problem_visualization.png")
        logger.info("Geometry problem visualization saved to geometry_problem_visualization.png")
        return fig

    def visualize_symbolic_regression(self, x_train, y_train, x_test, y_test, y_pred, formula: str):
        plt.figure(figsize=self.figsize)
        plt.scatter(x_train, y_train, alpha=0.3, label='Training data')
        plt.plot(x_test, y_test, 'r-', label='Ground truth')
        plt.plot(x_test, y_pred, 'g--', label='Model prediction')
        plt.legend()
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title(f'Discovered formula: {formula}')
        logger.info(f"Visualizing discovered formula: {formula}")
        plt.show()
        plt.savefig("symbolic_regression_visualization.png")
        logger.info("Symbolic regression visualization saved to symbolic_regression_visualization.png")

    def visualize(self, problem, result):
        # Visualize based on problem type
        if isinstance(result, dict) and "formula" in result:
            plt.figure(figsize=(10, 6))
            plt.title(f"Discovered formula: {result['formula']}")
            logger.info(f"Visualizing discovered formula: {result['formula']}")
        else:
            plt.figure(figsize=(8, 5))
            plt.title("Math Problem Visualization")
            logger.info(f"Visualizing result for problem: {problem}")
        plt.savefig("visualization_result.png")
        logger.info("Visualization saved to visualization_result.png")
        plt.show()
