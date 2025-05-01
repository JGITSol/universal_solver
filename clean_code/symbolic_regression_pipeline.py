import torch
import numpy as np
from typing import Tuple

class SymbolicRegressionPipeline:
    """
    Pipeline for symbolic regression using KAN or similar models.
    """
    def __init__(self, kan_model):
        self.kan = kan_model

    def generate_kepler_data(self, n_samples=1000) -> Tuple[torch.Tensor, torch.Tensor]:
        a = np.random.uniform(0.3, 30, n_samples)
        T = np.sqrt(a**3) + np.random.normal(0, 0.1, n_samples)
        return torch.tensor(a, dtype=torch.float32).reshape(-1, 1), torch.tensor(T, dtype=torch.float32)

    def run(self, steps=500):
        x_train, y_train = self.generate_kepler_data()
        x_test = torch.linspace(0.3, 30, 100).reshape(-1, 1)
        y_test = torch.sqrt(x_test**3)
        dataset = {
            'train_input': x_train,
            'train_label': y_train,
            'test_input': x_test,
            'test_label': y_test
        }
        self.kan.fit(dataset, steps=steps)
        self.kan.auto_symbolic()
        symbolic_formula, variables = self.kan.symbolic_formula()
        x_test = torch.linspace(0.3, 30, 100).reshape(-1, 1)
        y_test = torch.sqrt(x_test**3)
        y_pred = self.kan(x_test)
        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "y_pred": y_pred.detach().numpy(),
            "formula": symbolic_formula
        }
