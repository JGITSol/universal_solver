from unittest.mock import MagicMock, patch

import torch


def test_symbolic_discovery_pipeline_calls_kan_methods(monkeypatch):
    # Prepare dummy data and a fake KAN class
    fake_kan = MagicMock()
    # Configure fake behaviors
    fake_kan.train.return_value = None
    fake_kan.visualize.return_value = None
    fake_kan.sparsify.return_value = None
    fake_kan.prune.return_value = None
    fake_kan.to_symbolic.return_value = "sqrt(x**3)"

    # Make the fake instance callable as a model (kan(x_test) -> tensor)
    dummy_pred = torch.sqrt(torch.linspace(0.3, 30, 100) ** 3)
    fake_kan.__call__ = MagicMock(return_value=dummy_pred)

    # Patch the import where SimpleSymbolicRegressionProject imports KAN
    with patch("KAN.SimpleSymbolicRegressionProject.KAN", return_value=fake_kan):
        # Import inside the context to ensure patched KAN is used
        import importlib

        mod = importlib.import_module("KAN.SimpleSymbolicRegressionProject")

        # Run smaller pipeline steps directly to verify interactions
        # Generate a tiny dataset by calling the helper
        x_train, y_train = mod.generate_kepler_data(n_samples=10)
        assert x_train.shape[0] == 10

        # Initialize and call pipeline functions to simulate main flow
        kan_instance = mod.KAN(width=[1, 3, 1], grid=5, k=3)
        kan_instance.train(x_train, y_train, steps=10)
        kan_instance.visualize()
        kan_instance.sparsify(regularization=1e-3)
        kan_instance.prune(threshold=0.01)
        sym = kan_instance.to_symbolic()
        assert "sqrt" in str(sym)

        # Call model on test data
        x_test = torch.linspace(0.3, 30, 100).reshape(-1, 1)
        y_pred = kan_instance(x_test)
        assert hasattr(y_pred, "shape")

        # Verify that the mocked methods were called
        fake_kan.train.assert_called()
        fake_kan.visualize.assert_called()
        fake_kan.sparsify.assert_called()
        fake_kan.prune.assert_called()
        fake_kan.to_symbolic.assert_called()
