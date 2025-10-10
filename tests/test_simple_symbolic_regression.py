import os

from KAN.SimpleSymbolicRegressionProject import symbolic_discovery_pipeline


def test_symbolic_discovery_pipeline_runs_quickly(tmp_path):
    # Ensure output file goes to temporary directory to avoid repo noise
    out = tmp_path / "out.png"
    os.environ["KEPLER_OUTPUT"] = str(out)

    formula, x_test, y_test, y_pred = symbolic_discovery_pipeline(n_samples=50, seed=42)

    assert isinstance(formula, str)
    # shapes: x_test (100,1), y_test (100,1), y_pred (100,1)
    assert x_test.shape[1] == 1
    assert y_test.shape == y_pred.shape
    # predictions should be finite
    assert y_pred.numel() > 0
    assert not (y_pred != y_pred).any()
