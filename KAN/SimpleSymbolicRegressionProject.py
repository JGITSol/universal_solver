"""Simple KAN demo: Kepler's law discovery pipeline.

This module is written defensively: if an external `kan` package isn't
available on PYTHONPATH it falls back to a lightweight `DummyKAN` that
implements the subset of the API the demo expects. That makes the demo
runnable in CI/tests without requiring the external dependency.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple, Protocol, runtime_checkable, cast

import numpy as np
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

# Force a non-interactive backend for CI / headless environments.
plt.switch_backend("Agg")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


External_KAN = None
try:
    # prefer the modern package name if available
    import pykan as _pykan  # type: ignore

    External_KAN = getattr(_pykan, "KAN", None) or getattr(_pykan, "MultKAN", None)
except Exception:
    try:
        import kan as _kan  # type: ignore

        External_KAN = getattr(_kan, "KAN", None) or getattr(_kan, "MultKAN", None)
    except Exception:
        External_KAN = None


class DummyKAN:
    """A tiny, deterministic fallback KAN implementation used for tests.

    It provides the minimal API the demo expects: train, visualize,
    sparsify, prune, to_symbolic and __call__.
    """

    def __init__(self, width=None, grid: int = 5, k: int = 3):
        self.width = width
        self.grid = grid
        self.k = k

    def train(self, x: torch.Tensor, y: torch.Tensor, steps: int = 100):
        # Dummy training - compute a simple least-squares linear fit
        X = x.numpy().reshape(-1, 1)
        Y = y.numpy().reshape(-1, 1)
        A = np.concatenate([X, np.ones_like(X)], axis=1)
        self.coef_, *_ = np.linalg.lstsq(A, Y, rcond=None)

    def visualize(self) -> None:
        logger.debug("DummyKAN.visualize() called")

    def sparsify(self, regularization: float = 1e-3) -> None:
        logger.debug("DummyKAN.sparsify() called")

    def prune(self, threshold: float = 0.01) -> None:
        logger.debug("DummyKAN.prune() called")

    def to_symbolic(self) -> str:
        # Use scalar extraction with .ravel() to avoid numpy deprecation
        if hasattr(self, "coef_"):
            flat = np.asarray(self.coef_).ravel()
            f0 = flat[0]
            f1 = flat[1]
            a = float(f0.item() if hasattr(f0, "item") else f0)
            b = float(f1.item() if hasattr(f1, "item") else f1)
            return f"{a:.3g} * x + {b:.3g}"
        return "C0"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        X = x.numpy().reshape(-1, 1)
        if hasattr(self, "coef_"):
            flat = np.asarray(self.coef_).ravel()
            f0 = flat[0]
            f1 = flat[1]
            a = float(f0.item() if hasattr(f0, "item") else f0)
            b = float(f1.item() if hasattr(f1, "item") else f1)
            y = X * a + b
            return torch.tensor(y, dtype=torch.float32)
        return torch.zeros_like(x).float()


@runtime_checkable
class KANProtocol(Protocol):
    """Typing protocol for KAN-like objects used by the demo.

    The protocol is intentionally minimal and permissive so both the
    real external implementations and the DummyKAN satisfy it.
    """

    def train(self, *args, **kwargs) -> None:  # pragma: no cover - protocol
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - protocol
        ...

    def to_symbolic(self) -> str:  # pragma: no cover - protocol
        ...

    def visualize(self) -> None:  # pragma: no cover - protocol
        ...

    def sparsify(
        self, regularization: float = 1e-3
    ) -> None:  # pragma: no cover - protocol
        ...

    def prune(self, threshold: float = 0.01) -> None:  # pragma: no cover - protocol
        ...


# Expose KAN name in this module: prefer the external implementation
KAN = External_KAN if External_KAN is not None else DummyKAN


def generate_kepler_data(
    n_samples: int = 1000, seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate noisy Kepler's third law dataset: T = sqrt(a^3) + noise.

    Returns (a, T) as torch tensors shaped (n_samples, 1) and (n_samples,).
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    a = np.random.uniform(0.3, 30, n_samples)
    T = np.sqrt(a**3) + np.random.normal(0, 0.1, n_samples)
    return torch.tensor(a, dtype=torch.float32).reshape(-1, 1), torch.tensor(
        T, dtype=torch.float32
    ).reshape(-1, 1)


def symbolic_discovery_pipeline(
    kan_instance: Optional[KANProtocol] = None,
    *,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the discovery pipeline and return (symbolic_formula, x_test, y_test, y_pred).

    If ``kan_instance`` is None a new KAN (or DummyKAN) will be created.
    """
    x_train, y_train = generate_kepler_data(n_samples=n_samples, seed=seed)

    if kan_instance is None:
        kan_instance = KAN(width=[1, 3, 1], grid=5, k=3)
    # At this point we have a usable instance; help the type checker
    # understand that `kan_instance` follows the KANProtocol interface.
    kan_instance = cast(KANProtocol, kan_instance)

    # Train and refine
    # Call train with the most likely signatures supported by various KAN
    # implementations. Try multiple calling conventions so this demo works
    # whether the external package is present or we use DummyKAN.
    # TensorDataset was imported at module level to satisfy lint rules

    trained = False
    try:
        # try the common signature first
        kan_instance.train(x_train, y_train, steps=100)  # type: ignore[arg-type]
        trained = True
    except TypeError:
        logger.debug("KAN.train() didn't accept (x, y, steps), trying other signatures")

    if not trained:
        try:
            kan_instance.train(x_train, y_train)  # type: ignore[arg-type]
            trained = True
        except TypeError:
            logger.debug(
                "KAN.train() didn't accept (x, y), trying train() without args"
            )

    if not trained:
        try:
            # some implementations accept no-arg train()
            kan_instance.train()  # type: ignore[arg-type]
            trained = True
        except TypeError:
            logger.debug(
                "KAN.train() didn't accept empty call, trying single-tuple dataset"
            )

    if not trained:
        try:
            # some accept a single tuple argument
            kan_instance.train((x_train, y_train))  # type: ignore[arg-type]
            trained = True
        except TypeError:
            logger.debug("KAN.train() didn't accept single tuple, trying TensorDataset")

    if not trained:
        try:
            ds = TensorDataset(x_train, y_train)
            kan_instance.train(ds)  # type: ignore[arg-type]
            trained = True
        except Exception as exc:  # noqa: BLE001 - we log and re-raise
            logger.exception(
                "Failed to call kan.train() with known signatures: %s", exc
            )
            raise
    if hasattr(kan_instance, "visualize"):
        try:
            kan_instance.visualize()
        except Exception:
            logger.debug("kan.visualize() raised an exception, continuing")

    if hasattr(kan_instance, "sparsify"):
        try:
            kan_instance.sparsify(regularization=1e-3)
        except Exception:
            logger.debug("kan.sparsify() raised an exception, continuing")

    if hasattr(kan_instance, "prune"):
        try:
            kan_instance.prune(threshold=0.01)
        except Exception:
            logger.debug("kan.prune() raised an exception, continuing")

    if hasattr(kan_instance, "to_symbolic"):
        try:
            symbolic_formula = kan_instance.to_symbolic()
        except Exception:
            logger.debug(
                "kan.to_symbolic() raised an exception, using fallback representation"
            )
            symbolic_formula = repr(kan_instance)
    else:
        symbolic_formula = repr(kan_instance)

    logger.info("Discovered formula: %s", symbolic_formula)

    x_test = torch.linspace(0.3, 30, 100).reshape(-1, 1)
    y_test = torch.sqrt(x_test**3)
    y_pred = kan_instance(x_test)

    # Produce a plot for manual inspection when running locally
    try:
        plt.figure(figsize=(8, 5))
        plt.scatter(x_train.numpy(), y_train.numpy(), alpha=0.3, label="Training data")
        plt.plot(x_test.numpy(), y_test.numpy(), "r-", label="Ground truth")
        plt.plot(x_test.numpy(), y_pred.detach().numpy(), "g--", label="KAN prediction")
        plt.legend()
        plt.xlabel("Semi-major axis (AU)")
        plt.ylabel("Orbital period (years)")
        plt.title(f"Discovered formula: {symbolic_formula}")
        out = os.environ.get("KEPLER_OUTPUT", "kepler_law_discovery.png")
        plt.savefig(out)
        plt.close()
    except Exception as exc:
        logger.debug("Plotting failed: %s", exc)

    return symbolic_formula, x_test, y_test, y_pred


if __name__ == "__main__":
    # Run with small dataset for quick local runs
    symbolic_discovery_pipeline(n_samples=500)
