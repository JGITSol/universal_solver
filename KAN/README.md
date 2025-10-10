KAN module
===========

This package contains a small demo showing symbolic discovery for a Kepler
example. The code prefers the external package `pykan` as the canonical
implementation of Kolmogorov-Arnold Networks. If `pykan` is not installed,
the module will try to import `kan`. When neither package is available a
deterministic `DummyKAN` fallback is used so tests and CI remain reliable.

Quick notes
-----------
- Preferred package: `pykan` (install with pip install pykan).
- Optional: `kan` (legacy name, used if `pykan` not found).
- Fallback: `DummyKAN` (used automatically when no external package
  is present). This is intended for tests and for local development when
  you don't need the actual training routines.

Running tests
-------------
From the repository root run:

    python -m pytest tests/test_simple_symbolic_regression.py -q

If you want the real implementation, install `pykan` into your venv before
running the demo or tests.
# KAN Module

## Overview
Kolmogorov-Arnold Networks (KAN) for symbolic regression and interpretable machine learning.

## Structure
- `SimpleSymbolicRegressionProject.py`: Main entry for symbolic regression experiments.
- `requirements.txt`: Dependencies for KAN experiments.

## Usage
1. Activate the relevant Python environment.
2. Install requirements:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the symbolic regression demo:
   ```sh
   python SimpleSymbolicRegressionProject.py
   ```

## Testing
- Tests for KAN integration are in the top-level `tests/` directory.
- Run all tests from the project root:
   ```sh
   pytest --cov=KAN --cov-report=html
   ```

## Documentation
- See the main project `docs/` for guides and technical details.

## Contact
For questions, see the main project README or open an issue.
