<!-- markdownlint-disable -->

# Universal Solver

<!-- CI/CD and Build Status -->
[![CI](https://github.com/JGITSol/universal_solver/actions/workflows/ci.yml/badge.svg)](https://github.com/JGITSol/universal_solver/actions/workflows/ci.yml)
[![Test Coverage](./coverage.svg)](./htmlcov/index.html)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

<!-- Language and Version -->
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- License -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- Repository Stats -->
[![GitHub stars](https://img.shields.io/github/stars/JGITSol/universal_solver?style=social)](https://github.com/JGITSol/universal_solver/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/JGITSol/universal_solver?style=social)](https://github.com/JGITSol/universal_solver/network/members)
[![GitHub watchers](https://img.shields.io/github/watchers/JGITSol/universal_solver?style=social)](https://github.com/JGITSol/universal_solver/watchers)

<!-- Activity -->
[![GitHub issues](https://img.shields.io/github/issues/JGITSol/universal_solver)](https://github.com/JGITSol/universal_solver/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/JGITSol/universal_solver)](https://github.com/JGITSol/universal_solver/pulls)
[![GitHub contributors](https://img.shields.io/github/contributors/JGITSol/universal_solver)](https://github.com/JGITSol/universal_solver/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/JGITSol/universal_solver)](https://github.com/JGITSol/universal_solver/commits)

<!-- Repository Info -->
[![GitHub repo size](https://img.shields.io/github/repo-size/JGITSol/universal_solver)](https://github.com/JGITSol/universal_solver)
[![GitHub language count](https://img.shields.io/github/languages/count/JGITSol/universal_solver)](https://github.com/JGITSol/universal_solver)
[![GitHub top language](https://img.shields.io/github/languages/top/JGITSol/universal_solver)](https://github.com/JGITSol/universal_solver)

---

Universal Solver is a modular, extensible platform for advanced mathematical problem solving, symbolic regression, and AI-driven research workflows. It integrates state-of-the-art models, ensemble methods, and collaborative tools to accelerate research and innovation in mathematics, science, and engineering.

## Key Features

- **Advanced Math Ensemble Solver**: Combines multiple state-of-the-art tools and models—including LangChain, Ollama, OpenRouter, Google Gemini, and SymPy—for symbolic mathematics and regression.
- **Industry-Standard Math Benchmarking**: Supports a wide range of math benchmarks (MATH, GSM8K, MathQA, ASDiv, SVAMP, AQUA-RAT, MiniF2F, and more) via HuggingFace Datasets.
- **Flexible Interfaces**: Provides both a Command-Line Interface (CLI) and a modern graphical user interface (GUI) built with CustomTkinter, as well as Jupyter/Colab notebook support for collaborative and cloud-based workflows.
- **Extensible Architecture**: Easily add new models, solvers, and research workflows with a plugin-friendly architecture.
- **Comprehensive Benchmarking and Reporting**: Run large-scale benchmarks, export results to Excel/Parquet, and upload to cloud storage (GCP, Azure, Kaggle).
- **Modern Python Tooling**: Fully type-checked, linted, and covered by automated tests. Includes development tools for formatting, linting, and static analysis.

## Project Structure

```text
adv_resolver_math/         # Advanced math ensemble solver (LangChain, Ollama, OpenRouter, Gemini, SymPy, etc.)
KAN/                      # Symbolic regression with Kolmogorov-Arnold Networks (KAN)
benchmark_datasets.py     # Loader for standard math benchmarks
benchmark_cli.py          # CLI for running solver benchmarks
benchmark_showcase_colab.ipynb # Colab/cloud notebook for benchmarking and sharing
collab_training_ntbks/    # Collaborative model training notebooks
docs/                     # Documentation, guides, and testing
model/                    # Model state, configs, and history
project_guidelines/       # Hackathon and project guidelines, agent specs
math_cache/               # Exported math data and cache
tests/                    # Test suite (pytest compatible)
universal_solver_gui.py   # Modern GUI for solver interaction
... (see [Project Overview](docs/UNIVERSAL_SOLVER_PROJECT_OVERVIEW.md))
```

## Installation

### Prerequisites

- Python 3.10+
- `pip`
- (Optional) `make` for development shortcuts

### Clone and Prepare

```bash
git clone <repo-url> universal_solver
cd universal_solver
```

### Environment Setup

Create isolated environments for the two primary subsystems to keep dependency trees manageable.

#### Advanced Math Ensemble

```powershell
cd adv_resolver_math
python -m venv adv_res_venv
adv_res_venv\Scripts\activate
pip install -r requirements.txt
```

#### KAN Module

```powershell
cd ..\KAN
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Project Root (optional unified environment)

```powershell
pip install -r requirements.txt
# or for contributors
pip install -e .[dev]
```

### Configuration

Create a `.env` file in the project root with API keys and configuration values for external services (OpenRouter, Gemini, WolframAlpha, etc.). See `docs/API_*` guides for details.

## Usage

### CLI

```powershell
python benchmark_cli.py --help
```

### GUI

```powershell
python universal_solver_gui.py
```

### Jupyter / Colab

Launch notebooks from `collab_training_ntbks/` or `benchmark_showcase_colab.ipynb` to explore workflows in an interactive environment.

### Example Workflow

1. Select a math problem or benchmark dataset.
2. Choose solver options (ensemble, symbolic, neural, etc.).
3. Execute the pipeline and inspect intermediary reasoning / verification logs.
4. Review consensus voting results and diagnostics.
5. Export reports as HTML, CSV, or Parquet for downstream analysis.

## Testing & Quality Assurance

```powershell
pytest --cov=adv_resolver_math --cov-report=html
```

- Formatting: `black .`
- Linting: `ruff check .`
- Type checking: `mypy .`

## Versioning & Releases

- Project version is tracked in the top-level `VERSION` file and exposed via `universal_solver.__version__`.
- Update the version file following [Semantic Versioning](https://semver.org/) for every release candidate.
- See `docs/RELEASE.md` for branching, tagging, and publishing instructions.

## Dependencies

Key dependencies include `numpy`, `sympy`, `torch`, `scikit-learn`, `sentence-transformers`, `transformers`, `langchain`, `pykan`, `customtkinter`, `rich`, and more. Refer to `requirements.txt` and `setup.py` for the exhaustive list.

## License

Distributed under the MIT License.










