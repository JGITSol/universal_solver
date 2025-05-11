# Universal Solver

[![Test Coverage](./coverage.svg)](./htmlcov/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

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

Installation
Prerequisites
Python 3.8+
pip
Clone and Prepare
sh
CopyInsert
git clone <repo-url> universal_solver
cd universal_solver
Setup Environments
For Advanced Math Ensemble
sh
CopyInsert
cd adv_resolver_math
python -m venv adv_res_venv
adv_res_venv\Scripts\activate  # On Windows
pip install -r requirements.txt
For KAN Module
sh
CopyInsert
cd ../KAN
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Install Core Dependencies
Alternatively, install all core dependencies in the project root:

sh
CopyInsert
pip install -r requirements.txt
Or for development:

sh
CopyInsert
pip install -e .[dev]
Configuration
Create a .env file in the project root to store API keys and configuration parameters for external services (e.g., OpenAI, Gemini, etc.).

Usage
CLI
Run benchmarks or solve problems via the command line:

sh
CopyInsert
python benchmark_cli.py --help
GUI
Launch the graphical interface:

sh
CopyInsert
python universal_solver_gui.py
Jupyter/Colab
Use collaborative notebooks in collab_training_ntbks/ or benchmark_showcase_colab.ipynb for cloud-based workflows.

Example Workflow
Select a math problem or benchmark dataset.
Choose solver options (ensemble, symbolic, neural, etc.).
Process the problem and review intermediate logs and results.
Use voting and debugging panels (GUI) for transparency and inspection.
Export results and reports as needed.
Testing & Quality Assurance
Run all tests:
sh
CopyInsert
pytest --cov=adv_resolver_math --cov-report=html
Code is formatted with black, linted with flake8, and type-checked with mypy.
Dependencies
Core dependencies include:

numpy, sympy, torch, scikit-learn, sentence-transformers, transformers, pandas, matplotlib, seaborn, plotly, langchain, pykan, customtkinter, rich, requests, and more.
Full list in requirements.txt and setup.py.

License
MIT License





