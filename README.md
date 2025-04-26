# Universal Solver

[![Build Status](https://github.com/<your-org>/universal_solver/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-org>/universal_solver/actions)
[![Coverage Status](https://codecov.io/gh/<your-org>/universal_solver/branch/main/graph/badge.svg)](https://codecov.io/gh/<your-org>/universal_solver)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-NONE-lightgrey.svg)](#license)


---

Universal Solver is a modular, extensible platform for advanced mathematical problem solving, symbolic regression, and AI-driven research workflows. It integrates state-of-the-art models, ensemble methods, and collaborative tools to accelerate research and innovation.


## Features

- **Advanced Math Ensemble Solver**: Combines LangChain, Ollama, OpenRouter, Google Gemini, SymPy, and more for powerful symbolic math and regression.
- **SOTA Model Integration**: Benchmark with OpenRouter (Llama 3.1 Nemotron Ultra, Llama 4 Maverick, MAI DS R1), Google Gemini (2.5 Flash, 2.0 Flash-Lite), and local Ollama models.
- **Industry-Standard Math Benchmarking**: Supports MATH, GSM8K, MathQA, ASDiv, SVAMP, AQUA-RAT, MiniF2F, and more via HuggingFace Datasets.
- **CLI & Colab/Cloud Ready**: Run large-scale benchmarks via CLI or Colab/Jupyter notebook, with export to Excel/Parquet and cloud upload (GCP, Azure, Kaggle).
- **Collaborative Notebooks**: Jupyter/Colab notebooks for collaborative model training and benchmarking.
- **Extensible Architecture**: Easily add models, solvers, and workflows.
- **Modern Python Tooling**: Linting, formatting, type-checking, and full test coverage.

## Project Structure

```text
├── adv_resolver_math/         # Advanced math ensemble solver (LangChain, Ollama, OpenRouter, Gemini, SymPy, etc.)
├── KAN/                      # Symbolic regression with KAN (Kolmogorov-Arnold Networks)
├── benchmark_datasets.py      # Loader for all standard math benchmarks (MATH, GSM8K, MathQA, etc.)
├── benchmark_cli.py           # CLI for running solver benchmarks on datasets
├── benchmark_showcase_colab.ipynb # Colab/cloud notebook for benchmarking and sharing
├── collab_training_ntbks/     # Collaborative model training notebooks
├── docs/                     # Documentation, guides, and testing
├── model/                    # Model state, configs, and history
├── project_guidelines/       # Hackathon and project guidelines, agent specs
├── math_cache/               # Exported math data and cache
├── htmlcov/                  # Test coverage reports
├── ...
```

## Quickstart

### 1. Clone and Prepare
```sh
git clone <repo-url> universal_solver
cd universal_solver
```

### 2. Setup Environments
#### For Advanced Math Ensemble
```sh
cd adv_resolver_math
python -m venv adv_res_venv
adv_res_venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```
#### For KAN Module
```sh
cd ../KAN
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the project root with:
```
OPENROUTER_API_KEY=...
GEMINI_API_KEY=...
```

## Usage

### Benchmarking CLI
Run a benchmark on any supported dataset:
```sh
python benchmark_cli.py --dataset gsm8k --sample-size 20 --out-dir showcase_results
```

### Colab/Cloud Notebook
Upload and run `benchmark_showcase_colab.ipynb` in Google Colab, Kaggle, or Jupyter. Select dataset, sample size, and run all solvers. Example code for uploading results to GCP, Azure, or Kaggle included.

### Python API
Import and use solvers, datasets, and benchmarking in your own code:
```python
from benchmark_datasets import load_benchmark_dataset, get_problem_and_answer
from showcase_advanced_math import agents, solvers

ds = load_benchmark_dataset("gsm8k", sample_size=5)
for ex in ds:
    problem, answer = get_problem_and_answer(ex, "gsm8k")
    # ... run solvers ...
```

## Testing & Coverage
Run tests and generate a coverage report:
```sh
pytest --cov=adv_resolver_math
```
HTML coverage report will be output to `htmlcov/`.

## Linting & Type Checking
```sh
black . --check
isort . --check-only
flake8 .
mypy .
```

## Contributing
Pull requests are welcome! Please follow the project guidelines in `project_guidelines/` and ensure all tests and linters pass before submitting.

## License
**No license file present.** This project is currently proprietary. If you wish to use this code, please contact the maintainers.

## Contact
For questions or collaborations, open an issue or contact the maintainers via GitHub.

---

> _Accelerating research with modular, collaborative, and AI-driven math tools._

pip install -r requirements.txt
```

### 3. Run Example
- Advanced Math Ensemble:

```sh
cd adv_resolver_math
python math_ensemble_langchain_ollama.py
```

- KAN Symbolic Regression:

```sh
cd KAN
python SimpleSymbolicRegressionProject.py
```

### 4. Generate Test Coverage

```sh
pytest --cov=adv_resolver_math --cov-report=html
```

Open `htmlcov/index.html` in your browser to view the coverage report.

## Installation
Install via pip:
```sh
pip install .
```

## CLI Usage
Once installed, run:
```sh
usolve rstar "x = 2 + 2"
```
Results will be printed as JSON.

## Contribution Guidelines
- See `project_guidelines/` for hackathon rules, agent specs, and memory module details.
- Notebooks for collaborative training are in `collab_training_ntbks/`.
- All code should follow the structure above for clarity and maintainability.

## Documentation
- See `docs/` for detailed guides and testing instructions.
- Each module contains its own `README.md` for specifics.

## License
MIT

## Acknowledgements
- LangChain, Ollama, SymPy, KAN, and all open-source contributors.

---
For hackathon entry: This repository is organized for rapid onboarding, modular development, and reproducible research. Happy hacking!
