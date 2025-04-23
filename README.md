# Universal Solver

[![Build Status](https://github.com/<your-org>/universal_solver/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-org>/universal_solver/actions)
[![Coverage Status](https://codecov.io/gh/<your-org>/universal_solver/branch/main/graph/badge.svg)](https://codecov.io/gh/<your-org>/universal_solver)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-NONE-lightgrey.svg)](#license)

---

Universal Solver is a modular, extensible platform for advanced mathematical problem solving, symbolic regression, and AI-driven research workflows. It integrates state-of-the-art models, ensemble methods, and collaborative tools to accelerate research and innovation.

---

## Features
- **Advanced Math Ensemble Solver**: Combines LangChain, Ollama, SymPy, and more for powerful symbolic math and regression.
- **Symbolic Regression (KAN)**: Kolmogorov-Arnold Networks for interpretable regression.
- **Collaborative Notebooks**: Jupyter notebooks for collaborative model training.
- **Extensible Architecture**: Easily add models, solvers, and workflows.
- **Modern Python Tooling**: Linting, formatting, type-checking, and full test coverage.

## Project Structure

```text
├── adv_resolver_math/         # Advanced math ensemble solver (LangChain, Ollama, SymPy, etc.)
├── KAN/                      # Symbolic regression with KAN (Kolmogorov-Arnold Networks)
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

## Installation (Editable Mode)
From the project root:
```sh
pip install -e .
```

## Usage
After installation, you can use the main CLI:
```sh
usolve --help
```
Or import modules in your Python code:
```python
from adv_resolver_math import math_ensemble_adv_ms_hackaton
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
