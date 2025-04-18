# Universal Solver: Hackathon Edition

## Overview
Universal Solver is a modular, extensible platform for advanced mathematical problem solving, symbolic regression, and AI-driven research workflows. It integrates state-of-the-art models, ensemble methods, and collaborative tools to accelerate research and innovation.

## Project Structure

```
├── adv_resolver_math/         # Advanced math ensemble solver (LangChain, Ollama, SymPy, etc.)
│   ├── math_ensemble_adv_ms_hackaton.py
│   ├── math_ensemble_langchain_ollama.py
│   ├── callbacks.py
│   ├── math_prompts.py
│   ├── requirements.txt
│   ├── README.md
│   └── ...
├── KAN/                      # Symbolic regression with KAN (Kolmogorov-Arnold Networks)
│   ├── SimpleSymbolicRegressionProject.py
│   ├── requirements.txt
│   └── ...
├── collab_training_ntbks/     # Collaborative model training notebooks
│   └── *.ipynb
├── docs/                     # Documentation, guides, and testing
│   ├── TESTING.md
│   └── ...
├── model/                    # Model state, configs, and history
├── project_guidelines/       # Hackathon and project guidelines, agent specs
├── math_cache/               # Exported math data and cache
├── htmlcov/                  # Test coverage reports
├── .gitignore
├── ADVANCED_RESOLVER_MATH.md # Deep dive into advanced resolver math
├── MODELS_INFO.md            # Info on integrated models
├── models_list.md            # List of available models
└── README.md                 # (This file)
```

## Quickstart

### 1. Clone and Prepare
```sh
git clone <repo-url> universal_solver
cd universal_solver
```

### 2. Setup Environments
- For advanced math ensemble:
  ```sh
  cd adv_resolver_math
  python -m venv adv_res_venv
  adv_res_venv\Scripts\activate  # On Windows
  pip install -r requirements.txt
  ```
- For KAN module:
  ```sh
  cd ../KAN
  python -m venv venv
  venv\Scripts\activate
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

## Contribution Guidelines
- See `project_guidelines/` for hackathon rules, agent specs, and memory module details.
- Notebooks for collaborative training are in `collab_training_ntbks/`.
- All code should follow the structure above for clarity and maintainability.

## Documentation
- See `docs/` for detailed guides and testing instructions.
- Each module contains its own `README.md` for specifics.

## License
Specify your license here.

## Acknowledgements
- LangChain, Ollama, SymPy, KAN, and all open-source contributors.

---
For hackathon entry: This repository is organized for rapid onboarding, modular development, and reproducible research. Happy hacking!
