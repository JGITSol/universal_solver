# Universal Solver: Advanced Project Overview

---

## Introduction

Universal Solver is a modular, extensible research platform for advanced mathematical reasoning, symbolic regression, and AI-driven problem-solving. It integrates state-of-the-art models, ensemble methods, and collaborative tools to accelerate research and innovation across mathematics, vision, and AI domains.

---

## Modular Architecture & Directory Structure

- **`adv_resolver_math/`**: Core advanced math ensemble solvers, including registry, agent definitions, and ensemble iterations.
- **`adv_resolver_math/ensemble_iterations/`**: Experimental and SOTA ensemble solver variants (see below).
- **`KAN/`**: Symbolic regression with Kolmogorov-Arnold Networks.
- **`benchmark_datasets.py`**: Loader for standard math benchmarks (MATH, GSM8K, MathQA, etc.).
- **`benchmark_cli.py`**: CLI for running solver benchmarks.
- **`collab_training_ntbks/`**: Collaborative model training notebooks.
- **`kaggle_vision_training/`**: Vision-language model fine-tuning procedures (e.g., G-LLaVA + Unsloth).
- **`tests/`**: All tests (moved here for proper pytest discovery and import resolution).
- **`project_guidelines/`**: Guidelines, agent specs, and hackathon docs.
- **`model/`**, **`math_cache/`**, **`showcase_results/`**: Model states, cache, and output results.

---

## Ensemble Iterations: Advanced Solver Modules

The `ensemble_iterations` package contains several next-generation ensemble solver architectures, each building on the previous for greater capability, modularity, and explainability.

### 1. `EnhancedMathSolver`
- **Semantic Clustering**: Groups solutions by semantic similarity (using sentence embeddings) before voting.
- **Performance-Weighted Voting**: Weights agent votes by historical correctness.
- **Modular SOTA Features**: Easily extendable with new voting or clustering strategies.

### 2. `MemorySharingMathSolver`
- **Shared Memory Layer**: Agents share a vector-based memory (multi-head attention) for improved knowledge distillation.
- **Knowledge Aggregation**: Aggregates agent memories before consensus, allowing for richer collaboration.
- **Torch-based Implementation**: Uses PyTorch for efficient memory operations.

### 3. `LatentSpaceMathSolver`
- **Latent Reasoning**: Implements "Chain of Continuous Thought" (CoCoT/Coconut) style reasoning in latent space.
- **Continuous Thought Steps**: Agents refine their reasoning iteratively in embedding space, not just text.
- **Latent Voting**: Finds consensus in vector space, not just via answer text.

### 4. `RStarMathSolver`
- **Symbolic & Code Verification**: Each solution step can be verified symbolically (SymPy) and via code execution.
- **Process Reward Model**: Rewards solutions based on step coherence, conceptual consistency, and computational efficiency.
- **MCTS-Inspired Exploration**: Uses Monte Carlo Tree Search-style rounds for solution evolution and selection.
- **Highly Modular**: Designed for rapid experimentation with new reward models, verification methods, and search strategies.

---

## Ensemble Workflow

1. **Agent Generation**: Multiple agent models generate candidate solutions for a given math problem.
2. **Solution Embedding**: Solutions are embedded for semantic comparison and memory sharing.
3. **Voting & Consensus**: Ensemble modules cluster, weigh, and vote on solutions using advanced strategies (see above).
4. **Verification & Reward**: Solutions are verified for correctness and scored for process quality.
5. **Benchmarking**: Results are logged, compared, and visualized using standardized benchmarks and tools.

---

## Vision & Multimodal Integration

- **`kaggle_vision_training/`**: Contains procedures for fine-tuning large vision-language models (e.g., G-LLaVA with Unsloth).
- **Integration**: Vision solvers can be registered in the same ensemble framework for cross-modal benchmarking.

---

## Testing & Best Practices

- **All tests** are located in the `tests/` directory for compatibility with `pytest` and correct import resolution.
- **Imports**: All tests import from `adv_resolver_math` and submodules as packages.
- **Test Runner**: Run all tests from the project root for proper `PYTHONPATH` handling.
- **.env**: API keys and sensitive configuration are managed via `.env` in the root directory.

---

## Extending the Project

- **Add new solvers**: Create a new module in `adv_resolver_math/` or `ensemble_iterations/` and register it in `solver_registry.py`.
- **Expand ensemble logic**: Add new voting, clustering, or verification strategies by subclassing the ensemble base classes.
- **Integrate new datasets**: Add loaders in `benchmark_datasets.py` and update CLI/notebooks as needed.
- **Vision/Multimodal**: Add new training procedures or integrate new models in `kaggle_vision_training/`.
- **Document**: Update markdown guides for new procedures, models, or benchmarks.

---

## Example Usage

### CLI Benchmarking
```sh
python benchmark_cli.py --dataset gsm8k --sample-size 20 --out-dir showcase_results
```

### Registering a New Solver
- Implement your solver in `adv_resolver_math/ensemble_iterations/your_solver.py`
- Register it in `adv_resolver_math/solver_registry.py`:
```python
from adv_resolver_math.ensemble_iterations.your_solver import YourSolver
solver_registry.register('your_solver', YourSolver)
```

---

## References & Further Reading
- [G-LLaVA: Large Language and Vision Assistant](https://github.com/haotian-liu/LLaVA)
- [Unsloth: Efficient LLM Fine-Tuning](https://github.com/unslothai/unsloth)
- [SymPy: Symbolic Mathematics](https://www.sympy.org/en/index.html)
- [LangChain](https://python.langchain.com/)

---

## Contact & Contribution

- Please see `project_guidelines/` for contribution rules, agent specs, and hackathon instructions.
- Issues and PRs welcome!

---

This document provides a comprehensive, up-to-date description of the Universal Solver project and its advanced ensemble architecture. For further details or to contribute, see the README, project guidelines, or contact the maintainers.
