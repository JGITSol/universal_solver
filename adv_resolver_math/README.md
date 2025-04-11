# Advanced Math Ensemble Solver

This project implements a comprehensive math problem-solving system that integrates LangChain and Ollama for state-of-the-art mathematical reasoning. The system supports both sequential and parallel processing modes, implements industry-standard benchmarking, and integrates sophisticated symbolic math validation.

## Features

- Multiple model integration via Ollama (cogito:3b, gemma3, phi4-mini:latest)
- Sequential and parallel processing modes
- Meta-ensemble architecture for optimal strategy selection
- Symbolic validation with SymPy
- Comprehensive benchmarking against industry-standard datasets
- Performance visualization and analysis

## Setup

1. Create and activate the virtual environment:
   ```
   # Navigate to the project directory
   cd adv_resolver_math
   
   # Activate the virtual environment
   .\adv_res_venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```
   # Install dependencies using uv
   uv pip install -r requirements.txt
   ```

3. Install and run Ollama:
   - Download Ollama from [https://ollama.ai/](https://ollama.ai/)
   - Install and run the Ollama service
   - Pull required models: `ollama pull cogito:3b gemma3 phi4-mini:latest`

## Usage

Run the main script:
```
python math_ensemble_langchain_ollama.py
```

This will present a menu with options to:
1. Run a quick demo with sample problems
2. Run a sequential mode benchmark
3. Run a parallel mode benchmark
4. Run a meta-ensemble benchmark
5. Run all benchmarks (comprehensive)

## Project Structure

- `math_ensemble_langchain_ollama.py`: Main implementation file
- `requirements.txt`: Project dependencies
- `math_cache/`: Directory for caching solutions

## Requirements

- Python 3.11+
- Ollama service running locally
- Required Python packages (see requirements.txt)