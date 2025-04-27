from .gllava_solver import GLLaVASolver

def register_solvers():
    solvers = {
        # Existing solvers can be added here
        "gllava_ollama": GLLaVASolver(use_ollama=True),
        "gllava_lmstudio": GLLaVASolver(use_ollama=False, api_url="http://localhost:8080/v1/completions")
    }
    return solvers
