"""
Solver Registry Module

Provides a registry function for instantiating and accessing available solver classes.
"""
from .gllava_solver import GLLaVASolver

def register_solvers():
    """
    Register and instantiate available solver classes.

    Returns:
        dict: Dictionary mapping solver names to solver instances.
    """
    solvers = {
        # Existing solvers can be added here
        "gllava_ollama": GLLaVASolver(use_ollama=True),
        "gllava_lmstudio": GLLaVASolver(use_ollama=False, api_url="http://localhost:8080/v1/completions")
    }
    return solvers
