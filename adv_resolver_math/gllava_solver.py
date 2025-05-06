"""
GLLaVASolver Module

Provides a solver interface for G-LLaVA models to solve geometric problems with diagrams, supporting both Ollama and LM Studio backends.
"""
import os
from typing import Dict, Any, Optional

class GLLaVASolver:
    """
    Solver that uses the G-LLaVA model for solving geometric problems with diagrams.
    Supports both Ollama and LM Studio as backend model servers.
    """
    
    def __init__(self, use_ollama: bool = True, api_url: Optional[str] = None):
        """
        Initialize the G-LLaVA solver.

        Args:
            use_ollama (bool): Whether to use Ollama (True) or LM Studio (False).
            api_url (str, optional): API URL for the model server. Defaults to Ollama's endpoint if not provided.
        """
        self.use_ollama = use_ollama
        self.api_url = api_url or "http://localhost:11434/api/generate"
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve a geometric problem using the G-LLaVA model.

        Args:
            problem (dict): Dictionary containing 'text' and optionally 'image_path'.
        Returns:
            dict: Dictionary with solution text, model used, confidence score, and metadata.
        """
        if self.use_ollama:
            return self._solve_with_ollama(problem)
        else:
            return self._solve_with_lmstudio(problem)
    
    def _solve_with_ollama(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve a geometric problem using the Ollama G-LLaVA API.

        Args:
            problem (dict): Problem input with 'text' and optionally 'image_path'.
        Returns:
            dict: Solution and metadata from Ollama API.
        """
        import requests
        import json
        prompt = problem['text']
        image_path = problem.get('image_path')
        payload = {
            "model": "gllava",
            "prompt": prompt,
            "stream": False
        }
        if image_path and os.path.exists(image_path):
            import base64
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                payload["images"] = [encoded_image]
        response = requests.post(self.api_url, json=payload)
        result = response.json()
        return {
            "solution": result.get("response", ""),
            "model_used": "G-LLaVA (Ollama)",
            "confidence": 0.9,
            "metadata": result
        }
    
    def _solve_with_lmstudio(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve a geometric problem using the LM Studio API.

        Args:
            problem (dict): Problem input with 'text' and optionally 'image_path'.
        Returns:
            dict: Solution and metadata from LM Studio API (currently stub).
        """
        # Implementation for LM Studio API
        # To be implemented based on LM Studio API specifics
        return {"solution": "", "model_used": "G-LLaVA (LM Studio)", "confidence": 0.0, "metadata": {}}
