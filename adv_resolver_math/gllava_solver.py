import os
from typing import Dict, Any, Optional

class GLLaVASolver:
    """Solver that uses G-LLaVA model for geometric problems with diagrams."""
    
    def __init__(self, use_ollama: bool = True, api_url: Optional[str] = None):
        """
        Initialize the G-LLaVA solver.
        Args:
            use_ollama: Whether to use Ollama (True) or LM Studio (False)
            api_url: API URL for the model server (default: http://localhost:11434/api/generate for Ollama)
        """
        self.use_ollama = use_ollama
        self.api_url = api_url or "http://localhost:11434/api/generate"
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve a geometric problem using G-LLaVA.
        Args:
            problem: Dictionary containing 'text' and optionally 'image_path'
        Returns:
            Dictionary with solution and metadata
        """
        if self.use_ollama:
            return self._solve_with_ollama(problem)
        else:
            return self._solve_with_lmstudio(problem)
    
    def _solve_with_ollama(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve problem using Ollama API"""
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
        """Solve problem using LM Studio"""
        # Implementation for LM Studio API
        # To be implemented based on LM Studio API specifics
        return {"solution": "", "model_used": "G-LLaVA (LM Studio)", "confidence": 0.0, "metadata": {}}
