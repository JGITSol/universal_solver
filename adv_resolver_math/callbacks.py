# callbacks.py

import time
from rich.console import Console
from langchain.callbacks.base import BaseCallbackHandler

class MathSolvingCallbackHandler(BaseCallbackHandler):
    """
    Callback handler for monitoring math solving progress and performance metrics.
    
    This handler tracks token generation, timing, and provides real-time feedback
    during the model's solution generation process.
    """
    
    def __init__(self, console: Console, model_name: str):
        self.console = console
        self.model_name = model_name
        self.start_time = None
        self.tokens = 0
        self.step_markers = 0
        self.equation_count = 0
        
    def on_llm_start(self, *args, **kwargs):
        self.start_time = time.time()
        self.console.print(f"[dim]{self.model_name} is thinking...[/dim]")
        
    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens += 1
        
        # Track step markers for analysis
        if "Step" in token or "\item" in token:
            self.step_markers += 1
            
        # Track equation generation
        if "=" in token:
            self.equation_count += 1
            
        # Provide periodic updates
        if self.tokens % 50 == 0:
            elapsed = time.time() - self.start_time
            tokens_per_sec = self.tokens / max(0.1, elapsed)
            self.console.print(f"[dim]{self.model_name}: {self.tokens} tokens generated ({tokens_per_sec:.1f} tokens/sec)[/dim]", end="\r")
            
    def on_llm_end(self, *args, **kwargs):
        elapsed = time.time() - self.start_time
        tokens_per_sec = self.tokens / max(0.1, elapsed)
        
        self.console.print(f"[dim]{self.model_name} completed in {elapsed:.2f} seconds, generated {self.tokens} tokens ({tokens_per_sec:.1f} tokens/sec)[/dim]")
        self.console.print(f"[dim]Solution contains {self.step_markers} step markers and {self.equation_count} equations[/dim]")
        
    def on_llm_error(self, error: Exception, **kwargs):
        self.console.print(f"[red]Error with {self.model_name}: {error}[/red]")
        
    def get_metrics(self):
        """Return metrics collected during generation."""
        if not self.start_time:
            return {}
            
        elapsed = time.time() - self.start_time
        return {
            "model": self.model_name,
            "tokens": self.tokens,
            "time_seconds": elapsed,
            "tokens_per_second": self.tokens / max(0.1, elapsed),
            "step_markers": self.step_markers,
            "equation_count": self.equation_count
        }