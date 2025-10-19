# callbacks.py

from __future__ import annotations

import time
from typing import Dict, Optional

from langchain.callbacks.base import BaseCallbackHandler
from rich.console import Console


class MathSolvingCallbackHandler(BaseCallbackHandler):
    """
    Callback handler for monitoring math solving progress and performance metrics.

    This handler tracks token generation, timing, and provides real-time feedback
    during the model's solution generation process.
    """

    def __init__(self, console: Console, model_name: str):
        self.console = console
        self.model_name = model_name
        self.start_time: Optional[float] = None
        self.tokens = 0
        self.step_markers = 0
        self.equation_count = 0
        self._last_metrics: Optional[Dict[str, float]] = None

    def on_llm_start(self, *args, **kwargs):
        self.start_time = time.time()
        self.tokens = 0
        self.step_markers = 0
        self.equation_count = 0
        self._last_metrics = None
        self.console.print(f"[dim]{self.model_name} is thinking...[/dim]")

    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens += 1

        # Track step markers for analysis
        if "Step" in token or "\\item" in token:
            self.step_markers += 1

        # Track equation generation
        if "=" in token:
            self.equation_count += 1

        # Provide periodic updates
        if self.tokens % 50 == 0 and self.start_time is not None:
            elapsed = time.time() - self.start_time
            tokens_per_sec = self.tokens / max(0.1, elapsed)
            self.console.print(
                (
                    f"[dim]{self.model_name}: {self.tokens} tokens generated "
                    f"({tokens_per_sec:.1f} tokens/sec)[/dim]"
                ),
                end="\r",
            )

    def on_llm_end(self, *args, **kwargs):
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        tokens_per_sec = self.tokens / max(0.1, elapsed)

        self.console.print(
            (
                f"[dim]{self.model_name} completed in {elapsed:.2f} seconds, "
                f"generated {self.tokens} tokens "
                f"({tokens_per_sec:.1f} tokens/sec)[/dim]"
            )
        )
        self.console.print(
            (
                "[dim]Solution contains "
                f"{self.step_markers} step markers and {self.equation_count} "
                "equations[/dim]"
            )
        )
        self._last_metrics = {
            "time_seconds": elapsed,
            "tokens": float(self.tokens),
            "tokens_per_second": tokens_per_sec,
            "step_markers": float(self.step_markers),
            "equation_count": float(self.equation_count),
        }
        self.start_time = None

    def on_llm_error(self, error: BaseException, **kwargs):
        self.console.print(f"[red]Error with {self.model_name}: {error}[/red]")

    def get_metrics(self):
        """Return metrics collected during generation."""
        if self._last_metrics is None:
            return {}

        return {
            "model": self.model_name,
            **self._last_metrics,
        }
