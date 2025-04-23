import json
import threading
from pathlib import Path

class MemoryManager:
    """Simple persistent memory manager for caching solver results."""
    def __init__(self, path: str = 'memory.json'):
        self.path = Path(path)
        self._lock = threading.Lock()
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    self.mem = json.load(f)
            except Exception:
                self.mem = {}
        else:
            self.mem = {}
    def get(self, problem: str):
        """Retrieve cached result for a problem."""
        return self.mem.get(problem)
    def add(self, problem: str, result: dict):
        """Add result to memory and persist to disk."""
        with self._lock:
            self.mem[problem] = result
            with open(self.path, 'w') as f:
                json.dump(self.mem, f, indent=2)
