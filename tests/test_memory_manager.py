import os
import json
import pytest
from adv_resolver_math.memory import MemoryManager

def test_memory_manager(tmp_path):
    mem_path = tmp_path / "mem.json"
    # Initialize manager, should start empty
    mgr = MemoryManager(path=str(mem_path))
    assert mgr.get("problem1") is None

    result = {"answer": 42}
    mgr.add("problem1", result)
    # File should exist now
    assert mem_path.exists()
    with open(mem_path, 'r') as f:
        data = json.load(f)
    assert data == {"problem1": result}

    # New manager reads existing file
    mgr2 = MemoryManager(path=str(mem_path))
    assert mgr2.get("problem1") == result
