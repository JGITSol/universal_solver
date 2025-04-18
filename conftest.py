import sys
import os, sys
import pathlib

# Ensure project root is on PYTHONPATH
root = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'adv_resolver_math'))
