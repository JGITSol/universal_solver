import sys
import os

# Ensure adv_resolver_math is importable for all submodules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'adv_resolver_math')))
