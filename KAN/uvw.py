"""Simple wrapper to run the `uv` CLI using the adv_res_venv Python interpreter.

Usage (from repo root):
  D:/REPOS/universal_solver/adv_res_venv/Scripts/python.exe KAN/uvw.py --help

This ensures that when working inside KAN, the `uv` package installed in
adv_res_venv is used instead of any system Python.
"""

import os
import subprocess
import sys

VENV_PY = os.path.join(
    os.path.dirname(__file__), "..", "adv_res_venv", "Scripts", "python.exe"
)


def main():
    # Resolve absolute path
    venv_python = os.path.abspath(VENV_PY)
    if not os.path.exists(venv_python):
        print(f"adv_res_venv python not found at {venv_python}")
        sys.exit(2)

    # Forward all args to `-m uv` so the interpreter runs the uv module
    cmd = [venv_python, "-m", "uv"] + sys.argv[1:]
    try:
        proc = subprocess.run(cmd)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
