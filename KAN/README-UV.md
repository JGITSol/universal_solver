KAN — uv wrapper usage
======================

This repository includes a small wrapper to run the `uv` package using the
project virtual environment `adv_res_venv`. Use the provided wrapper so that
all package-management commands run against the correct interpreter.

Quick usage (PowerShell)
------------------------

From repository root:

```powershell
D:\REPOS\universal_solver\KAN\run_uv.ps1 --help
```

Or directly (any shell):

```sh
D:/REPOS/universal_solver/adv_res_venv/Scripts/python.exe KAN/uvw.py --help
```

Examples
--------

Install a package via uv:

```powershell
D:/REPOS/universal_solver/KAN/run_uv.ps1 install requests
```

Update packages:

```powershell
D:/REPOS/universal_solver/KAN/run_uv.ps1 update
```

Why this wrapper?
------------------

- Keeps KAN's operations isolated to the project's virtualenv (`adv_res_venv`).
- Avoids accidental installs into system Python or other venvs (like KAN/venv).
