# Setup script for Advanced Math Ensemble Solver

# Create virtual environment
Write-Host "Creating virtual environment in adv_res_venv..." -ForegroundColor Cyan
Python -m venv adv_res_venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
.\adv_res_venv\Scripts\Activate.ps1

# Install uv package manager
Write-Host "Installing uv package manager..." -ForegroundColor Cyan
pip install uv

# Install dependencies using uv
Write-Host "Installing dependencies using uv..." -ForegroundColor Cyan
uv pip install -r requirements.txt

Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host "To activate the environment in the future, run: .\adv_res_venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "To run the math solver, run: python math_ensemble_langchain_ollama.py" -ForegroundColor Yellow