# PowerShell helper to run uv through adv_res_venv
param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$wrapper = Join-Path $repoRoot "uvw.py"

if (-not (Test-Path $wrapper)) {
    Write-Error "Wrapper $wrapper not found."
    exit 2
}

# Use the adv_res_venv python exe (relative to repo root)
$venvPython = Join-Path $repoRoot '..\adv_res_venv\Scripts\python.exe'
$venvPython = Resolve-Path $venvPython -ErrorAction SilentlyContinue
if (-not $venvPython) {
    Write-Error "adv_res_venv python not found."
    exit 2
}

& $venvPython.Path $wrapper @Args
exit $LASTEXITCODE