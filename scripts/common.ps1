$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$PythonTargets = @(
    "trajectory_app",
    "tests",
    "app.py",
    "example_custom_solver.py"
)

function Assert-PythonModule {
    param([Parameter(Mandatory = $true)][string]$Module)

    & python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$Module') else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "Missing development tool '$Module'. Run: python -m pip install -r requirements-dev.txt"
    }
}

function Invoke-PythonModule {
    param(
        [Parameter(Mandatory = $true)][string]$Module,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & python -m $Module @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Module failed with exit code $LASTEXITCODE"
    }
}
