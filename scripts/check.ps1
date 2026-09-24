param(
    [switch]$SkipTests,
    [switch]$Coverage,
    [switch]$Fix
)

. "$PSScriptRoot/common.ps1"

foreach ($tool in @("black", "ruff", "mypy", "pytest")) {
    Assert-PythonModule $tool
}
if ($Coverage) {
    Assert-PythonModule "pytest_cov"
}

if ($Fix) {
    Write-Host "== Safe auto-fix ==" -ForegroundColor Cyan
    Invoke-PythonModule "ruff" (@("check", "--fix", "--show-fixes") + $PythonTargets)
    Invoke-PythonModule "black" $PythonTargets
}

Write-Host "== Compile check ==" -ForegroundColor Cyan
& python -m compileall -q trajectory_app tests app.py example_custom_solver.py
if ($LASTEXITCODE -ne 0) { throw "compileall failed with exit code $LASTEXITCODE" }

Write-Host "== Black check ==" -ForegroundColor Cyan
try {
    Invoke-PythonModule "black" (@("--check", "--diff") + $PythonTargets)
}
catch {
    if (-not $Fix) {
        Write-Host "Black formatting is required. Run: .\scripts\format.ps1 or .\scripts\check.ps1 -Fix" -ForegroundColor Yellow
    }
    throw
}

Write-Host "== Ruff lint ==" -ForegroundColor Cyan
Invoke-PythonModule "ruff" (@("check", "--output-format=full") + $PythonTargets)

Write-Host "== Mypy ==" -ForegroundColor Cyan
Invoke-PythonModule "mypy" (@("--show-error-codes", "--show-column-numbers") + $PythonTargets)

if (-not $SkipTests) {
    Write-Host "== Pytest ==" -ForegroundColor Cyan
    if ($Coverage) {
        Invoke-PythonModule "pytest" @(
            "-q",
            "--cov=trajectory_app",
            "--cov-report=term-missing",
            "--cov-report=xml"
        )
    }
    else {
        Invoke-PythonModule "pytest" @("-q")
    }
}

Write-Host "All checks passed." -ForegroundColor Green
