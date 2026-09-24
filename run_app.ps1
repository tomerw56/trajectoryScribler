$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$oldPythonPath = $env:PYTHONPATH
try {
    if ([string]::IsNullOrWhiteSpace($oldPythonPath)) {
        $env:PYTHONPATH = $ProjectRoot
    }
    else {
        $env:PYTHONPATH = "$ProjectRoot;$oldPythonPath"
    }

    Write-Host "Launching local Trajectory Scribbler from:" -ForegroundColor Cyan
    Write-Host "  $ProjectRoot"
    & python .\app.py
    if ($LASTEXITCODE -ne 0) {
        throw "Trajectory Scribbler exited with code $LASTEXITCODE"
    }
}
finally {
    $env:PYTHONPATH = $oldPythonPath
}
