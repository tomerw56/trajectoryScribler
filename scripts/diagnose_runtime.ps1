$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$oldPythonPath = $env:PYTHONPATH
try {
    if ([string]::IsNullOrWhiteSpace($oldPythonPath)) {
        $env:PYTHONPATH = $ProjectRoot
    }
    else {
        $env:PYTHONPATH = "$ProjectRoot;$oldPythonPath"
    }

    & python -c @'
from pathlib import Path
import trajectory_app
import trajectory_app.core as core
print("trajectory_app:", Path(trajectory_app.__file__).resolve())
print("core.py:", Path(core.__file__).resolve())
print("fixed_velocity_covariance_from_model:", hasattr(core, "fixed_velocity_covariance_from_model"))
try:
    import trajectory_app.main_window as main_window
    print("main_window.py:", Path(main_window.__file__).resolve())
    print("_prediction_visibility_changed:", hasattr(main_window.MainWindow, "_prediction_visibility_changed"))
except Exception as exc:
    print("main_window import failed:", repr(exc))
    raise
'@
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime diagnostic failed with code $LASTEXITCODE"
    }
}
finally {
    $env:PYTHONPATH = $oldPythonPath
}
