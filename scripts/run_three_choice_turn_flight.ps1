$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$simulation = Join-Path $projectRoot "/demos/crw_simulation.py"
$output = Join-Path $projectRoot "./demos/src/artifacts\three-choice-endpoints.png"

$env:PYTHONPATH = Join-Path $projectRoot "src"
& $python $simulation `
    --model three-choice-turn-flight `
    --steps 100 `
    --speed 2 `
    --turning-radius 5 `
    --time-step 1.0 `
    --initial-direction 1.57079632679 `
    --trajectories 100000 `
    --no-show `
    --output $output
exit $LASTEXITCODE