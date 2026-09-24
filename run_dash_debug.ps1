$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = $Root
$env:TRAJECTORY_DASH_DEBUG = "1"
Set-Location $Root

python .\dash_app.py --debug
