param(
    [switch]$UnsafeFixes
)

. "$PSScriptRoot/common.ps1"

foreach ($tool in @("ruff", "black")) {
    Assert-PythonModule $tool
}

Write-Host "== Ruff autofix ==" -ForegroundColor Cyan
$ruffArgs = @("check", "--fix", "--show-fixes")
if ($UnsafeFixes) {
    $ruffArgs += "--unsafe-fixes"
}
Invoke-PythonModule "ruff" ($ruffArgs + $PythonTargets)

Write-Host "== Black format ==" -ForegroundColor Cyan
Invoke-PythonModule "black" $PythonTargets

Write-Host "== Ruff verify remaining lint ==" -ForegroundColor Cyan
Invoke-PythonModule "ruff" (@("check", "--output-format=full") + $PythonTargets)

Write-Host "Formatting/fixes complete." -ForegroundColor Green
