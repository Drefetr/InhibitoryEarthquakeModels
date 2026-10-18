param([switch]$Status,[switch]$Benchmark)
$ErrorActionPreference = 'Stop'
$pilotWorkspace = Split-Path -Parent $PSScriptRoot
$pilotPython = (Get-Command python -ErrorAction Stop).Source
$pilotRunner = Join-Path $PSScriptRoot 'code/run_level_calibration.py'
$pilotStage = if ($Benchmark) { 'benchmark' } else { 'full' }
$pilotMode = if ($Status) { '--status' } else { '--resume' }
Push-Location -LiteralPath $pilotWorkspace
try {
    Write-Host "Python: $pilotPython"
    & $pilotPython -B $pilotRunner $pilotMode --stage $pilotStage
    $pilotExit = $LASTEXITCODE
    if ($pilotExit -eq 0 -and -not $Status -and -not $Benchmark) {
        $pilotCollector = Join-Path $PSScriptRoot 'code/calibrate_level_only.py'
        & $pilotPython -B $pilotCollector --collect
        $pilotExit = $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
exit $pilotExit
