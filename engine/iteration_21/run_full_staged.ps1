[CmdletBinding()]
param(
    [switch]$PrepareOnly,
    [switch]$Status
)

$ErrorActionPreference = 'Stop'
if ($PrepareOnly -and $Status) {
    Write-Error 'Use either -PrepareOnly or -Status, not both.'
    exit 2
}

$projectDirectory = Split-Path -Parent $PSScriptRoot
$pythonExecutable = (Get-Command python -ErrorAction Stop).Source
$runnerPath = Join-Path $PSScriptRoot 'code\run_staged_full.py'
$analysisPath = Join-Path $PSScriptRoot 'code\analyse_full_staged.py'
$runDirectory = Join-Path $PSScriptRoot 'results\full_staged'
$logDirectory = Join-Path $PSScriptRoot 'logs'
$utf8Encoding = New-Object System.Text.UTF8Encoding($false)
$invocationId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')
$logPath = Join-Path $logDirectory ('launch_' + $invocationId + '.log')
$eventPath = Join-Path $logDirectory ('launch_' + $invocationId + '.jsonl')
$exitCode = 2

function Write-LaunchEvent {
    param([string]$Event, [hashtable]$Details)
    $record = @{ utc = [DateTime]::UtcNow.ToString('o'); event = $Event }
    foreach ($key in $Details.Keys) { $record[$key] = $Details[$key] }
    $line = ConvertTo-Json -InputObject $record -Depth 8 -Compress
    [System.IO.File]::AppendAllText($eventPath, $line + [Environment]::NewLine, $utf8Encoding)
}

function Invoke-ResearchPython {
    param([string]$Stage, [string]$Script, [string[]]$Arguments)
    Write-Host ("`n{0}: {1}" -f $Stage, [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))
    Write-LaunchEvent -Event 'stage_started' -Details @{ stage = $Stage; python = $pythonExecutable; script = $Script; arguments = $Arguments }
    $savedErrorPreference = $ErrorActionPreference
    try {
        # Windows PowerShell can wrap native stderr in NativeCommandError.
        # Keep the native exit code authoritative while retaining both streams.
        $ErrorActionPreference = 'Continue'
        & $pythonExecutable -B -u $Script @Arguments 2>&1 |
            ForEach-Object {
                $displayLine = $_.ToString()
                Write-Host $displayLine
                [System.IO.File]::AppendAllText($logPath, $displayLine + [Environment]::NewLine, $utf8Encoding)
            }
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedErrorPreference
    }
    if ($null -eq $nativeExitCode) { throw 'Python returned no process exit code.' }
    Write-LaunchEvent -Event 'stage_finished' -Details @{ stage = $Stage; exit_code = $nativeExitCode }
    return [int]$nativeExitCode
}

try {
    foreach ($requiredPath in @($pythonExecutable, $runnerPath, $analysisPath)) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw ('Required file is missing: ' + $requiredPath)
        }
    }
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    Push-Location -LiteralPath $projectDirectory
    try {
        Write-Host ('Python: ' + $pythonExecutable)
        Write-Host ('Checkpoints: ' + $runDirectory)
        Write-Host ('This invocation log: ' + $logPath)
        Write-LaunchEvent -Event 'launch_started' -Details @{ prepare_only = [bool]$PrepareOnly; status_only = [bool]$Status; run_directory = $runDirectory }
        if ($Status) {
            $exitCode = Invoke-ResearchPython -Stage 'Saved staged status' -Script $runnerPath -Arguments @('--status', '--run-dir', $runDirectory)
            if ($exitCode -eq 0) {
                $exitCode = Invoke-ResearchPython -Stage 'Saved analysis status' -Script $analysisPath -Arguments @('--status', '--run-dir', $runDirectory)
            }
        }
        elseif ($PrepareOnly) {
            $exitCode = Invoke-ResearchPython -Stage 'Prepare and verify; no fitting' -Script $runnerPath -Arguments @('--prepare-only', '--run-dir', $runDirectory)
        }
        else {
            Write-Host 'Full staged evaluation: 2,149 saved histories; two methods with six prescribed starts; eight-hour cumulative parent limit.'
            $exitCode = Invoke-ResearchPython -Stage 'Staged fitting / resume' -Script $runnerPath -Arguments @('--resume', '--run-dir', $runDirectory)
            if ($exitCode -eq 0) {
                $exitCode = Invoke-ResearchPython -Stage 'Matched null calibration and paired analysis' -Script $analysisPath -Arguments @('--run-dir', $runDirectory)
                if ($exitCode -ne 0) {
                    Write-Host ('All fitting cases completed. Analysis returned exit code {0}; numerical-envelope decisions may be unresolved. This is separate from a fit failure. Inspect the saved analysis output and log: {1}' -f $exitCode, $logPath)
                }
            }
            else {
                Write-Host ('Staged fitting stopped with exit code {0}. Analysis was not launched. Saved checkpoints and failure records remain in {1}.' -f $exitCode, $runDirectory)
            }
        }
        if ($exitCode -eq 0 -and -not $Status -and -not $PrepareOnly) {
            Write-Host ('Evaluation and analysis completed. Analysis pointer: ' + (Join-Path $runDirectory 'latest_analysis.json'))
        }
    }
    finally {
        Pop-Location
    }
}
catch {
    $exitCode = 2
    Write-Host ('Launcher stopped: ' + $_.Exception.Message) -ForegroundColor Red
    if (Test-Path -LiteralPath $logDirectory -PathType Container) {
        [System.IO.File]::AppendAllText($logPath, $_.ToString() + [Environment]::NewLine, $utf8Encoding)
        Write-LaunchEvent -Event 'launcher_error' -Details @{ error = $_.Exception.Message }
    }
}
finally {
    if (Test-Path -LiteralPath $eventPath -PathType Leaf) {
        Write-LaunchEvent -Event 'launch_finished' -Details @{ exit_code = $exitCode }
    }
}
exit $exitCode
