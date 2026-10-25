<#
.SYNOPSIS
    Start and supervise the live NAB Sentry system (V2 Requirement 16).

.DESCRIPTION
    Starts these services as child processes and restarts any that exits, with back-off
    (2, 4, 8 ... 60 s; reset after 60 s of healthy running):

      recorder  scripts\record.py               when config\cameras.yaml exists
      watcher   scripts\ingest.py --watch       always (ingest, tracking, rules, retention)
      notifier  scripts\notify.py               when NAB_TELEGRAM_TOKEN and NAB_TELEGRAM_CHATS are set
      api       python -m nab_sentry.api.app    the search console on 127.0.0.1

    Each service writes data\logs\<name>.out.log and <name>.err.log; starts, exits and restarts
    go to data\logs\run_live.log and the console. Ctrl+C stops every service (whole process
    trees). Windows PowerShell 5.1 compatible. Writes only under data\.

.PARAMETER Port
    API_Server port on 127.0.0.1. Default 8765.

.PARAMETER NoRecorder
    Do not start the recorder (e.g. footage arrives by other means, or scripts\replay_camera.py).

.PARAMETER NoNotifier
    Do not start the Telegram notifier.

.PARAMETER NoServer
    Do not start the API_Server.

.PARAMETER Vlm
    Start the notifier with --vlm: /describe and rules with a verify question use the local
    Ollama VLM (qwen3-vl:2b-instruct; Ollama must be running).

.PARAMETER DryRun
    Print what would be started and exit 0.

.EXAMPLE
    $env:NAB_TELEGRAM_TOKEN = '123456:ABC...'; $env:NAB_TELEGRAM_CHATS = '123456789'
    .\run_live.ps1
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [switch]$NoRecorder,
    [switch]$NoNotifier,
    [switch]$NoServer,
    [switch]$Vlm,
    [switch]$DryRun
)

$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }
Set-Location -LiteralPath $root

$py = Join-Path $root 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    Write-Error 'venv/ not found: create it with py -3.12 -m venv venv and pip install -r requirements.txt'
    exit 1
}
$logs = Join-Path $root 'data\logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$supervisorLog = Join-Path $logs 'run_live.log'

function Write-Log([string]$Message) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Write-Host $line
    Add-Content -LiteralPath $supervisorLog -Value $line -Encoding UTF8
}

# --- which services ------------------------------------------------------------------------
$services = New-Object System.Collections.ArrayList
$cameras = Join-Path $root 'config\cameras.yaml'
if ($NoRecorder) {
    Write-Log 'recorder: skipped (-NoRecorder)'
} elseif (-not (Test-Path -LiteralPath $cameras -PathType Leaf)) {
    Write-Log 'recorder: skipped (no config\cameras.yaml; copy config\cameras.example.yaml and edit it)'
} else {
    [void]$services.Add(@{ Name = 'recorder'; Args = @('scripts\record.py') })
}
[void]$services.Add(@{ Name = 'watcher'; Args = @('scripts\ingest.py', '--watch') })
if ($NoNotifier) {
    Write-Log 'notifier: skipped (-NoNotifier)'
} elseif (-not $env:NAB_TELEGRAM_TOKEN -or -not $env:NAB_TELEGRAM_CHATS) {
    Write-Log 'notifier: skipped (set NAB_TELEGRAM_TOKEN and NAB_TELEGRAM_CHATS to send alerts to Telegram)'
} else {
    $notifyArgs = @('scripts\notify.py')
    if ($Vlm) { $notifyArgs += '--vlm' }
    [void]$services.Add(@{ Name = 'notifier'; Args = $notifyArgs })
}
if ($NoServer) {
    Write-Log 'api: skipped (-NoServer)'
} else {
    [void]$services.Add(@{ Name = 'api'; Args = @('-m', 'nab_sentry.api.app', '--set', "port=$Port") })
}

if ($DryRun) {
    foreach ($s in $services) { Write-Host ('would start {0}: {1} {2}' -f $s.Name, $py, ($s.Args -join ' ')) }
    exit 0
}

# --- supervise -----------------------------------------------------------------------------
function Start-Service-Proc($s) {
    $out = Join-Path $logs ('{0}.out.log' -f $s.Name)
    $err = Join-Path $logs ('{0}.err.log' -f $s.Name)
    $s.Proc = Start-Process -FilePath $py -ArgumentList $s.Args -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    # PowerShell 5.1 reports ExitCode only if the process handle was taken while it ran.
    $null = $s.Proc.Handle
    $s.Started = Get-Date
    Write-Log ('{0}: started (pid {1})' -f $s.Name, $s.Proc.Id)
}

function Stop-Tree($proc) {
    if ($null -ne $proc -and -not $proc.HasExited) {
        & taskkill.exe /PID $proc.Id /T /F 2>&1 | Out-Null
    }
}

foreach ($s in $services) {
    $s.Failures = 0
    $s.RestartAt = $null
    Start-Service-Proc $s
}
Write-Log ('running {0} service(s); Ctrl+C to stop' -f $services.Count)
if (-not $NoServer) { Write-Log ('NAB Sentry console: http://127.0.0.1:{0}/' -f $Port) }

try {
    while ($true) {
        Start-Sleep -Seconds 2
        foreach ($s in $services) {
            if ($null -ne $s.RestartAt) {
                if ((Get-Date) -ge $s.RestartAt) {
                    $s.RestartAt = $null
                    Start-Service-Proc $s
                }
                continue
            }
            if ($s.Proc.HasExited) {
                $ran = ((Get-Date) - $s.Started).TotalSeconds
                if ($ran -ge 60) { $s.Failures = 0 }
                $s.Failures += 1
                $delay = [Math]::Min(60, [Math]::Pow(2, $s.Failures))
                Write-Log ('{0}: exited with code {1} after {2:N0} s; restarting in {3} s (see data\logs\{0}.err.log)' -f `
                    $s.Name, $s.Proc.ExitCode, $ran, $delay)
                $s.RestartAt = (Get-Date).AddSeconds($delay)
            }
        }
    }
} finally {
    Write-Log 'stopping services'
    foreach ($s in $services) { Stop-Tree $s.Proc }
}
