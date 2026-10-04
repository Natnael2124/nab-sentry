<#
.SYNOPSIS
    Start the NAB Sentry demo: ingest if needed, start the API_Server, print the Console URL.

.DESCRIPTION
    Steps (design "run_demo.ps1", Requirement 17):
      1. Check venv\Scripts\python.exe (exit 1 if missing); activate the venv when the
         execution policy allows. Every call uses the venv python by full path either way.
      2. If data\nab_sentry.db or data\vectors.faiss is missing, run
         scripts\ingest.py data\videos and print the ingested / failed counts. A non-zero
         ingest exit prints an error naming data\videos\ and exits 1 without starting the server.
      3. Start "python -m nab_sentry.api.app" (bound to 127.0.0.1 by the server itself).
      4. Poll /api/health every 500 ms for up to 30 s from script start. If the server exits
         first, print its error output and the corrective step, and exit with its code.
      5. Print "NAB Sentry console: http://127.0.0.1:<port>/" once both models report loaded,
         then keep the server in the foreground until it exits or Ctrl+C is pressed.

    Windows PowerShell 5.1 compatible. Writes only under data\ (Requirement 13.7).

.PARAMETER Port
    Port for the API_Server on 127.0.0.1 (passed as --set port=<Port>). Default 8765.

.EXAMPLE
    .\run_demo.ps1
    .\run_demo.ps1 -Port 8800
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765
)

$ErrorActionPreference = 'Continue'
$clock = [System.Diagnostics.Stopwatch]::StartNew()
$StartupTimeoutSec = 30
$PollIntervalMs = 500

$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }
Set-Location -LiteralPath $root

function Fail([string]$Message, [int]$Code) {
    Write-Error $Message
    exit $Code
}

function Get-CorrectiveStep([int]$Code) {
    switch ($Code) {
        2 { return 'invalid configuration or non-loopback host: check the --set values (the server binds only to 127.0.0.1)' }
        3 { return "cannot bind port $Port on 127.0.0.1: stop the other server using it, or run .\run_demo.ps1 -Port <free port>" }
        4 { return 'model manifest check failed: run venv\Scripts\python.exe scripts\fetch_models.py on a connected machine and copy models\ to this machine' }
        5 { return 'vector index unavailable or inconsistent: run venv\Scripts\python.exe scripts\ingest.py --repair' }
        6 { return 'model load failed: re-run venv\Scripts\python.exe scripts\fetch_models.py on a connected machine and copy models\ to this machine' }
        130 { return 'interrupted' }
        default { return 'see the server error output above and data\logs\' }
    }
}

function Stop-ServerTree($Proc) {
    if ($null -ne $Proc -and -not $Proc.HasExited) {
        # venv\Scripts\python.exe is a launcher with a child interpreter: kill the whole tree.
        & taskkill.exe /PID $Proc.Id /T /F 2>&1 | Out-Null
        if (-not $Proc.HasExited) {
            try { $Proc.Kill() } catch { }
        }
    }
}

# --- 1. venv -------------------------------------------------------------------------
$py = Join-Path $root 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    Fail 'venv/ not found: create it with py -3.12 -m venv venv and pip install -r requirements.txt' 1
}

$activate = Join-Path $root 'venv\Scripts\Activate.ps1'
$policy = [string](Get-ExecutionPolicy)
if ((Test-Path -LiteralPath $activate -PathType Leaf) -and ($policy -ne 'Restricted') -and ($policy -ne 'AllSigned')) {
    try {
        . $activate
    } catch {
        Write-Host "venv activation skipped ($($_.Exception.Message)); using $py directly."
    }
} else {
    Write-Host "venv activation skipped (execution policy $policy); using $py directly."
}

$dataDir = Join-Path $root 'data'
$logsDir = Join-Path $dataDir 'logs'
$dbPath = Join-Path $dataDir 'nab_sentry.db'
$indexPath = Join-Path $dataDir 'vectors.faiss'

# --- 2. ingest when the Metadata_Store or Vector_Index is missing ----------------------
if (-not (Test-Path -LiteralPath $dbPath -PathType Leaf) -or -not (Test-Path -LiteralPath $indexPath -PathType Leaf)) {
    Write-Host 'Metadata store or vector index not found: ingesting data\videos\ ...'
    $ingestLines = New-Object System.Collections.Generic.List[string]
    & $py (Join-Path $root 'scripts\ingest.py') 'data\videos' 2>&1 | ForEach-Object {
        $line = "$_"
        $ingestLines.Add($line)
        Write-Host $line
    }
    $ingestCode = $LASTEXITCODE

    $summary = $ingestLines | Where-Object { $_ -match 'ingested=(\d+)\s+failed=(\d+)' } | Select-Object -Last 1
    if ($summary -and ($summary -match 'ingested=(\d+)\s+failed=(\d+)')) {
        Write-Host "Ingest finished: $($Matches[1]) video(s) ingested, $($Matches[2]) failed."
    }

    if ($ingestCode -ne 0) {
        $msg = "ingest of data\videos\ failed (exit code $ingestCode): "
        if ($ingestCode -eq 1) {
            $msg += 'no video files found in data\videos\ or zero vectors indexed; add video files to data\videos\ and run .\run_demo.ps1 again'
        } else {
            $msg += (Get-CorrectiveStep $ingestCode)
        }
        Fail $msg 1
    }
}

# --- 3. start the API_Server -----------------------------------------------------------
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$outLog = Join-Path $logsDir 'server.stdout.log'
$errLog = Join-Path $logsDir 'server.stderr.log'

$proc = Start-Process -FilePath $py `
    -ArgumentList @('-m', 'nab_sentry.api.app', '--set', "port=$Port") `
    -WorkingDirectory $root -NoNewWindow -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog
# Cache the handle so ExitCode is available after the process exits (PowerShell 5.1 quirk).
$null = $proc.Handle

function Show-ServerError {
    foreach ($f in @($errLog, $outLog)) {
        if (Test-Path -LiteralPath $f) {
            $tail = Get-Content -LiteralPath $f -Tail 30 -ErrorAction SilentlyContinue
            foreach ($l in $tail) { [Console]::Error.WriteLine($l) }
        }
    }
}

try {
    # --- 4. poll /api/health -------------------------------------------------------------
    $healthUrl = "http://127.0.0.1:$Port/api/health"
    $ready = $false
    while ($clock.Elapsed.TotalSeconds -lt $StartupTimeoutSec) {
        if ($proc.HasExited) { break }
        try {
            $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 2 -ErrorAction Stop
            if ($health -and $health.models_loaded -and $health.models_loaded.detector -and $health.models_loaded.embedder) {
                $ready = $true
                break
            }
        } catch {
            # not listening yet
        }
        Start-Sleep -Milliseconds $PollIntervalMs
    }

    if (-not $ready) {
        if ($proc.HasExited) {
            $proc.WaitForExit()
            $code = $proc.ExitCode
            if ($null -eq $code -or $code -eq 0) { $code = 1 }
            Show-ServerError
            Fail "API server exited during startup (exit code $code): $(Get-CorrectiveStep $code)" $code
        }
        Stop-ServerTree $proc
        Show-ServerError
        Fail "API server did not report both models loaded at $healthUrl within $StartupTimeoutSec s; see data\logs\" 1
    }

    # --- 5. Console URL, then keep the server in the foreground --------------------------
    Write-Host ("NAB Sentry console: http://127.0.0.1:{0}/" -f $Port)
    Write-Host ("Started in {0:N1} s. Server logs: data\logs\server.stderr.log. Press Ctrl+C to stop." -f $clock.Elapsed.TotalSeconds)
    while (-not $proc.WaitForExit($PollIntervalMs)) { }
    $code = $proc.ExitCode
    if ($null -eq $code) { $code = 0 }
    if ($code -ne 0) {
        Show-ServerError
        Fail "API server stopped (exit code $code): $(Get-CorrectiveStep $code)" $code
    }
    exit 0
} finally {
    # Ctrl+C or any early exit: do not leave the server running.
    Stop-ServerTree $proc
}
