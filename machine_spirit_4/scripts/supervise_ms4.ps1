# supervise_ms4.ps1
# Idempotent bring-up + watchdog for MS3 + MS4 (gateway/MCP).
# Launched by the 'MS4-Spirit-AutoStart' scheduled task (At-Logon + 5-min repetition).
# Only launches services that are currently down; never kills anything.
$ErrorActionPreference = 'SilentlyContinue'

$Root   = 'C:\Users\nexus-hc-win-00\Downloads\TMR'
$Py     = Join-Path $Root 'machine_spirit_4\.venv\Scripts\python.exe'
$Start  = Join-Path $Root 'machine_spirit_4\scripts\start_ms4.py'
$LogDir = Join-Path $Root 'machine_spirit_4\logs\supervisor'
$Ports  = '9080','9180','9181'   # MS3 sidecar, MS4 gateway, MS4 MCP

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir ("ms4_supervise_{0}.log" -f (Get-Date -Format 'yyyy-MM-dd'))
function Write-Log($m) { ("{0}  {1}" -f (Get-Date -Format s), $m) | Out-File -FilePath $Log -Append -Encoding utf8 }

if (-not (Test-Path $Py)) { Write-Log ("ABORT: venv python missing at {0}" -f $Py); return }

$listening = netstat -ano | Select-String 'LISTENING'
$down = $Ports | Where-Object { -not ($listening | Select-String (":{0}\b" -f $_)) }

if ($down) {
    $hive = try { (Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:6089/v1/health' -TimeoutSec 4).StatusCode } catch { 'unreachable' }
    Write-Log ("down=[{0}] hivemind /v1/health={1} -> start_ms4 --skip-validation" -f ($down -join ','), $hive)
    & $Py $Start --skip-validation *>> $Log
    Write-Log ("start_ms4 exit={0}" -f $LASTEXITCODE)
} else {
    Write-Log 'ok: 9080/9180/9181 all listening'
}
