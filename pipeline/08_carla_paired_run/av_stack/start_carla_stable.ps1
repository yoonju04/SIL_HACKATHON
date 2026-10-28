# Start CARLA (it always opens Town10HD_Opt; run_benchmark switches to the route's map).
# A Low-quality + -ExecCmds launch crashed once mid-route; Epic at 960x540 has been stable.
#   powershell -ExecutionPolicy Bypass -File .\start_carla_stable.ps1 [-Quality Epic]
param([ValidateSet('Low', 'Epic')][string]$Quality = 'Epic')
$ErrorActionPreference = 'Stop'
# CARLA_ROOT (the WindowsNoEditor folder) if set, else a CARLA folder in a parent directory, e.g. next to the repository.
$carlaRoot = $env:CARLA_ROOT
$parent = Split-Path $PSScriptRoot -Parent
while (-not $carlaRoot -and $parent) {
    $carlaRoot = @('CARLA_0.9.15', 'CALRA 0.9.15') | ForEach-Object { Join-Path $parent "$_\WindowsNoEditor" } |
        Where-Object { Test-Path $_ } | Select-Object -First 1
    $parent = Split-Path $parent -Parent
}
$carlaProgram = Join-Path $carlaRoot 'CarlaUE4.exe'
$runningCarla = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'CarlaUE4-Win64-Shipping.exe' }
if ($runningCarla) {
    throw 'CARLA is already running. Close its window before using this Epic launcher.'
}
$serverLogFolder = Join-Path $PSScriptRoot 'runs\server_logs'
New-Item -ItemType Directory -Path $serverLogFolder -Force | Out-Null
$serverLogPath = Join-Path $serverLogFolder ('CarlaUE4-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
$launchArguments = @('-windowed', '-ResX=960', '-ResY=540', "-quality-level=$Quality", '-nosound', ('-abslog="' + $serverLogPath + '"'))
Start-Process -FilePath $carlaProgram -ArgumentList $launchArguments -WorkingDirectory (Split-Path $carlaProgram) -WindowStyle Normal
Write-Host "CARLA launched: quality $Quality. Wait for the game window, then run the driving command."
Write-Host ('Requested Unreal log file: ' + $serverLogPath)
