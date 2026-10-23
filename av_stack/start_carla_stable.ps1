$ErrorActionPreference = 'Stop'
# CARLA_ROOT (the WindowsNoEditor folder) if set, else the CARLA folder next to the repository.
$carlaRoot = $env:CARLA_ROOT
if (-not $carlaRoot) {
    $carlaRoot = @('CARLA_0.9.15', 'CALRA 0.9.15') | ForEach-Object { Join-Path $PSScriptRoot "..\..\$_\WindowsNoEditor" } |
        Where-Object { Test-Path $_ } | Select-Object -First 1
}
$carlaProgram = Join-Path $carlaRoot 'CarlaUE4.exe'
$runningCarla = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'CarlaUE4-Win64-Shipping.exe' }
if ($runningCarla) {
    throw 'CARLA is already running. Close its window before using this Epic launcher.'
}
$serverLogFolder = Join-Path $PSScriptRoot 'runs\server_logs'
New-Item -ItemType Directory -Path $serverLogFolder -Force | Out-Null
$serverLogPath = Join-Path $serverLogFolder ('CarlaUE4-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
$launchArguments = @('Town05', '-windowed', '-ResX=960', '-ResY=540', '-quality-level=Epic', '-nosound', ('-abslog="' + $serverLogPath + '"'))
Start-Process -FilePath $carlaProgram -ArgumentList $launchArguments -WorkingDirectory (Split-Path $carlaProgram) -WindowStyle Normal
Write-Host 'CARLA launched with Epic quality. Wait for the game window, then run the driving command.'
Write-Host ('Requested Unreal log file: ' + $serverLogPath)
