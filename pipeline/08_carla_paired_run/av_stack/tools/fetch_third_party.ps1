# Fetch the large / non-redistributable third-party code at pinned commits.
# Run from av_stack:  powershell -ExecutionPolicy Bypass -File tools\fetch_third_party.ps1 [-WithTCP] [-WithModels]
#   carla_garage (MIT, with Bench2Drive) : cloned + our two patches applied (third_party/patches)
#   bench2drive_tcp (CC BY-NC-ND 4.0)     : only with -WithTCP; kept unmodified, never committed
#   model weights                         : only with -WithModels (TF++ CC BY 4.0, TCP checkpoint)
param([switch]$WithTCP, [switch]$WithModels)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Get-Pinned($url, $commit, $dest) {
    if (Test-Path $dest) { Write-Host "skip $dest (exists)"; return }
    git clone --no-checkout $url $dest
    git -C $dest checkout $commit
}

Get-Pinned 'https://github.com/autonomousvision/carla_garage.git' '48e6a6dbcb133220333bd65e242fbc2b492bb15b' 'third_party/carla_garage'
git -C third_party/carla_garage apply --check ../patches/carla_garage.patch 2>$null
if ($LASTEXITCODE -eq 0) { git -C third_party/carla_garage apply ../patches/carla_garage.patch; Write-Host 'applied carla_garage.patch' }
else { Write-Host 'carla_garage.patch already applied or does not match; check git -C third_party/carla_garage diff' }

if ($WithTCP) {
    Get-Pinned 'https://github.com/Thinklab-SJTU/Bench2DriveZoo.git' '8a08b07883f10b7d83f6bf5dd475bda91a91c50a' 'third_party/bench2drive_tcp'
}

if ($WithModels) {
    New-Item -ItemType Directory -Force models, work | Out-Null
    if (-not (Test-Path 'models/transfuserpp/pretrained_models/all_towns/model_0030_0.pth')) {
        $zip = 'work/pretrained_models.zip'
        if (-not (Test-Path $zip)) {
            Invoke-WebRequest 'https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip' -OutFile $zip
        }
        Expand-Archive $zip -DestinationPath 'models/transfuserpp' -Force
    }
    if ($WithTCP -and -not (Test-Path 'models/tcp_b2d.ckpt')) {
        Invoke-WebRequest 'https://huggingface.co/rethinklab/Bench2DriveZoo/resolve/140aea58c02185e787c3df39d7ad79e910967b8b/tcp_b2d.ckpt' -OutFile 'models/tcp_b2d.ckpt'
        $hash = (Get-FileHash 'models/tcp_b2d.ckpt' -Algorithm SHA256).Hash.ToLower()
        if ($hash -ne 'e6573ff1f8ea910b9a53eddfb68f69cac469bf5bfa253a516578f6126110b4fe') { throw "tcp_b2d.ckpt SHA256 mismatch: $hash" }
    }
}
