# Kartrix installer for Windows (PowerShell). Usage:
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/RoshanKarthik/Kartrix/main/scripts/install.ps1 | iex"
# Installs uv (the free Python tool manager) if it's missing, then Kartrix with its own Python. No admin rights,
# no Docker, no database setup. Set KARTRIX_SOURCE to install from elsewhere (a folder, a .whl file, a git URL).
$ErrorActionPreference = "Stop"

$DefaultSource = "git+https://github.com/RoshanKarthik/Kartrix"

$Source = if ($env:KARTRIX_SOURCE) { $env:KARTRIX_SOURCE } else { $DefaultSource }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (free Python tool manager)..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Installing Kartrix from $Source ..."
uv tool install --python 3.12 --force "$Source"
uv tool update-shell | Out-Null

Write-Host ""
Write-Host "Kartrix is installed." -ForegroundColor Green
Write-Host "Open a NEW terminal, go to one of your repositories and run:  kartrix"
Write-Host "The first run asks for a free NVIDIA key (https://build.nvidia.com). Check everything with:  kartrix doctor"
