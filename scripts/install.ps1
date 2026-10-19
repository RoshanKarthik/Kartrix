# Kartrix installer for Windows (PowerShell). Usage:
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/RoshanKarthik/Kartrix/main/scripts/install.ps1 | iex"
# Installs uv (the free Python tool manager) if it's missing, then Kartrix with its own Python. No admin rights,
# no Docker, no database setup. Set KARTRIX_SOURCE to install from elsewhere (a folder, a .whl file, a git URL).
# Dependencies get the tested versions from constraints.txt (uv.lock exported): the published one, a source folder's
# own, or KARTRIX_CONSTRAINTS (a path or URL; "none" for the newest versions pyproject.toml allows).
$ErrorActionPreference = "Stop"

$DefaultSource = "git+https://github.com/RoshanKarthik/Kartrix"
$DefaultConstraints = "https://raw.githubusercontent.com/RoshanKarthik/Kartrix/main/constraints.txt"

$Source = if ($env:KARTRIX_SOURCE) { $env:KARTRIX_SOURCE } else { $DefaultSource }

$Constraints = $env:KARTRIX_CONSTRAINTS
if (-not $Constraints) {
    if ($Source -eq $DefaultSource) {
        $Constraints = $DefaultConstraints
    } elseif (($Source -notmatch "://") -and (Test-Path -PathType Leaf (Join-Path $Source "constraints.txt"))) {
        $Constraints = Join-Path $Source "constraints.txt"
    }
}
if ($Constraints -eq "none") { $Constraints = $null }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (free Python tool manager)..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Installing Kartrix from $Source ..."
if ($Constraints) {
    uv tool install --python 3.12 --force --constraints "$Constraints" "$Source"
} else {
    Write-Host "No constraints file: dependencies get the newest versions allowed, not the tested ones."
    uv tool install --python 3.12 --force "$Source"
}
if ($LASTEXITCODE -ne 0) { throw "uv tool install failed (exit code $LASTEXITCODE)." }
uv tool update-shell | Out-Null

Write-Host ""
Write-Host "Kartrix is installed." -ForegroundColor Green
Write-Host "Open a NEW terminal, go to one of your repositories and run:  kartrix"
Write-Host "The first run asks for a free NVIDIA key (https://build.nvidia.com). Check everything with:  kartrix doctor"
