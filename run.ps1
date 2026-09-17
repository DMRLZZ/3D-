# SPATIAL TWIN · instalación + arranque (Windows PowerShell)
#   .\run.ps1          -> CPU (por defecto)
#   .\run.ps1 -Gpu     -> torch con CUDA 12.6
param([switch]$Gpu, [int]$Port = 8000)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$index = if ($Gpu) { "https://download.pytorch.org/whl/cu126" } else { "https://download.pytorch.org/whl/cpu" }
Write-Host "[1/3] Instalando dependencias Python..." -ForegroundColor Cyan
python -m pip install -q -r requirements.txt
python -m pip install -q torch torchvision --index-url $index

Write-Host "[2/3] Abriendo el navegador en http://localhost:$Port ..." -ForegroundColor Cyan
Start-Job -ScriptBlock { param($p) Start-Sleep -Seconds 4; Start-Process "http://localhost:$p" } -ArgumentList $Port | Out-Null

Write-Host "[3/3] Levantando backend (Ctrl+C para salir)" -ForegroundColor Cyan
$env:PORT = "$Port"
Set-Location backend
python main.py
