#!/usr/bin/env bash
# SPATIAL TWIN · instalación + arranque (macOS / Linux)
#   ./run.sh          -> CPU (por defecto)
#   ./run.sh --gpu    -> torch con CUDA 12.6
set -euo pipefail
cd "$(dirname "$0")"

INDEX="https://download.pytorch.org/whl/cpu"
[[ "${1:-}" == "--gpu" ]] && INDEX="https://download.pytorch.org/whl/cu126"
PORT="${PORT:-8000}"

echo "[1/3] Instalando dependencias Python..."
python3 -m pip install -q -r requirements.txt
python3 -m pip install -q torch torchvision --index-url "$INDEX"

echo "[2/3] Abriendo el navegador en http://localhost:$PORT ..."
( sleep 4; (command -v xdg-open >/dev/null && xdg-open "http://localhost:$PORT") || open "http://localhost:$PORT" ) >/dev/null 2>&1 &

echo "[3/3] Levantando backend (Ctrl+C para salir)"
cd backend
PORT="$PORT" exec python3 main.py
