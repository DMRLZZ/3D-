"""
SPATIAL TWIN · Backend de reconstrucción 3D RGB-D
==================================================

Pipeline real (sin mocks):

    video (smartphone)
      └─► 1. extracción de fotogramas clave   (nitidez Laplaciana + cobertura visual greedy)
      └─► 2. profundidad métrica monocular    (Depth Anything V2 · Metric Indoor · HuggingFace)
      └─► 3. registro multi-vista             (ORB + PnP RANSAC sobre puntos 3D métricos)
      └─► 4. desproyección RGB-D → malla       (X=(u-cx)·Z/fx, Y=(v-cy)·Z/fy, grid de triángulos + UV)
      └─► 5. exportación                      (GLB texturizado + PLY de nube de puntos + metadata JSON)
      └─► 6. API FastAPI                      (/api/scan, /api/status, /api/model, /api/metadata, ...)

Ejecutar:   cd backend && python main.py      →  http://localhost:8000
Variables:  DEPTH_MODEL, DEPTH_BACKEND (local|remote), HF_TOKEN, CAMERA_HFOV,
            MAX_KEYFRAMES, MESH_STEP, TEXTURE_MAX, MAX_DEPTH, PORT
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# --------------------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
PREVIEW_DIR = OUTPUT_DIR / "previews"
for _d in (UPLOAD_DIR, OUTPUT_DIR, PREVIEW_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEPTH_MODEL = os.getenv("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
DEPTH_BACKEND = os.getenv("DEPTH_BACKEND", "local").lower()          # local | remote
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
CAMERA_HFOV_DEG = float(os.getenv("CAMERA_HFOV", "70"))               # FOV horizontal típico de smartphone
DEFAULT_KEYFRAMES = int(os.getenv("MAX_KEYFRAMES", "4"))
DEFAULT_MESH_STEP = int(os.getenv("MESH_STEP", "3"))                  # zancada en píxeles del grid de triángulos
TEXTURE_MAX = int(os.getenv("TEXTURE_MAX", "960"))                    # lado mayor de la textura por keyframe
MAX_DEPTH_M = float(os.getenv("MAX_DEPTH", "12.0"))                   # recorte de profundidad (metros)
CANDIDATE_FRAMES = int(os.getenv("CANDIDATE_FRAMES", "36"))           # fotogramas candidatos a muestrear
PORT = int(os.getenv("PORT", "8000"))

MODEL_GLB = OUTPUT_DIR / "scene.glb"
MODEL_PLY = OUTPUT_DIR / "scene.ply"
METADATA_JSON = OUTPUT_DIR / "metadata.json"
VIDEO_COPY = OUTPUT_DIR / "source_video.mp4"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("spatial-twin")


class PipelineError(RuntimeError):
    """Error controlado del pipeline (se reporta al cliente con mensaje legible)."""


# --------------------------------------------------------------------------------------
# Estado del job (un único escaneo activo: suficiente para la demo)
# --------------------------------------------------------------------------------------
@dataclass
class JobState:
    id: str = ""
    state: str = "idle"                # idle | running | done | error
    stage: str = ""
    progress: float = 0.0              # 0..1 real, actualizado por cada etapa
    log: list = field(default_factory=list)
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reset(self, job_id: str) -> None:
        with self.lock:
            self.id, self.state, self.stage, self.progress = job_id, "running", "init", 0.0
            self.log, self.error = [], ""
            self.started_at, self.finished_at = time.time(), 0.0

    def push(self, stage: str, msg: str, progress: Optional[float] = None) -> None:
        with self.lock:
            self.stage = stage
            if progress is not None:
                self.progress = float(min(max(progress, 0.0), 1.0))
            self.log.append({"t": round(time.time() - self.started_at, 2), "stage": stage, "msg": msg})
        log.info("[%s] %s", stage, msg)

    def finish(self, error: str = "") -> None:
        with self.lock:
            self.state = "error" if error else "done"
            self.error = error
            self.progress = self.progress if error else 1.0
            self.finished_at = time.time()

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = (self.finished_at or time.time()) - self.started_at if self.started_at else 0.0
            return {
                "id": self.id, "state": self.state, "stage": self.stage, "progress": round(self.progress, 4),
                "log": list(self.log[-80:]), "error": self.error, "elapsed": round(elapsed, 2),
                "model_ready": MODEL_GLB.exists() and METADATA_JSON.exists(),
            }


JOB = JobState()


# --------------------------------------------------------------------------------------
# 1. Extracción de fotogramas clave
# --------------------------------------------------------------------------------------
@dataclass
class Keyframe:
    index: int                 # índice del fotograma en el video
    time_s: float              # instante en segundos
    rgb: np.ndarray            # HxWx3 uint8 (BGR hasta el final de la selección, luego RGB)
    sharpness: float           # varianza del Laplaciano (nitidez)
    depth: Optional[np.ndarray] = None   # HxW float32 en metros (etapa 2)


def _resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(2, int(round(w * s))), max(2, int(round(h * s)))), interpolation=cv2.INTER_AREA)


def _sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(_resize_max(bgr, 320), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _thumb_descriptor(bgr: np.ndarray) -> np.ndarray:
    """Descriptor compacto de apariencia (thumb 24x24 en Lab) para medir cobertura visual."""
    lab = cv2.cvtColor(cv2.resize(bgr, (24, 24), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB)
    return lab.astype(np.float32).ravel() / 255.0


def _read_frame_at(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok and frame is not None and frame.size else None


def extract_keyframes(video_path: Path, k: int) -> tuple[list, dict]:
    """
    Muestrea CANDIDATE_FRAMES fotogramas uniformemente, mide nitidez (Laplaciano) y
    selecciona k fotogramas con un greedy que maximiza cobertura visual (distancia
    entre descriptores) ponderada por nitidez. Robusto a videos muy cortos.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PipelineError("No se pudo abrir el video. Formato no soportado o archivo corrupto.")
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)   # respeta la rotación del móvil si el backend lo soporta
    except Exception:
        pass

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    if total <= 0:   # algunos contenedores no reportan longitud: contamos con grab()
        total = 0
        while cap.grab():
            total += 1
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
    if total <= 0:
        cap.release()
        raise PipelineError("El video no contiene fotogramas legibles.")

    n_cand = int(min(CANDIDATE_FRAMES, total))
    cand_idx = np.unique(np.linspace(0, max(total - 1, 0), n_cand).round().astype(int))
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    JOB.push("frames", f"Video {src_w}x{src_h}: {total} fotogramas @ {fps:.1f} fps ({total / fps:.1f} s). "
                       f"Muestreando {len(cand_idx)} candidatos...", 0.03)

    candidates: list[Keyframe] = []
    descriptors: list[np.ndarray] = []
    for i, idx in enumerate(cand_idx):
        frame = _read_frame_at(cap, int(idx))
        if frame is None:
            continue
        candidates.append(Keyframe(index=int(idx), time_s=float(idx / fps), rgb=frame, sharpness=_sharpness(frame)))
        descriptors.append(_thumb_descriptor(frame))
        if i % 8 == 0:
            JOB.push("frames", f"Analizando nitidez (varianza Laplaciana)... {i + 1}/{len(cand_idx)}",
                     0.03 + 0.09 * (i + 1) / len(cand_idx))
    cap.release()

    if not candidates:   # último recurso: lectura secuencial del primer fotograma
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise PipelineError("No se pudo decodificar ningún fotograma del video.")
        candidates = [Keyframe(index=0, time_s=0.0, rgb=frame, sharpness=_sharpness(frame))]
        descriptors = [_thumb_descriptor(frame)]

    k = int(max(1, min(k, len(candidates))))
    sharp = np.array([c.sharpness for c in candidates], dtype=np.float64)
    sharp_n = (sharp - sharp.min()) / (np.ptp(sharp) + 1e-9) if len(sharp) > 1 else np.ones_like(sharp)
    D = np.stack(descriptors)                                          # (n, 1728)
    dist = np.linalg.norm(D[:, None, :] - D[None, :, :], axis=-1)      # distancias de apariencia
    dist_n = dist / (dist.max() + 1e-9)

    chosen = [int(np.argmax(sharp_n))]                                 # arranca con el más nítido
    while len(chosen) < k:
        min_d = dist_n[:, chosen].min(axis=1)                          # distancia al keyframe elegido más parecido
        score = 0.65 * min_d + 0.35 * sharp_n
        score[chosen] = -1.0
        dup = min_d < 0.02
        score[dup] = np.minimum(score[dup], -0.5)                      # penaliza casi-duplicados
        nxt = int(np.argmax(score))
        if score[nxt] <= -1.0:
            break
        chosen.append(nxt)

    chosen.sort(key=lambda j: candidates[j].index)                     # orden temporal para encadenar poses
    keyframes = [candidates[j] for j in chosen]
    for kf in keyframes:
        kf.rgb = cv2.cvtColor(_resize_max(kf.rgb, TEXTURE_MAX), cv2.COLOR_BGR2RGB)

    info = {
        "total_frames": total, "fps": round(fps, 3), "duration_s": round(total / fps, 2),
        "source_resolution": [src_w, src_h], "candidates": len(candidates),
        "selected": [kf.index for kf in keyframes],
        "selected_times_s": [round(kf.time_s, 2) for kf in keyframes],
        "sharpness": [round(kf.sharpness, 1) for kf in keyframes],
    }
    JOB.push("frames", f"Keyframes elegidos: {info['selected']} (t = {info['selected_times_s']} s)", 0.14)
    return keyframes, info
