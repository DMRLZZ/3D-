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


# --------------------------------------------------------------------------------------
# 2. Profundidad métrica monocular (Depth Anything V2 · HuggingFace transformers)
# --------------------------------------------------------------------------------------
class DepthEstimator:
    """
    Singleton perezoso. Backend 'local' usa transformers + torch (CPU o CUDA).
    Backend 'remote' usa la Inference API de HuggingFace (requiere HF_TOKEN) para máquinas sin GPU/torch.
    Devuelve profundidad en METROS (HxW float32). Para modelos relativos (MiDaS, DA-V2 no métrico)
    convierte la disparidad a profundidad y la escala heurísticamente a una habitación típica.
    """

    _instance: Optional["DepthEstimator"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.is_metric = "metric" in DEPTH_MODEL.lower()
        self.backend = DEPTH_BACKEND
        self.load_time = 0.0

    @classmethod
    def get(cls) -> "DepthEstimator":
        with cls._lock:
            if cls._instance is None:
                cls._instance = DepthEstimator()
            return cls._instance

    def ensure_loaded(self) -> None:
        if self.backend == "remote" or self.model is not None:
            return
        t0 = time.time()
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:  # pragma: no cover
            raise PipelineError(f"Faltan dependencias de inferencia ({e}). Ejecuta: pip install -r requirements.txt") from e
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            torch.set_num_threads(max(1, (os.cpu_count() or 4)))
        JOB.push("depth", f"Cargando modelo {DEPTH_MODEL} en {self.device.upper()}...", 0.16)
        try:
            self.processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL, token=HF_TOKEN)
            self.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, token=HF_TOKEN).to(self.device).eval()
        except Exception as e:
            raise PipelineError(f"No se pudo cargar el modelo de profundidad {DEPTH_MODEL}: {e}. "
                                f"Revisa la conexión a HuggingFace o define DEPTH_BACKEND=remote con HF_TOKEN.") from e
        self.load_time = time.time() - t0
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        JOB.push("depth", f"Modelo listo en {self.load_time:.1f} s ({n_params:.1f} M parámetros)", 0.2)

    # ---- inferencia -------------------------------------------------------------
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        raw = self._predict_remote(rgb) if self.backend == "remote" else self._predict_local(rgb)
        raw = cv2.resize(raw.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        return self._to_metric(raw)

    def _predict_local(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        self.ensure_loaded()
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model(**inputs)
        pred = out.predicted_depth
        if pred.ndim == 3:
            pred = pred[0]
        return pred.detach().float().cpu().numpy()

    def _predict_remote(self, rgb: np.ndarray) -> np.ndarray:
        if not HF_TOKEN:
            raise PipelineError("DEPTH_BACKEND=remote requiere la variable HF_TOKEN.")
        try:
            from huggingface_hub import InferenceClient
        except ImportError as e:
            raise PipelineError("Instala huggingface_hub para usar el backend remoto.") from e
        client = InferenceClient(token=HF_TOKEN)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
        model_id = DEPTH_MODEL if "metric" not in DEPTH_MODEL.lower() else "depth-anything/Depth-Anything-V2-Small-hf"
        result = client.depth_estimation(buf.getvalue(), model=model_id)
        img = result if isinstance(result, Image.Image) else Image.open(io.BytesIO(result))
        arr = np.asarray(img).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        self.is_metric = False   # la API devuelve un mapa relativo normalizado
        return arr

    def _to_metric(self, raw: np.ndarray) -> np.ndarray:
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        if self.is_metric:
            depth = raw
        else:
            # Modelos relativos devuelven disparidad (inversa). Convertimos y escalamos para que la
            # mediana de la escena quede a ~2.5 m: heurística razonable para interiores.
            disp = np.clip(raw, 1e-3, None)
            depth = 1.0 / disp
            med = float(np.median(depth[depth > 0])) if np.any(depth > 0) else 1.0
            depth = depth * (2.5 / max(med, 1e-6))
        depth = np.clip(depth, 0.0, MAX_DEPTH_M).astype(np.float32)
        return cv2.bilateralFilter(depth, d=5, sigmaColor=0.08, sigmaSpace=3)   # suaviza ruido, respeta bordes


def estimate_depths(keyframes: list) -> dict:
    est = DepthEstimator.get()
    est.ensure_loaded()
    times = []
    for i, kf in enumerate(keyframes):
        t0 = time.time()
        kf.depth = est.predict(kf.rgb)
        dt = time.time() - t0
        times.append(round(dt, 2))
        valid = kf.depth[kf.depth > 0.05]
        rng = (float(valid.min()), float(valid.max())) if valid.size else (0.0, 0.0)
        JOB.push("depth", f"Keyframe {i + 1}/{len(keyframes)}: profundidad {rng[0]:.2f}-{rng[1]:.2f} m en {dt:.2f} s",
                 0.2 + 0.3 * (i + 1) / len(keyframes))
    return {"model": DEPTH_MODEL, "backend": est.backend, "device": est.device, "metric": est.is_metric,
            "inference_s": times, "model_load_s": round(est.load_time, 2)}


# --------------------------------------------------------------------------------------
# 3. Cámara e intrínsecos + registro multi-vista (ORB + PnP RANSAC)
# --------------------------------------------------------------------------------------
def intrinsics_for(w: int, h: int) -> np.ndarray:
    """Matriz K estimada a partir del FOV horizontal del smartphone (píxeles cuadrados, centro óptico en el centro)."""
    fx = 0.5 * w / math.tan(math.radians(CAMERA_HFOV_DEG) / 2.0)
    return np.array([[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def unproject(u: np.ndarray, v: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Desproyección pinhole: X=(u-cx)·Z/fx, Y=(v-cy)·Z/fy (convención OpenCV: X derecha, Y abajo, Z delante)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)


def relative_pose_pnp(kf_a, kf_b, K: np.ndarray) -> Optional[dict]:
    """
    Estima T_b<-a (4x4) que lleva puntos del sistema de la cámara A al de la cámara B.
    Usa correspondencias ORB A<->B, levanta los puntos de A a 3D con su profundidad métrica
    y resuelve PnP con RANSAC contra los píxeles de B. Devuelve None si no hay geometría fiable.
    """
    gray_a = cv2.cvtColor(kf_a.rgb, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(kf_b.rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8, fastThreshold=12)
    kpa, da = orb.detectAndCompute(gray_a, None)
    kpb, db = orb.detectAndCompute(gray_b, None)
    if da is None or db is None or len(kpa) < 30 or len(kpb) < 30:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(da, db, k=2)
    good = [pair[0] for pair in knn if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < 20:
        return None

    pts_a = np.float32([kpa[m.queryIdx].pt for m in good])
    pts_b = np.float32([kpb[m.trainIdx].pt for m in good])
    h, w = kf_a.depth.shape
    ui = np.clip(pts_a[:, 0].round().astype(int), 0, w - 1)
    vi = np.clip(pts_a[:, 1].round().astype(int), 0, h - 1)
    z = kf_a.depth[vi, ui]
    ok_z = (z > 0.15) & (z < MAX_DEPTH_M * 0.95)
    if ok_z.sum() < 20:
        return None
    obj = unproject(pts_a[ok_z, 0].astype(np.float64), pts_a[ok_z, 1].astype(np.float64), z[ok_z].astype(np.float64), K)
    img = pts_b[ok_z].astype(np.float64)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), K, None,
        iterationsCount=1000, reprojectionError=4.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not success or inliers is None or len(inliers) < 15:
        return None
    inl = inliers.ravel()
    # refinamiento Levenberg-Marquardt sobre los inliers
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl].reshape(-1, 1, 3), img[inl].reshape(-1, 1, 2), K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    if not np.all(np.isfinite(R)) or np.linalg.norm(t) > 6.0:     # un paso de >6 m entre keyframes no es plausible
        return None
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return {"T_b_from_a": T, "inliers": int(len(inl)), "matches": int(len(good)),
            "translation_m": round(float(np.linalg.norm(t)), 3),
            "rotation_deg": round(float(np.degrees(np.linalg.norm(rvec))), 1)}


def register_keyframes(keyframes: list, K: np.ndarray) -> tuple[list, list]:
    """
    Encadena poses: mundo = cámara del keyframe 0. Para cada par consecutivo intenta PnP;
    si falla, cae a un abanico rotacional (30° alrededor de Y) para que la escena siga siendo explorable.
    Devuelve lista de matrices 4x4 'mundo<-cámara_i' y estadísticas de registro por keyframe.
    """
    poses = [np.eye(4)]
    stats = [{"method": "origin", "inliers": 0, "matches": 0}]
    for i in range(1, len(keyframes)):
        rel = relative_pose_pnp(keyframes[i - 1], keyframes[i], K)
        if rel is not None:
            pose = poses[-1] @ np.linalg.inv(rel["T_b_from_a"])
            stats.append({"method": "pnp_ransac", **{k: v for k, v in rel.items() if k != "T_b_from_a"}})
            JOB.push("register", f"Keyframe {i} registrado por PnP: {rel['inliers']}/{rel['matches']} inliers, "
                                 f"|t|={rel['translation_m']} m, rot={rel['rotation_deg']} grados",
                     0.5 + 0.08 * i / len(keyframes))
        else:
            ang = math.radians(30.0)
            Ry = np.array([[math.cos(ang), 0, math.sin(ang), 0], [0, 1, 0, 0],
                           [-math.sin(ang), 0, math.cos(ang), 0], [0, 0, 0, 1]])
            pose = poses[-1] @ Ry
            stats.append({"method": "fallback_fan", "inliers": 0, "matches": 0})
            JOB.push("register", f"Keyframe {i}: sin geometría fiable, colocado en abanico (+30 grados)",
                     0.5 + 0.08 * i / len(keyframes))
        poses.append(pose)
    return poses, stats


# --------------------------------------------------------------------------------------
# 4. Desproyección RGB-D → malla texturizada por keyframe
# --------------------------------------------------------------------------------------
@dataclass
class FrameMesh:
    vertices_cam: np.ndarray   # (N,3) coordenadas cámara OpenCV, metros
    faces: np.ndarray          # (M,3) int32
    uv: np.ndarray             # (N,2) convención OpenGL (origen abajo-izquierda)
    colors: np.ndarray         # (N,3) uint8 RGB
    texture: Image.Image       # textura JPEG del keyframe


def build_frame_mesh(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, step: int) -> Optional[FrameMesh]:
    """
    Convierte un par RGB-D en una malla: cada píxel (con zancada `step`) es un vértice
    desproyectado con la pinhole; los vecinos del grid forman dos triángulos por celda.
    Se descartan las celdas que cruzan una discontinuidad de profundidad (bordes de objetos),
    para no "estirar" goma entre primer plano y fondo.
    """
    H, W = depth.shape
    vs = np.arange(0, H, step)
    us = np.arange(0, W, step)
    uu, vv = np.meshgrid(us, vs)                                  # (h, w)
    Z = depth[vv, uu].astype(np.float64)
    valid = np.isfinite(Z) & (Z > 0.05) & (Z < MAX_DEPTH_M)
    if valid.sum() < 16:
        return None

    P = unproject(uu.astype(np.float64), vv.astype(np.float64), Z, K).reshape(-1, 3)
    uv = np.stack([uu / max(W - 1, 1), 1.0 - vv / max(H - 1, 1)], axis=-1).reshape(-1, 2)
    colors = rgb[vv, uu].reshape(-1, 3)

    h, w = Z.shape
    idx = np.arange(h * w).reshape(h, w)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    za, zb, zc, zd = Z[:-1, :-1], Z[:-1, 1:], Z[1:, :-1], Z[1:, 1:]
    cell_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    zmin = np.minimum(np.minimum(za, zb), np.minimum(zc, zd))
    zmax = np.maximum(np.maximum(za, zb), np.maximum(zc, zd))
    jump_tol = np.maximum(0.06, 0.08 * zmin)                      # 8 % de la profundidad (mín. 6 cm)
    ok = cell_valid & ((zmax - zmin) < jump_tol)
    if ok.sum() < 8:
        return None
    # dos triángulos por celda, orientados CCW vistos desde la cámara (tras la conversión a OpenGL)
    tri1 = np.stack([a[ok], c[ok], b[ok]], axis=-1)
    tri2 = np.stack([b[ok], c[ok], d[ok]], axis=-1)
    faces = np.concatenate([tri1, tri2], axis=0).astype(np.int64)

    # compactar: conservar solo vértices válidos (mantiene el orden row-major para el frontend)
    keep = valid.reshape(-1)
    remap = -np.ones(h * w, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    faces = remap[faces]
    faces = faces[(faces >= 0).all(axis=1)].astype(np.int32)

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=88)
    buf.seek(0)
    texture = Image.open(buf)
    texture.load()
    return FrameMesh(vertices_cam=P[keep].astype(np.float32), faces=faces, uv=uv[keep].astype(np.float32),
                     colors=colors[keep].astype(np.uint8), texture=texture)


# --------------------------------------------------------------------------------------
# 5. Fusión, exportación (GLB + PLY) y metadata
# --------------------------------------------------------------------------------------
CV_TO_GL = np.diag([1.0, -1.0, -1.0])   # OpenCV (Y abajo, Z delante) → OpenGL/three.js (Y arriba, Z hacia el espectador)


def export_scene(frame_meshes: list, poses: list, keyframes: list, K: np.ndarray, step: int) -> dict:
    import trimesh

    scene = trimesh.Scene()
    all_pts, all_cols = [], []
    total_v, total_f = 0, 0
    for i, (fm, pose) in enumerate(zip(frame_meshes, poses)):
        if fm is None:
            continue
        Vw = (pose[:3, :3] @ fm.vertices_cam.T).T + pose[:3, 3]          # cámara_i → mundo (OpenCV)
        Vgl = (Vw @ CV_TO_GL.T).astype(np.float32)                        # → OpenGL
        mesh = trimesh.Trimesh(vertices=Vgl, faces=fm.faces, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(uv=fm.uv, image=fm.texture)
        scene.add_geometry(mesh, node_name=f"keyframe_{i}", geom_name=f"keyframe_{i}")
        all_pts.append(Vgl)
        all_cols.append(fm.colors)
        total_v += len(Vgl)
        total_f += len(fm.faces)
        JOB.push("mesh", f"Keyframe {i}: {len(Vgl):,} vértices · {len(fm.faces):,} triángulos", 0.62 + 0.1 * (i + 1) / len(frame_meshes))

    if total_v == 0:
        raise PipelineError("La reconstrucción no produjo geometría válida (¿video demasiado oscuro o uniforme?).")

    pts = np.concatenate(all_pts)
    cols = np.concatenate(all_cols)
    JOB.push("export", f"Exportando GLB texturizado ({total_v:,} vértices, {total_f:,} triángulos)...", 0.76)
    glb_bytes = scene.export(file_type="glb")
    MODEL_GLB.write_bytes(glb_bytes)
    JOB.push("export", f"GLB escrito: {len(glb_bytes) / 1e6:.1f} MB", 0.86)

    JOB.push("export", "Exportando nube de puntos PLY con color real...", 0.88)
    cloud = trimesh.PointCloud(vertices=pts, colors=np.concatenate([cols, np.full((len(cols), 1), 255, np.uint8)], axis=1))
    MODEL_PLY.write_bytes(cloud.export(file_type="ply"))

    # --- métricas espaciales reales ---
    bb_min, bb_max = pts.min(axis=0), pts.max(axis=0)
    dims = (bb_max - bb_min)
    floor_y = float(np.percentile(pts[:, 1], 2.0))
    ceil_y = float(np.percentile(pts[:, 1], 98.0))
    footprint = float(max(dims[0] * dims[2], 1e-6))
    cams = []
    for i, pose in enumerate(poses):
        pos = (CV_TO_GL @ pose[:3, 3]).tolist()
        fwd = (CV_TO_GL @ (pose[:3, :3] @ np.array([0.0, 0.0, 1.0]))).tolist()
        cams.append({"index": i, "position": [round(x, 4) for x in pos], "forward": [round(x, 4) for x in fwd]})

    return {
        "vertices": int(total_v), "triangles": int(total_f), "keyframes_meshed": int(sum(fm is not None for fm in frame_meshes)),
        "glb_bytes": len(glb_bytes), "ply_bytes": MODEL_PLY.stat().st_size,
        "bbox_min": [round(float(x), 3) for x in bb_min], "bbox_max": [round(float(x), 3) for x in bb_max],
        "dimensions_m": {"width": round(float(dims[0]), 2), "height": round(float(dims[1]), 2), "depth": round(float(dims[2]), 2)},
        "floor_y": round(floor_y, 3), "ceiling_y": round(ceil_y, 3),
        "eye_height_m": round(float(min(max(0.0 - floor_y, 0.9), 2.2)), 3),
        "point_density_per_m2": round(total_v / footprint, 1),
        "mesh_step_px": step, "intrinsics": {"fx": round(float(K[0, 0]), 2), "fy": round(float(K[1, 1]), 2),
                                             "cx": round(float(K[0, 2]), 2), "cy": round(float(K[1, 2]), 2),
                                             "hfov_deg": CAMERA_HFOV_DEG},
        "cameras": cams,
    }


def save_previews(keyframes: list) -> list:
    """Guarda miniaturas RGB y mapas de profundidad coloreados (evidencia visual del pipeline)."""
    for old in PREVIEW_DIR.glob("*"):
        old.unlink(missing_ok=True)
    out = []
    for i, kf in enumerate(keyframes):
        rgb_small = _resize_max(kf.rgb, 480)
        cv2.imwrite(str(PREVIEW_DIR / f"frame_{i}.jpg"), cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        d = kf.depth
        valid = d[d > 0.05]
        lo, hi = (float(np.percentile(valid, 1)), float(np.percentile(valid, 99))) if valid.size else (0.0, 1.0)
        norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        colored = cv2.applyColorMap((255 - norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)   # cerca = claro
        cv2.imwrite(str(PREVIEW_DIR / f"depth_{i}.jpg"), _resize_max(colored, 480), [cv2.IMWRITE_JPEG_QUALITY, 82])
        out.append({"index": i, "frame": f"/api/preview/frame/{i}", "depth": f"/api/preview/depth/{i}",
                    "depth_range_m": [round(lo, 2), round(hi, 2)], "video_frame": kf.index, "time_s": round(kf.time_s, 2)})
    return out


# --------------------------------------------------------------------------------------
# Orquestación del pipeline
# --------------------------------------------------------------------------------------
def run_pipeline(video_path: Path, n_keyframes: int, step: int) -> dict:
    t_start = time.time()
    timings: dict = {}

    t0 = time.time()
    keyframes, video_info = extract_keyframes(video_path, n_keyframes)
    timings["keyframes_s"] = round(time.time() - t0, 2)

    t0 = time.time()
    depth_info = estimate_depths(keyframes)
    timings["depth_s"] = round(time.time() - t0, 2)

    h, w = keyframes[0].rgb.shape[:2]
    K = intrinsics_for(w, h)
    JOB.push("register", f"Intrínsecos estimados: fx={K[0, 0]:.1f} px, HFOV={CAMERA_HFOV_DEG:.0f} grados, textura {w}x{h}", 0.5)
    t0 = time.time()
    poses, reg_stats = register_keyframes(keyframes, K) if len(keyframes) > 1 else ([np.eye(4)], [{"method": "origin", "inliers": 0, "matches": 0}])
    timings["registration_s"] = round(time.time() - t0, 2)

    JOB.push("mesh", f"Desproyectando RGB-D a malla (zancada {step} px)...", 0.6)
    t0 = time.time()
    frame_meshes = [build_frame_mesh(kf.rgb, kf.depth, K, step) for kf in keyframes]
    timings["mesh_s"] = round(time.time() - t0, 2)

    t0 = time.time()
    geo = export_scene(frame_meshes, poses, keyframes, K, step)
    timings["export_s"] = round(time.time() - t0, 2)

    previews = save_previews(keyframes)
    timings["total_s"] = round(time.time() - t_start, 2)

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "job_id": JOB.id,
        "video": video_info, "depth": depth_info, "registration": reg_stats,
        "geometry": geo, "previews": previews, "timings": timings,
        "model_url": "/api/model", "pointcloud_url": "/api/model?format=ply", "video_url": "/api/video",
    }
    METADATA_JSON.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    JOB.push("done", f"Gemelo digital listo en {timings['total_s']} s: {geo['vertices']:,} vértices, "
                     f"{geo['dimensions_m']['width']}x{geo['dimensions_m']['depth']} m", 1.0)
    return meta


def _pipeline_thread(video_path: Path, n_keyframes: int, step: int) -> None:
    try:
        run_pipeline(video_path, n_keyframes, step)
        JOB.finish()
    except PipelineError as e:
        log.warning("Pipeline detenido: %s", e)
        JOB.push("error", str(e))
        JOB.finish(error=str(e))
    except Exception as e:  # noqa: BLE001 - cualquier fallo inesperado debe reportarse, no tumbar el servidor
        log.exception("Fallo inesperado en el pipeline")
        msg = f"Error interno del pipeline: {type(e).__name__}: {e}"
        JOB.push("error", msg)
        JOB.finish(error=msg)
