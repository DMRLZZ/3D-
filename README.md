# SPATIAL TWIN · de un video de móvil a un gemelo digital explorable

Sistema full-stack que convierte un video grabado con un smartphone en un entorno 3D
texturizado con las imágenes reales de la cámara, explorable en primera persona (estilo FPS)
desde el navegador. **Sin mocks ni timers falsos**: cada línea del log que ves en la interfaz
es una etapa real ejecutándose en el backend.

```
video.mp4 ─► keyframes ─► profundidad métrica ─► registro PnP ─► malla RGB-D ─► GLB ─► WebGL FPS
            (OpenCV)     (Depth Anything V2)     (ORB+RANSAC)    (pinhole+UV)  (trimesh)  (Three.js)
```

## Ejecución rápida (3 comandos)

```powershell
pip install -r requirements.txt ; pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
cd backend ; python main.py
start http://localhost:8000
```

En macOS/Linux sustituye `start` por `open`/`xdg-open`. Con GPU NVIDIA usa `--index-url https://download.pytorch.org/whl/cu126`
en lugar de `cpu`: el backend detecta CUDA automáticamente. La primera ejecución descarga el modelo (~100 MB).

También puedes usar `./run.ps1` (Windows) o `./run.sh` (Unix), que hacen exactamente eso.

## Qué hace el backend (`backend/main.py`)

| Etapa | Técnica | Salida real |
|---|---|---|
| 1. Keyframes | Muestreo uniforme de candidatos, nitidez por varianza del Laplaciano y selección greedy que maximiza cobertura visual | 1-5 fotogramas |
| 2. Profundidad | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` vía `transformers` (CPU/CUDA). Backend remoto opcional por `HF_TOKEN` | mapa de profundidad en **metros** |
| 3. Intrínsecos | Pinhole estimada a partir del FOV horizontal del móvil (70° sobre el lado largo) | `fx, fy, cx, cy` |
| 4. Registro | ORB → matches → puntos 3D métricos → `solvePnPRansac` + refinamiento LM. Si no hay solape, panorama contiguo por FOV | poses 4×4 por keyframe |
| 5. Malla | Desproyección `X=(u-cx)·Z/fx, Y=(v-cy)·Z/fy`, grid de triángulos con UV, descarte de discontinuidades | malla texturizada por keyframe |
| 6. Export | `trimesh` → GLB multi-material con texturas JPEG reales + PLY con color por vértice | `output/scene.glb`, `output/scene.ply`, `metadata.json` |

### API

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/scan?frames=4&step=3` | Sube el video (multipart `file`) y lanza el pipeline en segundo plano. Devuelve `202` con `job_id`. |
| `GET` | `/api/status` | Estado real: etapa, progreso 0-1, log con timestamps, error. |
| `GET` | `/api/model?format=glb\|ply` | Modelo 3D generado. |
| `GET` | `/api/metadata` | Vértices, triángulos, dimensiones de la sala, altura del suelo, poses de cámara, tiempos por etapa. |
| `GET` | `/api/preview/{frame\|depth}/{i}` | Miniaturas RGB y mapas de profundidad de cada keyframe. |
| `GET` | `/api/video` | Video fuente para la vista "Realidad vs. Gemelo". |
| `GET` | `/api/health` | Modelo, dispositivo (CPU/CUDA), versión de torch. |
| `DELETE` | `/api/model` | Borra el gemelo actual entre demos. |

CORS está abierto; el frontend se sirve desde el mismo origen (`/`) o puede abrirse como archivo
apuntando al backend con `index.html?api=http://localhost:8000`.

### Variables de entorno útiles

`DEPTH_MODEL` (por defecto el métrico interior; `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` para exteriores,
`Intel/dpt-hybrid-midas` también soportado), `DEPTH_BACKEND=remote` + `HF_TOKEN` (inferencia serverless sin torch local),
`CAMERA_HFOV` (70), `MAX_KEYFRAMES` (4), `MESH_STEP` (3), `TEXTURE_MAX` (960), `MAX_DEPTH` (12 m), `PORT` (8000).

## Qué hace el frontend (`frontend/index.html`, un solo archivo)

- **Visor WebGL (Three.js r170)** que carga el GLB real: `MeshStandardMaterial` con la textura fotográfica, normales suaves,
  sombras PCF, linterna dinámica (`L`) y un shader de revelado tipo barrido láser al entrar.
- **Modo nube de puntos (`M`)**: shader propio que muestrea la misma textura con tamaño de partícula adaptativo a la distancia
  y al espaciado local de cada vértice para tapar huecos.
- **Controlador FPS**: `PointerLockControls`, `WASD`, `Shift` para correr, `Espacio` salto, gravedad, plano de sustentación en la
  altura real del suelo estimada y colisión suave con paredes mediante rejilla de ocupación XZ (deslizamiento). `N` activa vuelo libre.
- **Post-procesado (`P`)**: bloom suave + aberración cromática + viñeta + grano.
- **HUD de telemetría real**: vértices, triángulos, densidad (pts/m²), dimensiones de la sala, rango de profundidad, keyframes usados,
  resultado del registro PnP, modelo/dispositivo/tiempo de inferencia, posición, brújula y FPS.
- **Realidad vs. Gemelo**: el video original en miniatura y las parejas RGB ⇄ profundidad de cada keyframe.
- Si ya existe un gemelo procesado, la landing ofrece entrar sin re-procesar (ideal para un demo de 30 segundos).

## Guion de demo (30 s)

1. Arrastra el video del salón → el log muestra en vivo keyframes, profundidad métrica y malla (≈40 s en CPU con 4K, ≈15 s en 1080p o GPU).
2. Pulsa **Entrar al gemelo**: barrido láser de revelado y HUD activo.
3. Camina con `WASD`, enciende la linterna con `L`, alterna a nube de puntos con `M`, y muestra la miniatura "Realidad" frente al gemelo.

## Consejos de captura

Recorrido lento, buena luz, sin giros bruscos y con solape entre vistas: así el registro PnP encadena las cámaras.
Si el video es muy rápido, los keyframes se colocan en panorama contiguo y la escena sigue siendo explorable.
Videos 1080p se procesan unas 4 veces más rápido que 4K.

## Estructura

```
backend/main.py        pipeline completo + API FastAPI (sirve también el frontend)
backend/output/        scene.glb, scene.ply, metadata.json, previews/   (generado)
backend/uploads/       videos recibidos                                 (generado)
frontend/index.html    cliente WebGL completo (HTML + CSS + JS)
requirements.txt       dependencias Python
run.ps1 / run.sh       instalación + arranque en un comando
```
