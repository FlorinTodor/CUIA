#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Museo AR – bucle principal
• Renderiza modelos 3D sobre marcadores ArUco.
• Souvenir JPG (“museo souvenir”) → e-mail del visitante.
"""

# ────────────────────────── IMPORTS BÁSICOS ────────────────────────────────
import os, sys, signal, queue, threading, tempfile, subprocess, hashlib
os.environ["PYOPENGL_PLATFORM"]  = "osmesa"        # pyrender off-screen
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           # silenciar TF / Mediapipe

import logging ; logging.getLogger("absl").setLevel(logging.ERROR)
import cv2, numpy as np, pyrender
from gtts import gTTS

from cuia            import alphaBlending
from reconocimiento  import reconocer_zona, detectar_marcadores
from models          import modelos_precargados               # mallas cacheadas
from camera          import cameraMatrix as K, distCoeffs as D
from visitor_extras  import souvenir                           # envía por e-mail
import voice, qa, contextlib, json

# ────────────────────────── GUI login (Tkinter) ────────────────────────────
from login_ui import get_user          # (username, email)  ó  None
user = get_user()
if user is None:
    sys.exit(0)                        # usuario canceló
username, email_addr = user
current_user = {"username": username, "email": email_addr}

# ───────────────────── Datos del museo (zonas ↔ id) ────────────────────────
with open("museo_data.json", encoding="utf-8") as f:
    _DATA = json.load(f)
ZONA_POR_ID = {int(k): v for k, v in _DATA["zonas_por_id"].items()}

# ───────────────────────── PARÁMETROS GLOBALES ─────────────────────────────
MARKER_SIZE        = 0.05        # lado del ArUco (m)
UMBRAL_VISIBILIDAD = 5
UMBRAL_ESTABILIDAD = 10
FACE_TTL           = 30          # frames que “vive” la misma cara

# ───────────────────────────── VOZ / TTS ───────────────────────────────────
cmd_q, tts_q = queue.Queue(), queue.Queue()
stop_evt     = threading.Event()
voice.start_listener(cmd_q, stop_evt)          # wake-word = «museo»

def _tts_worker():
    while True:
        txt = tts_q.get()
        if txt is None:
            break
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                gTTS(txt, lang="es").save(f.name)
            subprocess.run(["mpg123", "-q", f.name], check=False)
        finally:
            with contextlib.suppress(Exception):
                os.unlink(f.name)

threading.Thread(target=_tts_worker, daemon=True).start()

# ─────────────────────── ESTADO DE SESIÓN ────────────────────────────────
zona_estable     = "Zona desconocida"
marcador_visible = None
contador_vis     = 0
contador_zona    = {}
face_bbox        = None
current_facehash = None
face_ttl         = 0

# ────────────────── CÁMARA y ESCENA 3D PERMANENTES ───────────────────────
cap      = cv2.VideoCapture(0)

WIDTH, HEIGHT = 640, 480          # ajusta si tu webcam va mejor a 1280×720
renderer = pyrender.OffscreenRenderer(WIDTH, HEIGHT)

scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[.3, .3, .3, 1])

cam_node = scene.add(
    pyrender.IntrinsicsCamera(K[0,0], K[1,1], K[0,2], K[1,2], znear=0.01, zfar=5.0),
    pose=np.eye(4)
)

# luces sencillas
scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=5.0),
          pose=np.eye(4))
scene.add(pyrender.PointLight(color=[1,1,1], intensity=2.0),
          pose=[[1,0,0, .25],
                [0,1,0, .25],
                [0,0,1, .25],
                [0,0,0, 1]])

mesh_nodes: dict[int, pyrender.Node] = {}   # id_ArUco → Node en escena

# ───────────────────────── OpenCV util ────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def cv_to_gl_pose(rvec, tvec) -> np.ndarray:
    """Mat 4×4 (obj→cam) en convención OpenGL."""
    R, _  = cv2.Rodrigues(rvec)
    M     = np.eye(4, dtype=np.float32)
    M[:3,:3] = R
    M[:3, 3] = tvec.reshape(3)
    M[[1,2]] *= -1      # Y,Z → −Y,−Z
    return M

# ──────────────────────── SHUT-DOWN ORDENADO ─────────────────────────────
def cerrar(*_):
    stop_evt.set(); tts_q.put(None)
    with contextlib.suppress(Exception): cap.release()
    with contextlib.suppress(Exception): renderer.delete()
    cv2.destroyAllWindows(); sys.exit(0)

signal.signal(signal.SIGINT,  cerrar)
signal.signal(signal.SIGTERM, cerrar)

# ═════════════════════════ BUCLE PRINCIPAL ════════════════════════════════
while True:
    if stop_evt.is_set():
        cerrar()

    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── 1 · Reconocimiento por imagen ─────────────────────────────────────
    zona_detectada = reconocer_zona(gray)

    # ── 2 · ArUco: detección y poses ─────────────────────────────────────
    corners, ids = detectar_marcadores(gray)
    detecciones = []                    # [(id, rvec, tvec), …]

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                              corners, MARKER_SIZE, K, distCoeffs=D)
        for i, id_arr in enumerate(ids):
            mid = int(id_arr[0])

            detecciones.append((mid, rvecs[i], tvecs[i]))

            # lógica de estabilización / zona
            contador_vis  = contador_vis+1 if mid==marcador_visible else 1
            marcador_visible = mid
            zona_detectada  = ZONA_POR_ID.get(mid, zona_detectada)

            # dibujado auxiliar (opcional)
            cv2.drawFrameAxes(frame, K, D, rvecs[i], tvecs[i], 0.05, 3)
            cv2.polylines(frame, [corners[i].astype(int)], True, (0,255,0), 2)

    else:
        contador_vis = max(0, contador_vis-1)
        if contador_vis == 0:
            marcador_visible = None

    # ── 3 · Actualiza nodos y renderiza en un único paso ─────────────────
    visibles = set()
    for mid, rvec, tvec in detecciones:
        visibles.add(mid)
        if mid not in mesh_nodes:           # añadir la primera vez
            mesh_nodes[mid] = scene.add(modelos_precargados[mid],
                                        pose=cv_to_gl_pose(rvec, tvec))
        else:                               # solo actualiza pose
            scene.set_pose(mesh_nodes[mid], cv_to_gl_pose(rvec, tvec))

    # oculta nodos que dejaron de verse
    for mid in list(mesh_nodes):
        if mid not in visibles:
            scene.remove_node(mesh_nodes[mid])
            del mesh_nodes[mid]

    # render & alpha-blend
    color, depth = renderer.render(scene)   # RGBA por defecto
    if color.shape[2] == 3:                 # asegúrate de tener canal α
        h, w, _ = color.shape
        color = np.dstack((color,
                  (depth > 0).astype(np.uint8)*255))
    else:
        color = color.copy()
        color[:, :, 3] = (depth > 0).astype(np.uint8)*255

    frame = cv2.cvtColor(alphaBlending(color, frame), cv2.COLOR_BGRA2BGR)

    # ── 4 · Detección facial (bbox souvenir) ────────────────────────────
    if face_ttl > 0:
        face_ttl -= 1
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces):
        x, y, w, h = faces[0]
        face_bbox  = (x, y, w, h)
        face_roi   = frame[y:y+h, x:x+w]
        new_hash   = hashlib.sha1(cv2.resize(face_roi,(32,32)).tobytes()).hexdigest()
        face_ttl   = FACE_TTL if new_hash != current_facehash else min(FACE_TTL, face_ttl+1)
        current_facehash = new_hash

    # ── 5 · Estabilizar zona ────────────────────────────────────────────
    if zona_detectada:
        contador_zona[zona_detectada] = contador_zona.get(zona_detectada,0)+1
        if contador_zona[zona_detectada] >= UMBRAL_ESTABILIDAD:
            zona_estable  = zona_detectada
            contador_zona = {zona_estable: contador_zona[zona_estable]}
    else:
        for z in list(contador_zona):
            contador_zona[z] = max(0, contador_zona[z]-1)

    # ── 6 · Comandos de voz ─────────────────────────────────────────────
    while not cmd_q.empty():
        q = cmd_q.get().strip().lower()
        if q == "__awoken__":
            tts_q.put("Te escucho")
            continue

        if q == "souvenir":
            if face_bbox and face_ttl > 0:
                souvenir.request(frame, face_bbox, zona_estable,
                                 current_user["email"])
                tts_q.put("¡Souvenir enviado a tu correo!")
            else:
                tts_q.put("Necesito ver tu cara para el souvenir")
            continue

        respuesta = qa.responder(q, marcador_visible, zona_estable)
        tts_q.put(respuesta or "Lo siento, no tengo respuesta para eso.")

    # ── 7 · HUD ─────────────────────────────────────────────────────────
    cv2.putText(frame, f"Te encuentras en: {zona_estable}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Museo AR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cerrar()
