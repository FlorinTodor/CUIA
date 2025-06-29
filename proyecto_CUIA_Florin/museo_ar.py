#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Museo AR – bucle principal
• Renderiza modelos 3D sobre marcadores ArUco (versión «overlay_modelo»).
• Souvenir JPG (“museo souvenir”) → e-mail del visitante.
"""
# ───────────────────────── IMPORTS BÁSICOS ──────────────────────────
import os, sys, signal, queue, threading, tempfile, subprocess, hashlib
os.environ["PYOPENGL_PLATFORM"]  = "osmesa"          # pyrender off-screen
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"             # silenciar TF/Mediapipe

import logging ; logging.getLogger("absl").setLevel(logging.ERROR)
import cv2, numpy as np, pyrender
from gtts import gTTS

from cuia            import alphaBlending
from reconocimiento  import reconocer_zona, detectar_marcadores
from models          import modelos_precargados
from camera          import cameraMatrix as cam_matrix, distCoeffs as dist_coeffs
from visitor_extras  import souvenir                        # envía por e-mail
import voice, qa, contextlib, json

# ───────────────────── GUI login (Tkinter) ───────────────────────
from login_ui import get_user
user = get_user()                           # (username, email) ó None
if user is None:
    sys.exit(0)
username, email_addr = user
current_user = {"username": username, "email": email_addr}

# ───────────────────── Datos del museo ───────────────────────────
with open("museo_data.json", encoding="utf-8") as f:
    _DATA = json.load(f)
ZONA_POR_ID = {int(k): v for k, v in _DATA["zonas_por_id"].items()}

# ───────────────────── PARÁMETROS ────────────────────────────────
MARKER_SIZE        = 0.05   # m
UMBRAL_VISIBILIDAD = 5
UMBRAL_ESTABILIDAD = 10
FACE_TTL           = 30

# ───────────────────── VOZ / TTS ─────────────────────────────────
cmd_q, tts_q = queue.Queue(), queue.Queue() 
stop_evt     = threading.Event()
voice.start_listener(cmd_q, stop_evt)   # wake-word = «museo»

# ───────────────────── TTS worker ──────────────────────────────
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

# ───────────────────── ESTADO DE SESIÓN ──────────────────────────
zona_estable       = "Zona desconocida"
marcador_visible   = None
contador_vis       = 0
contador_zona      = {}
face_bbox, current_facehash, face_ttl = None, None, 0

# ───────────────────── Renderizador y cámara ─────────────────────
WIDTH, HEIGHT = 640, 480
renderer      = pyrender.OffscreenRenderer(WIDTH, HEIGHT)
cap           = cv2.VideoCapture(0)

# ───────────────────── OpenCV util ───────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 2----- en tu overlay_modelo (museo_ar.py) ────────────────────────


def overlay_modelo(frame, esquinas, rvec, tvec, mesh):
    """Renderiza el modelo 3D sobre la imagen usando pyrender."""
    try:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3], _ = cv2.Rodrigues(rvec)
        pose[:3, 3] = tvec
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0],
                               ambient_light=[.4, .4, .4, 1])
        scene.add(mesh, pose=pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=5.),
                  pose=np.eye(4))

        cam = pyrender.IntrinsicsCamera(cam_matrix[0, 0], cam_matrix[1, 1],
                                        cam_matrix[0, 2], cam_matrix[1, 2],
                                        0.005, 10)
        cam_pose = np.eye(4); cam_pose[2, 3] = 0.2
        scene.add(cam, pose=cam_pose)

        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        if color.shape[2] == 3:                       # sin alpha → añádelo
            h, w, _ = color.shape
            color = np.dstack((color, np.zeros((h, w), np.uint8)))
        color[:, :, 3] = (depth > 0).astype(np.uint8) * 255
        return cv2.cvtColor(alphaBlending(color, frame), cv2.COLOR_BGRA2BGR)
    except Exception:
        return frame

# ───────────────────── Función de cierre ─────────────────────────
def cerrar(*_):
    stop_evt.set(); tts_q.put(None)
    with contextlib.suppress(Exception): cap.release()
    with contextlib.suppress(Exception): renderer.delete()
    cv2.destroyAllWindows(); sys.exit(0)

signal.signal(signal.SIGINT,  cerrar)
signal.signal(signal.SIGTERM, cerrar)

# ═══════════════════════ BUCLE PRINCIPAL ═════════════════════════
while True:
    if stop_evt.is_set(): cerrar()

    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1 ◉ Reconocimiento de zona
    zona_detectada = reconocer_zona(gray)

    # 2 ◉ ArUco
    esquinas, ids = detectar_marcadores(gray)
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            esquinas, MARKER_SIZE, cam_matrix, distCoeffs=dist_coeffs)

        for i, id_arr in enumerate(ids):
            mid = int(id_arr[0])
            contador_vis  = contador_vis+1 if mid==marcador_visible else 1
            marcador_visible = mid
            zona_detectada  = ZONA_POR_ID.get(mid, zona_detectada)

            if contador_vis >= UMBRAL_VISIBILIDAD:
                mesh = modelos_precargados.get(mid)
                if mesh is not None:
                    frame = overlay_modelo(frame, esquinas[i],
                                           rvecs[i], tvecs[i], mesh)
                    cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs,
                                      rvecs[i], tvecs[i], 0.05, 3)
                    cv2.polylines(frame, [esquinas[i].astype(int)],
                                  True, (0,255,0), 2)
    else:
        contador_vis = max(0, contador_vis-1)
        if contador_vis == 0:
            marcador_visible = None

    # 3 ◉ Detección facial (bbox souvenir)
    if face_ttl > 0: face_ttl -= 1
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces):
        x,y,w,h   = faces[0]
        face_bbox = (x,y,w,h)
        face_roi  = frame[y:y+h, x:x+w]
        new_hash  = hashlib.sha1(cv2.resize(face_roi,(32,32)).tobytes()).hexdigest()
        face_ttl  = FACE_TTL if new_hash != current_facehash else min(FACE_TTL, face_ttl+1)
        current_facehash = new_hash

    # 4 ◉ Estabilizar zona
    if zona_detectada:
        contador_zona[zona_detectada] = contador_zona.get(zona_detectada,0)+1
        if contador_zona[zona_detectada] >= UMBRAL_ESTABILIDAD:
            zona_estable  = zona_detectada
            contador_zona = {zona_estable: contador_zona[zona_estable]}
    else:
        for z in list(contador_zona):
            contador_zona[z] = max(0, contador_zona[z]-1)

    # 5 ◉ Comandos de voz
    while not cmd_q.empty():
        q = cmd_q.get().strip().lower()
        if q == "__awoken__":
            tts_q.put("Te escucho"); continue

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

    # 6 ◉ HUD
    cv2.putText(frame, f"Te encuentras en: {zona_estable}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Museo AR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cerrar()
