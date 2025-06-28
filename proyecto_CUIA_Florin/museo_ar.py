#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Museo AR – bucle principal

• Souvenir PNG cuando se oye «museo souvenir».
• Preguntas por voz usando qa.py (voice.py).
"""

import os, signal, sys, queue, threading, tempfile, subprocess, hashlib, time
os.environ["PYOPENGL_PLATFORM"] = "osmesa"         # pyrender off-screen
# Evitar mensajes de TensorFlow y Deep Face
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # 0=TODO · 1=INFO · 2=WARNING · 3=ERROR

# ── SILENCIAR COMPLETAMENTE STDERR DESDE ESTE PUNTO ──────────────────────
#DEVNULL_FD = os.open(os.devnull, os.O_WRONLY)
#os.dup2(DEVNULL_FD, 2)        # todo lo que vaya a stderr se descarta

import logging; logging.getLogger("absl").setLevel(logging.ERROR)
import cv2, numpy as np, pyrender
from gtts import gTTS
from cuia          import alphaBlending
from reconocimiento import reconocer_zona, detectar_marcadores
from models        import modelos_precargados
from camera        import cameraMatrix as cam_matrix, distCoeffs as dist_coeffs
import voice, qa
from visitor_extras import souvenir  
import contextlib
import cuia
import json
import reconocer_caras as rc
import user_data, telegram_bot
with open("museo_data.json", encoding="utf-8") as f:
    _DATA  = json.load(f)
    
ZONA_POR_ID: dict[int, str] = {int(k): v for k, v in _DATA["zonas_por_id"].items()}

# ────────── parámetros ──────────────────────────────────────────────
UMBRAL_VISIBILIDAD, UMBRAL_ESTABILIDAD = 5, 10


# ────────── voz / TTS ───────────────────────────────────────────────
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
            try:
                os.unlink(f.name)
            except FileNotFoundError:
                pass

threading.Thread(target=_tts_worker, daemon=True).start()

# ────────── estado de sesión ───────────────────────────────────────
zona_estable       = "Zona desconocida"
marcador_visible   = None
contador_vis       = 0
contador_zona      = {}

face_bbox        = None        # (x,y,w,h) última caja válida
current_facehash = None
FACE_TTL         = 30          # frames que “vive” la misma cara
face_ttl         = 0




# ────────── login / registro ─────────────────────────────────────
def login_or_register(cap):
    print("⏳ Buscando rostro para login/registro...")
    while True:
        ok, frame = cap.read()
        if not ok:  raise RuntimeError("⛔ No hay cámara")

        auth = rc.authenticate(frame)
        if auth:
            username, info = auth
            print(f"✅ Bienvenido de nuevo, {username}")
            return username, info["telegram"]

        cv2.putText(frame, "Pulsa 'r' para registrarte", (60,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Login", frame)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('r'):          # registro
            username = input("Introduce tu nombre de usuario: ").strip()
            telegram = input("Introduce tu @telegram: ").strip()
            try:
                rc.register(frame, username, telegram)
                print("🆗 Registro completado. ¡Hola,", username, "!")
                return username, telegram
            except ValueError as e:
                print(e)
                continue

# ────────── cámara + renderer ──────────────────────────────────────
cap       = cv2.VideoCapture(0)
renderer  = pyrender.OffscreenRenderer(640, 480)
# login ───────────────────────────────────────────────────────────
username, tg_handle = login_or_register(cap)
current_user = {"username": username, "telegram": tg_handle}
cv2.destroyWindow("Login")



# ────────── OpenCV util ────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def cv_to_gl_pose(rvec, tvec):
    """Devuelve matriz 4×4 en coordenadas OpenGL a partir de rvec/tvec OpenCV"""
    M          = np.eye(4, dtype=np.float32)
    M[:3, :3]  = cv2.Rodrigues(rvec)[0]
    M[:3, 3]   = tvec.reshape(3)          # FLATTEN: (3,1) → (3,)
    M[[1, 2]] *= -1                       # invierte ejes Y y Z
    return np.linalg.inv(M)               # cámara activa ≡ invertir

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

# ────────── shut-down ordenado ─────────────────────────────────────
def cerrar(*_):
    stop_evt.set(); tts_q.put(None)
    with contextlib.suppress(Exception): cap.release()
    with contextlib.suppress(Exception): renderer.delete()
    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT,  cerrar)
signal.signal(signal.SIGTERM, cerrar)




# ═════════════════════════  bucle principal  ══════════════════════
while True:
    if stop_evt.is_set():
        cerrar()

    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1 ◉ reconocimiento por imagen
    zona_detectada = reconocer_zona(gray)

    # 2 ◉ ArUco
    esquinas, ids = detectar_marcadores(gray)
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            esquinas, 0.05, cam_matrix, distCoeffs=dist_coeffs
        )
        print("tvec:", tvecs[0].ravel(), "  rvec:", rvecs[0].ravel())
        for i, id_arr in enumerate(ids):
            mid           = int(id_arr[0])
            #print("ID ArUco detectado:", mid) # debug
            contador_vis  = contador_vis + 1 if mid == marcador_visible else 1
            marcador_visible = mid
            zona_detectada  = ZONA_POR_ID.get(mid, zona_detectada)

            if contador_vis >= UMBRAL_VISIBILIDAD:
                mesh = modelos_precargados.get(mid)
                if mesh is not None:
                    frame = overlay_modelo(
                        frame, esquinas[i], rvecs[i], tvecs[i], mesh
                    )
                    cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs,
                                  rvecs[i], tvecs[i],
                                  length=0.05, thickness=3)
                    
                    cv2.polylines(frame,
                                  [esquinas[i].astype(int)],
                                  True, (0, 255, 0), 2)
    else:
        contador_vis = max(0, contador_vis - 1)
        if contador_vis == 0:
            marcador_visible = None

    # 3 ◉ detección facial (solo para conocer bbox del souvenir)
    if face_ttl > 0:
        face_ttl -= 1
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces):
        x, y, w, h = faces[0]
        face_bbox  = (x, y, w, h)
        face_roi   = frame[y:y + h, x:x + w]

        # hash para saber si es la misma cara
        new_hash = hashlib.sha1(
            cv2.resize(face_roi, (32, 32)).tobytes()
        ).hexdigest()
        if new_hash != current_facehash:
            current_facehash = new_hash
            face_ttl = FACE_TTL
        else:
            face_ttl = min(FACE_TTL, face_ttl + 1)

        # opcional: dibujar bbox
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)

    # 4 ◉ estabilizar zona
    if zona_detectada:
        c = contador_zona.get(zona_detectada, 0) + 1
        contador_zona[zona_detectada] = c
        if c >= UMBRAL_ESTABILIDAD:
            zona_estable  = zona_detectada
            contador_zona = {zona_estable: c}
    else:
        for z in list(contador_zona):
            contador_zona[z] = max(0, contador_zona[z] - 1)

    # 5 ◉ comandos de voz
    while not cmd_q.empty():
        q = cmd_q.get().strip().lower()
        if q == "__awoken__":
            tts_q.put("Te escucho")
            continue

        if q == "souvenir":
            if face_bbox and face_ttl > 0:
                # El worker de visitor_extras ya lo envía a Telegram
                souvenir.request(frame,
                                 face_bbox,
                                 zona_estable,
                                 current_user["telegram"])
                tts_q.put("¡Souvenir enviado a tu Telegram!")
            else:
                tts_q.put("Necesito ver tu cara para el souvenir")

        # otras preguntas
        respuesta = qa.responder(q, marcador_visible, zona_estable)
        tts_q.put(respuesta or "Lo siento, no tengo respuesta para eso.")

    # 6 ◉ HUD
    cv2.putText(frame,
                f"Te encuentras en: {zona_estable}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2)
    

   

    cv2.imshow("Museo AR", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cerrar()
