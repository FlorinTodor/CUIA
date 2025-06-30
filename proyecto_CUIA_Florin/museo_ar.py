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

#───────────────────── Renderizador, Escena y Cámara Virtual ─────────────────────

WIDTH, HEIGHT = 640, 480
cap           = cv2.VideoCapture(0)

# 1. Crear el renderizador una sola vez
renderer = pyrender.OffscreenRenderer(WIDTH, HEIGHT)

# 2. Crear la escena una sola vez
scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])

# 3. Crear la cámara virtual (intrínseca) una sola vez
# Usamos los parámetros de tu cámara real
cam_virt = pyrender.IntrinsicsCamera(fx=cam_matrix[0, 0], fy=cam_matrix[1, 1],
                                     cx=cam_matrix[0, 2], cy=cam_matrix[1, 2],
                                     znear=0.01, zfar=100.0)
cam_node = scene.add(cam_virt, pose=np.eye(4)) # Añadimos a la escena, la pose se actualizará luego

# 4. Añadir una luz a la escena una sola vez
light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.0)
scene.add(light, pose=np.eye(4))

# 5. Crear un NODO CONTENEDOR para el modelo. Lo reutilizaremos.
model_node = scene.add(pyrender.Mesh.from_trimesh([]), pose=np.eye(4))
# ───────────────────── OpenCV util ───────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 2----- en tu overlay_modelo (museo_ar.py) ────────────────────────

## METODO NO FUNCIONAL
# ESTA FUNCIÓN ESTÁ BIEN Y AHORA LA VAMOS A USAR
def cv_to_gl_pose(rvec, tvec):
    """
    Convierte la pose de OpenCV (rvec, tvec) a la matriz de pose 4x4 para
    la cámara en OpenGL/Pyrender, siguiendo la lógica del notebook del profesor.
    """
    # 1. Crear la matriz de pose del marcador respecto a la cámara (estilo OpenCV)
    #    pose = [ R | t ]
    #         [ 0 | 1 ]
    pose_marker_cv = np.eye(4, dtype=np.float32)
    pose_marker_cv[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose_marker_cv[:3, 3] = tvec.ravel()

    # 2. Invertir los ejes Y y Z de la matriz de POSE COMPLETA.
    #    Esto es equivalente a rotar 180 grados sobre el eje X.
    #    [x, y, z] -> [x, -y, -z]
    pose_marker_cv[[1, 2]] *= -1

    # 3. La pose de la CÁMARA es la INVERSA de la pose del MARCADOR.
    #    Si el mundo está en la pose M respecto a la cámara, la cámara
    #    está en la pose M^-1 respecto al mundo.
    cam_pose = np.linalg.inv(pose_marker_cv)

    return cam_pose

def overlay_modelo(frame, mesh, cam_pose):
    """
    Renderiza un modelo en la escena global y lo superpone al frame.
    REUTILIZA los nodos existentes de la cámara y del modelo.
    """
    # 1. Actualizar el mesh del nodo del modelo
    scene.set_pose(model_node, pose=np.eye(4)) # Aseguramos que el modelo esté en el origen
    model_node.mesh = mesh

    # 2. Actualizar la pose de la CÁMARA VIRTUAL
    scene.set_pose(cam_node, pose=cam_pose)

    # 3. Renderizar la escena
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

    # 4. Lógica de Alpha Blending (la tuya es perfecta)
    mask = (depth > 0).astype(np.uint8) * 255
    if color.shape[2] == 3:
        color = np.dstack((color, mask))
    else:
        color[:, :, 3] = mask
        
    return alphaBlending(color, frame)

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
        """
        El bucle for recorre los IDs detectados y actualiza el estado:
        - Si el ID es el mismo que el último marcador visible, incrementa el contador.
        - Si el ID es diferente, reinicia el contador a 1.
        - Si el contador alcanza el umbral de visibilidad, renderiza el modelo.
        - Si no hay IDs detectados, decrementa el contador y resetea el marcador visible.
        """
        for i, id_arr in enumerate(ids):
            mid = int(id_arr[0])
            contador_vis  = contador_vis+1 if mid==marcador_visible else 1
            marcador_visible = mid
            zona_detectada  = ZONA_POR_ID.get(mid, zona_detectada)

            if contador_vis >= UMBRAL_VISIBILIDAD:
                mesh = modelos_precargados.get(mid)
                if mesh is not None:
                    # 1. Calcular la pose de la cámara usando la función correcta
                   # (A) Asegúrate de que rvecs[i] y tvecs[i] se usan aquí
                    current_rvec = rvecs[i]
                    current_tvec = tvecs[i]

                    # (B) Se calcula una NUEVA pose en CADA fotograma
                    camera_pose = cv_to_gl_pose(current_rvec, current_tvec)
                    
                    # (C) Se pasa esta nueva pose a la función de overlay
                    frame = overlay_modelo(frame, mesh, camera_pose)
                    
                    # El resto es para depuración visual
                    cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs,
                                      current_rvec, current_tvec, 0.05, 3)
                    cv2.polylines(frame, [esquinas[i].astype(int)],
                                  True, (0,255,0), 2)

    else:
        contador_vis = max(0, contador_vis-1)
        if contador_vis == 0:
            marcador_visible = None
            # Ocultamos el modelo asignando una malla vacía
            model_node.mesh = pyrender.Mesh.from_trimesh([])

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
