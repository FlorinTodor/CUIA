import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import cv2
import numpy as np
import speech_recognition as sr
import pyrender
import trimesh
from cuia import popup
from reconocimiento import reconocer_zona, detectar_marcadores, zonas_imagenes
import copy
import traceback
import pyrender

# ➋ Crea UN solo renderer global
renderer = pyrender.OffscreenRenderer(640, 480)

# Modelos 3D reales (formatos .glb)
marcadores_modelos = {
    0: "CUIA_models/pzkpfw_vi_tiger_1.glb",
    1: "CUIA_models/consolidated_b-24_liberator.glb",
    2: "CUIA_models/mp_40_submachine_gun.glb",
}



# Precargar modelos
modelos_precargados = {}

for marcador_id, ruta in marcadores_modelos.items():
    try:
       
        mesh = trimesh.load(ruta)
        print(f"[DEBUG] Modelo {ruta} - Bounds: {mesh.bounds}")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.geometry[list(mesh.geometry.keys())[0]]

        scale_factor = 0.2
        mesh.apply_transform(trimesh.transformations.scale_matrix(scale_factor))
        #print(f"[DEBUG] Añadiendo modelo con pose:\n{pose}")
        render_mesh = pyrender.Mesh.from_trimesh(mesh)
        modelos_precargados[marcador_id] = render_mesh
        print(f"[INFO] Modelo precargado: {ruta}")
    except Exception as e:
        print(f"[ERROR] Fallo al cargar modelo {ruta}: {e}")

# Inicialización reconocimiento de voz
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Inicialización de parámetros de cámara y ArUco
from camara import cameraMatrix as cam_matrix, distCoeffs as dist_coeffs

# Renderizador offscreen para RA
renderer = pyrender.OffscreenRenderer(640, 480)

# Variables de estabilidad
zona_estable = "Zona desconocida"
contador_zona = {}
UMBRAL_ESTABILIDAD = 10
ultimo_marcador_mostrado = None

# Mostrar modelo en overlay RA sobre marcador
def overlay_modelo(frame, rvec, tvec, render_mesh):
    try:
        rvec = np.asarray(rvec, dtype=np.float32).reshape(3)
        tvec = np.asarray(tvec, dtype=np.float32).reshape(3)

        rot_matrix, _ = cv2.Rodrigues(rvec)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_matrix
        pose[:3, 3]  = tvec

        # ⚠️ Primero la escena y la luz
        scene = pyrender.Scene()
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.ambient_light = np.array([0.3, 0.3, 0.3, 1.0])
        scene.add(light, pose=np.eye(4))

        # Solo si hay render_mesh válido, lo añadimos
        if render_mesh:
            scene.add(render_mesh, pose=pose)

        cam = pyrender.IntrinsicsCamera(
            fx=float(cam_matrix[0, 0]), fy=float(cam_matrix[1, 1]),
            cx=float(cam_matrix[0, 2]), cy=float(cam_matrix[1, 2])
        )
        scene.add(cam, pose=np.eye(4, dtype=np.float32))

        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)
        if color.shape[:2] != frame.shape[:2]:
            color = cv2.resize(color, (frame.shape[1], frame.shape[0]))

        return cv2.addWeighted(frame, 0.7, color, 0.3, 0)

    except Exception as e:
        traceback.print_exc()
        print("[ERROR] Overlay RA fallido:", e)
        return frame
# Escuchar pregunta por voz
def escuchar_pregunta():
    with microphone as source:
        print("Escuchando pregunta...")
        audio = recognizer.listen(source, timeout=5)
        try:
            return recognizer.recognize_google(audio, language="es-ES").lower()
        except:
            return None

# Bucle principal
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    zona_detectada = reconocer_zona(gris)
    esquinas, ids = detectar_marcadores(gris)

    if ids is not None:
        for id_array in ids:
            marcador = id_array[0]
            for nombre, datos in zonas_imagenes.items():
                if marcador in datos['ids']:
                    zona_detectada = nombre
                    break

    if zona_detectada:
        contador_zona[zona_detectada] = contador_zona.get(zona_detectada, 0) + 1
        if contador_zona[zona_detectada] >= UMBRAL_ESTABILIDAD:
            zona_estable = zona_detectada
            contador_zona = {zona_estable: contador_zona[zona_detectada]}
    else:
        contador_zona = {}
        zona_estable = "Zona desconocida"

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(esquinas, 0.05, cam_matrix, dist_coeffs)
        for i, id_array in enumerate(ids):
            marcador = id_array[0]
            render_mesh = modelos_precargados.get(marcador)

            if render_mesh:
                cv2.polylines(frame, [esquinas[i].astype(int)], True, (0, 255, 0), 2)
                cv2.putText(frame, f"Marcador {marcador}", tuple(esquinas[i][0][0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                frame = overlay_modelo(frame, rvecs[i], tvecs[i], render_mesh)

                if marcador != ultimo_marcador_mostrado:
                    ultimo_marcador_mostrado = marcador
                    pregunta = escuchar_pregunta()
                    if pregunta and ("información" in pregunta or "más" in pregunta):
                        popup("Respuesta", np.zeros((200, 400, 3), dtype=np.uint8))

    cv2.putText(frame, f"Te encuentras en: {zona_estable}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Museo AR', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()