import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import cv2
import numpy as np
import speech_recognition as sr
import pyrender
import trimesh
from cuia import popup, alphaBlending
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

def cargar_modelo_glb(ruta):
    try:
        scene_or_mesh = trimesh.load(ruta)

        # Unificar geometría si es escena
        if isinstance(scene_or_mesh, trimesh.Scene):
            print(f"[DEBUG] Modelo {ruta} es escena. Unificando geometría...")
            geometries = []
            for name, g in scene_or_mesh.geometry.items():
                try:
                    transform, _ = scene_or_mesh.graph.get(name)
                    g_copy = g.copy()
                    g_copy.apply_transform(transform)
                    geometries.append(g_copy)
                except Exception as e:
                    print(f"[WARN] No se pudo aplicar transform a {name}: {e}")
            if not geometries:
                raise ValueError("La escena no contiene geometría válida.")
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = scene_or_mesh

        # Centrar el modelo en su origen
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2
        mesh.apply_translation(-center)

        # Escalado automático para RA (~5 cm como máximo)
        target_size = 0.05
        scale_factor = target_size / np.max(mesh.extents)
        mesh.apply_scale(scale_factor)

        # Orientación: girar para que mire hacia la cámara en RA
        Ry = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
        Rx = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(Rx @ Ry)

        # Convertir a mesh de pyrender
        render_mesh = pyrender.Mesh.from_trimesh(mesh)
        print(f"[INFO] Modelo precargado correctamente: {ruta}")
        return render_mesh

    except Exception as e:
        print(f"[ERROR] Fallo al cargar modelo {ruta}: {e}")
        return None
    

modelos_precargados = {}
for marcador_id, ruta in marcadores_modelos.items():
    modelos_precargados[marcador_id] = cargar_modelo_glb(ruta)

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
##
def overlay_modelo(frame, rvec, tvec, render_mesh):
    try:
        # Crear matriz de pose desde rvec y tvec
        rvec = np.asarray(rvec, dtype=np.float32).reshape(3)
        tvec = np.asarray(tvec, dtype=np.float32).reshape(3)
        rot_matrix, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_matrix
        pose[:3, 3] = tvec

        # Crear escena
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3, 1.0])
        scene.add(render_mesh, pose=pose)

        # Luz
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(light, pose=np.eye(4))

        # Cámara virtual
        cam = pyrender.IntrinsicsCamera(
            fx=cam_matrix[0, 0], fy=cam_matrix[1, 1],
            cx=cam_matrix[0, 2], cy=cam_matrix[1, 2]
        )
        cam_pose = np.eye(4)
        cam_pose[2, 3] = 0.2  # moderado
        scene.add(cam, pose=cam_pose)

        # Render con canal alfa
        render_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        # DEBUG: Mostrar lo renderizado
        cv2.imshow("DEBUG Render Only", cv2.cvtColor(render_rgba, cv2.COLOR_RGBA2BGR))
        cv2.waitKey(1)

        # Si la imagen no tiene canal alfa, añadirlo manualmente
        if render_rgba.shape[2] == 3:
            alpha_channel = np.ones(render_rgba.shape[:2], dtype=np.uint8) * 255
            render_rgba = np.dstack((render_rgba, alpha_channel))

        # Hacer alpha blending con el frame real
        blended = alphaBlending(render_rgba, frame)
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
        return blended_bgr

    except Exception as e:
        import traceback
        traceback.print_exc()
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