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


# Variables globales para persistencia de detección
marcador_visible = None
contador_visibilidad = 0
UMBRAL_VISIBILIDAD = 5


# ➋ Crea UN solo renderer globalizado 
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
#recognizer = sr.Recognizer()
#microphone = sr.Microphone()

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
# Overlay mejorado con estabilidad y perspectiva
def overlay_modelo_estable(frame, esquinas, rvec, tvec, render_mesh):
    global marcador_visible, contador_visibilidad

    try:
        # Pose del modelo 3D desde marcador
        rot_matrix, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_matrix
        pose[:3, 3] = tvec

        # Crear escena transparente
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.4, 0.4, 0.4, 1.0])
        scene.add(render_mesh, pose=pose)

        # Luz fuerte desde la cámara
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        scene.add(light, pose=np.eye(4))

        # Cámara calibrada
        cam = pyrender.IntrinsicsCamera(
            fx=cam_matrix[0, 0], fy=cam_matrix[1, 1],
            cx=cam_matrix[0, 2], cy=cam_matrix[1, 2]
        )
        cam_pose = np.eye(4)
        cam_pose[2, 3] = 0.2  # Aleja la cámara virtual para ver el objeto
        scene.add(cam, pose=cam_pose)

        # Render RGBA
        render_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

       # Fuerza transparencia en píxeles negros
        alpha_channel = np.where(
            np.all(render_rgba[:, :, :3] == [0, 0, 0], axis=-1),
            0, 255
        ).astype(np.uint8)
        render_rgba = np.dstack((render_rgba[:, :, :3], alpha_channel))

        # Combinar con la imagen real de cámara
        blended = alphaBlending(render_rgba, frame)
        return cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)

    except Exception as e:
        traceback.print_exc()
        return frame

# Escuchar pregunta por voz
'''
def escuchar_pregunta():
    with microphone as source:
        print("Escuchando pregunta...")
        audio = recognizer.listen(source, timeout=5)
        try:
            return recognizer.recognize_google(audio, language="es-ES").lower()
        except:
            return None
''' 
# Bucle principal
cap = cv2.VideoCapture(0)

# Bucle principal mejorado
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    zona_detectada = reconocer_zona(gris)
    esquinas, ids = detectar_marcadores(gris)
    if  ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(esquinas, 0.05, cam_matrix, dist_coeffs)
        for i, id_array in enumerate(ids):
            marcador_actual = id_array[0]

            # Actualiza visibilidad estable
            if marcador_visible == marcador_actual:
                contador_visibilidad += 1
            else:
                marcador_visible = marcador_actual
                contador_visibilidad = 1

            # Estabilidad de zona
            for nombre_zona, datos in zonas_imagenes.items():
                if marcador_actual in datos["ids"]:
                    zona_detectada = nombre_zona
                    break

            if zona_detectada:
                contador_zona[zona_detectada] = contador_zona.get(zona_detectada, 0) + 1
                if contador_zona[zona_detectada] >= UMBRAL_ESTABILIDAD:
                    zona_estable = zona_detectada
                    contador_zona = {zona_estable: contador_zona[zona_detectada]}
            else:
                contador_zona = {}

            if contador_visibilidad >= UMBRAL_VISIBILIDAD:
                render_mesh = modelos_precargados.get(marcador_actual)
                if render_mesh:
                    frame = overlay_modelo_estable(frame, esquinas[i], rvecs[i], tvecs[i], render_mesh)

                    # Dibujar contorno y texto
                    cv2.polylines(frame, [esquinas[i].astype(int)], True, (0, 255, 0), 2)
                    cv2.putText(frame, f"Marcador {marcador_actual}", tuple(esquinas[i][0][0].astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    if marcador_actual != ultimo_marcador_mostrado:
                        ultimo_marcador_mostrado = marcador_actual
                        #pregunta = escuchar_pregunta()
                        #if pregunta and ("información" in pregunta or "más" in pregunta):
                         #   popup("Respuesta", np.zeros((200, 400, 3), dtype=np.uint8))
    else:
        contador_visibilidad = max(0, contador_visibilidad - 1)
        if contador_visibilidad == 0:
            marcador_visible = None

    # Texto zona estable
    cv2.putText(frame, f"Te encuentras en: {zona_estable}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Museo AR', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
