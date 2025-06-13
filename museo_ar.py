import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import cv2
import numpy as np
import pyrender
from cuia import popup, alphaBlending
from reconocimiento import reconocer_zona, detectar_marcadores
from models import modelos_precargados

from camera import cameraMatrix as cam_matrix, distCoeffs as dist_coeffs
import traceback

# Variables globales para persistencia de detección
marcador_visible = None
contador_visibilidad = 0
UMBRAL_VISIBILIDAD = 5

# Crea UN solo renderer globalizado
renderer = pyrender.OffscreenRenderer(640, 480)

# Mapeo de zona por Id de marcador
ZONA_POR_ID = {
    0: "Zona de tanques", 1: "Zona de tanques",
    2: "Zona de armas",   3: "Zona de armas",
    4: "Zona de aviones", 5: "Zona de aviones",
}

# Variables de estabilidad
zona_estable = "Zona desconocida"
contador_zona = {}
UMBRAL_ESTABILIDAD = 10
ultimo_marcador_mostrado = None


def overlay_modelo_estable(frame, esquinas, rvec, tvec, render_mesh):
    """Renderiza el modelo 3D alineado con el marcador y lo funde en la imagen."""
    global marcador_visible, contador_visibilidad
    try:
        rot_matrix, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_matrix
        pose[:3, 3] = tvec

        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.4, 0.4, 0.4, 1.0])
        scene.add(render_mesh, pose=pose)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        scene.add(light, pose=np.eye(4))

        cam = pyrender.IntrinsicsCamera(
            fx=cam_matrix[0, 0], fy=cam_matrix[1, 1],
            cx=cam_matrix[0, 2], cy=cam_matrix[1, 2]
        )
        cam_pose = np.eye(4)
        cam_pose[2, 3] = 0.2
        scene.add(cam, pose=cam_pose)

        render_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        alpha_channel = np.where(
            np.all(render_rgba[:, :, :3] == [0, 0, 0], axis=-1),
            0, 255
        ).astype(np.uint8)
        render_rgba = np.dstack((render_rgba[:, :, :3], alpha_channel))

        blended = alphaBlending(render_rgba, frame)
        return cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)

    except Exception:
        traceback.print_exc()
        return frame


# Bucle principal
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Zona por imagen si no hay marcador
    zona_detectada = reconocer_zona(gris)

    # Detectar ArUco
    esquinas, ids = detectar_marcadores(gris)
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            esquinas, 0.05, cam_matrix, dist_coeffs
        )

        for i, id_array in enumerate(ids):
            marcador_actual = id_array[0]

            # Visibilidad estable
            if marcador_visible == marcador_actual:
                contador_visibilidad += 1
            else:
                marcador_visible = marcador_actual
                contador_visibilidad = 1

            # Zona definida por ArUco
            zona_detectada = ZONA_POR_ID.get(marcador_actual, zona_detectada)

            # Contador de estabilidad de zona
            if zona_detectada:
                contador_zona[zona_detectada] = contador_zona.get(zona_detectada, 0) + 1
                if contador_zona[zona_detectada] >= UMBRAL_ESTABILIDAD:
                    zona_estable = zona_detectada
                    contador_zona = {zona_estable: contador_zona[zona_detectada]}

            if contador_visibilidad >= UMBRAL_VISIBILIDAD:
                render_mesh = modelos_precargados.get(marcador_actual)
                if render_mesh is not None:
                    frame = overlay_modelo_estable(
                        frame, esquinas[i], rvecs[i], tvecs[i], render_mesh
                    )
                    cv2.polylines(frame, [esquinas[i].astype(int)], True, (0, 255, 0), 2)
                    cv2.putText(frame, f"Marcador {marcador_actual}",
                                tuple(esquinas[i][0][0].astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                                2, lineType=cv2.LINE_AA)
    else:
        contador_visibilidad = max(0, contador_visibilidad - 1)
        if contador_visibilidad == 0:
            marcador_visible = None

    # Texto de zona estable
    cv2.putText(frame, f"Te encuentras en: {zona_estable}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    cv2.imshow("Museo AR", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
