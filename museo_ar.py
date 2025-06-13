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

# ---------------- Configuración ----------------
UMBRAL_VISIBILIDAD  = 5   # frames necesarios para mostrar modelo
UMBRAL_ESTABILIDAD  = 10  # frames consecutivos para fijar zona

# Mapeo de ArUco → Zona
ZONA_POR_ID = {
    0: "Zona de tanques", 1: "Zona de tanques",
    2: "Zona de armas",   3: "Zona de armas",
    4: "Zona de aviones", 5: "Zona de aviones",
}

# ---------------- Estado global ----------------
marcador_visible      = None   # id aruco actual
contador_visibilidad  = 0
zona_estable          = "Zona desconocida"
contador_zona: dict[str, int] = {}
ultimo_marcador_mostrado = None

# Renderer off‑screen único
renderer = pyrender.OffscreenRenderer(640, 480)

# ------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------

def overlay_modelo_estable(frame, esquinas, rvec, tvec, render_mesh):
    """Renderiza el modelo 3D alineado y lo funde en la imagen."""
    try:
        rot_matrix, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_matrix
        pose[:3, 3]  = tvec

        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[.4, .4, .4, 1])
        scene.add(render_mesh, pose=pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=5.), pose=np.eye(4))

        cam = pyrender.IntrinsicsCamera(fx=cam_matrix[0,0], fy=cam_matrix[1,1],
                                         cx=cam_matrix[0,2], cy=cam_matrix[1,2])
        cam_pose = np.eye(4); cam_pose[2,3] = 0.2  # alejar cámara virtual
        scene.add(cam, pose=cam_pose)

        rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        alpha = np.where(np.all(rgba[:,:,:3]==0, axis=-1), 0, 255).astype(np.uint8)
        rgba  = np.dstack((rgba[:,:,:3], alpha))
        blended = alphaBlending(rgba, frame)
        return cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)

    except Exception:
        traceback.print_exc()
        return frame

# ------------------------------------------------
# Bucle principal captura
# ------------------------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------- Reconocimiento por imagen (si no hay ArUco) --------
    zona_detectada = reconocer_zona(gris)

    # -------- Detección de ArUco --------
    esquinas, ids = detectar_marcadores(gris)
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            esquinas, 0.05, cam_matrix, distCoeffs=dist_coeffs)

        for i, id_array in enumerate(ids):
            marcador_actual = int(id_array[0])

            # Visibilidad estable
            if marcador_visible == marcador_actual:
                contador_visibilidad += 1
            else:
                marcador_visible, contador_visibilidad = marcador_actual, 1

            # Zona por Id ArUco (sobre‑escribe a la de la imagen)
            zona_detectada = ZONA_POR_ID.get(marcador_actual, zona_detectada)

            # Mostrar modelo si visible varios frames
            if contador_visibilidad >= UMBRAL_VISIBILIDAD:
                mesh = modelos_precargados.get(marcador_actual)
                if mesh is not None:
                    frame = overlay_modelo_estable(frame, esquinas[i], rvecs[i], tvecs[i], mesh)
                    cv2.polylines(frame, [esquinas[i].astype(int)], True, (0,255,0), 2)
                    cv2.putText(frame, f"Marcador {marcador_actual}", tuple(esquinas[i][0][0].astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)
    else:
        # No hay marcadores: reset visibilidad
        contador_visibilidad = max(0, contador_visibilidad - 1)
        if contador_visibilidad == 0:
            marcador_visible = None

    # --------- Actualizar zona estable (si hay una propuesta) ---------
    if zona_detectada:
        contador_zona[zona_detectada] = contador_zona.get(zona_detectada, 0) + 1
        if contador_zona[zona_detectada] >= UMBRAL_ESTABILIDAD:
            zona_estable = zona_detectada
            contador_zona = {zona_estable: contador_zona[zona_detectada]}
    else:
        # Si no hay detección, reduce contadores lentamente
        for z in list(contador_zona):
            contador_zona[z] = max(0, contador_zona[z] - 1)

    # --------------- HUD ---------------
    cv2.putText(frame, f"Te encuentras en: {zona_estable}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Museo AR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
