#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
museo_ar_pygfx.py – Versión Pygfx del bucle principal para el museo AR.
Reemplaza pyrender por cuia.escenaPYGFX y cuia.modeloGLTF.
"""
import cv2
import numpy as np
import cuia

from cuia import alphaBlending
from reconocimiento import detectar_marcadores, reconocer_zona  # módulo del alumno
from camera import cameraMatrix as K, distCoeffs as D           # intrínsecos
import models_pygfx as models

# Configuración ArUco
MARKER_SIZE = 0.05  # metros

def cv_to_gl_pose(rvec, tvec) -> np.ndarray:
    """Convierte rvec/tvec (OpenCV) → matriz 4×4 en convención OpenGL."""
    R, _ = cv2.Rodrigues(rvec)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = tvec.reshape(3)
    M[[1, 2]] *= -1      # Y,Z → −Y,−Z
    return M


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    alto  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    escena = cuia.escenaPYGFX(fov=60, ancho=ancho, alto=alto)
    escena.iluminar(1.0)
    escena.mostrar_ejes()

    modelos_activos = {}  # id → modeloGLTF

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1 · Reconocimiento por imagen (opcional, no interfiere)
        _ = reconocer_zona(gray)

        # 2 · Detección de marcadores
        corners, ids = detectar_marcadores(gray)
        visibles = set()

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE, K, D)
            for i, id_arr in enumerate(ids):
                mid = int(id_arr[0])
                visibles.add(mid)

                pose = cv_to_gl_pose(rvecs[i], tvecs[i])

                if mid not in modelos_activos:
                    modelo = models.cargar(mid)
                    escena.agregar_modelo(modelo)
                    modelos_activos[mid] = modelo

                modelos_activos[mid].model_obj.local.matrix = pose
                escena.ilumina_modelo(modelos_activos[mid])

                # ──────── DIBUJAR EJE XYZ SOBRE LA MARCA ────────
                if hasattr(cv2.aruco, "drawAxis"):
                    # Compilaciones opencv-contrib que sí traen el binding
                    cv2.aruco.drawAxis(frame, K, D, rvecs[i], tvecs[i], 0.03)
                else:
                    # Fallback universal (OpenCV ≥ 4.5)
                    cv2.drawFrameAxes(frame, K, D, rvecs[i], tvecs[i], 0.03)
                # ────────────────────────────────────────────────

        # 3 · Retirar modelos ya no visibles
        for mid in list(modelos_activos.keys()):
            if mid not in visibles:
                modelo = modelos_activos.pop(mid)
                escena.scene.remove(modelo.model_obj)

        # 4 · Render y composición
        rgba = escena.render()                         # RGBA uint8
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cam_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        composed = alphaBlending(bgra, cam_bgra)

        cv2.imshow("Museo AR (Pygfx)", composed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
