# reconocimiento.py  – v7 (usa lógica del cuaderno del profesor)
"""Reconoce zona (tanques / armas / aviones) cuando se muestra la foto
correspondiente frente a la cámara.

➡ Implementa la misma lógica que aparece en los cuadernos `07-OpenCV-*
   Reconocimiento‑avanzado.ipynb` del profesor:
   1. `matches = matcher.knnMatch(desc_f, desc_db, k=2)`  (frame→db)
   2. ratio‑test de Lowe (0.75)
   3. Sin homografía – basta con nº «good» ≥ UMBRAL.

Con esto el código es idéntico al usado en clase y las imágenes impresas
(tanques.jpg, aviones.jpg, armas.jpg) se reconocen con la misma
fiabilidad que en los cuadernos.

Ajustes por defecto:
    • SIFT (nfeatures=800).  Fallback ORB (nfeatures=1200)
    • UMBRAL_SIFT = 12   (≥ 12 good‑matches → aceptación)
    • UMBRAL_ORB  = 25
    • SKIP_FRAMES  = 5   (reducir carga CPU)
    • DEBUG = False      (poner True para ver conteos)
"""
from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np

# ---------- ajustes ----------
SKIP_FRAMES   = 5
DEBUG         = False
UMBRAL_SIFT   = 12
UMBRAL_ORB    = 25

# ---------- 1. Detector + Matcher ----------
try:
    fe = cv2.SIFT_create(nfeatures=800)
    FLANN_INDEX_KDTREE = 1
    matcher = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
                                    dict(checks=40))
    sift_mode = True
    print("[INFO] SIFT activado (800 features)")
except Exception:
    fe = cv2.ORB_create(nfeatures=1200)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    sift_mode = False
    print("[WARN] SIFT no disponible → usando ORB (1200 features)")

# ---------- 2. Base de datos ----------
BASE_DIR = Path(__file__).resolve().parent
ZONAS_IMAGENES = {
    "Zona de tanques": str(BASE_DIR / "images" / "tanques.jpg"),
    "Zona de aviones": str(BASE_DIR / "images" / "aviones.jpg"),
    "Zona de armas"  : str(BASE_DIR / "images" / "armas.jpg"),
}

db: dict[str, dict] = {}
for nombre, ruta in ZONAS_IMAGENES.items():
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] No se pudo abrir {ruta}")
        continue
    kp, des = fe.detectAndCompute(img, None)
    db[nombre] = {"kp": kp, "des": des}

_frame_count = 0
KNN_K = 2

# ---------- 3. Función pública ----------

def reconocer_zona(frame_gray) -> str | None:
    global _frame_count
    _frame_count = (_frame_count + 1) % SKIP_FRAMES
    if _frame_count:
        return None  # saltamos para aligerar FPS

    kp_f, des_f = fe.detectAndCompute(frame_gray, None)
    if des_f is None or len(des_f) < KNN_K:
        return None

    mejor, n_good_mejor = None, 0

    for nombre, datos in db.items():
        des_db, kp_db = datos["des"], datos["kp"]
        if des_db is None or len(des_db) < KNN_K:
            continue

        try:
            matches = matcher.knnMatch(des_f, des_db, k=KNN_K)  # frame → db (igual que cuaderno)
        except cv2.error:
            continue

        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        n_good = len(good)
        if DEBUG:
            print(f"{nombre}: {n_good} good‑matches")

        if sift_mode and n_good >= UMBRAL_SIFT and n_good > n_good_mejor:
            mejor, n_good_mejor = nombre, n_good
        elif (not sift_mode) and n_good >= UMBRAL_ORB and n_good > n_good_mejor:
            mejor, n_good_mejor = nombre, n_good

    return mejor

# ---------- 4. ArUco ----------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, _detector_params)

def detectar_marcadores(frame_gray):
    corners, ids, _ = detector.detectMarkers(frame_gray)
    return corners, ids
