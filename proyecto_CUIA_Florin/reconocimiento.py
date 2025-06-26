# reconocimiento.py  – v9 (verificación geométrica + umbral adaptativo)
"""Reconoce la *zona* (tanques / armas / aviones) comparando el fotograma
con las imágenes de referencia.


───────────────────

1. **Umbral adaptativo** – `MIN_GOOD_SIFT/ORB` siguen siendo mínimos, pero si el
   número de *good matches* es muy alto se pide también un ratio mínimo de
   *inliers/ good matches* para asegurar coherencia.
2. Mensajes de *debug* más claros (`DEBUG=True`).


"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
import os

# ═════════════════════  Parámetros globales  ════════════════════════
DEBUG            = False     # ← pon True para ver detalles en consola
FORCE_ORB        = False     # fuerza ORB aunque SIFT esté disponible

# 1) Coincidencias "buenas" mínimas para considerar la zona
MIN_GOOD_SIFT    = 12        # antes 8 → más estricto
MIN_GOOD_ORB     = 25        # antes 15

# 2) Inliers (RANSAC) – validación geométrica -----------------------
MIN_INLIERS_SIFT = 8         # nº mínimo de inliers para SIFT
MIN_INLIERS_ORB  = 15        # idem ORB
RATIO_INLIERS    = 0.5       # inliers deben ser al menos 50 % de good matches

# 3) Lowe ratio ------------------------------------------------------
LOWE_RATIO       = 0.75

# ════════════════════  Inicializar detector + matcher  ══════════════
if not FORCE_ORB:
    try:
        fe = cv2.SIFT_create(nfeatures=1000)
        FLANN_INDEX_KDTREE = 1
        index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=60)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        sift_mode = True
        if DEBUG:
            print("[INFO] SIFT activado para reconocimiento de imágenes")
    except Exception as e:
        if DEBUG:
            print("[WARN] SIFT no disponible (", e, ") → usando ORB")
        sift_mode = False
else:
    sift_mode = False

if not sift_mode:
    fe = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    if DEBUG and FORCE_ORB:
        print("[INFO] ORB forzado por configuración")

# ════════════════════  Cargar imágenes de referencia  ═══════════════
zona_dir = Path(__file__).parent / "images"
zonas_imagenes = {
    "Zona de tanques":  zona_dir / "tanques.jpg",
    "Zona de aviones":  zona_dir / "aviones.jpg",
    "Zona de armas":    zona_dir / "armas.jpg",
}

db = {}
for nombre, ruta in zonas_imagenes.items():
    img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    if img is None:
        if DEBUG:
            print(f"[ERROR] No se pudo abrir {ruta}")
        continue
    kp, des = fe.detectAndCompute(img, None)
    db[nombre] = {"kp": kp, "des": des}
    if DEBUG:
        print(f"[INIT] {nombre}: {len(kp)} keypoints")

# ══════════════════════  Funciones auxiliares  ═════════════════════

def _good_matches(des_f, des_db):
    """Devuelve lista de *good matches* tras el Lowe‑ratio test."""
    try:
        matches = matcher.knnMatch(des_f, des_db, k=2)
    except cv2.error:
        return []
    good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]
    return good


def _validate_homography(kp_f, kp_db, good):
    """Homografía + RANSAC. Devuelve nº de inliers."""
    if len(good) < 4:
        return 0
    src_pts = np.float32([kp_f[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_db[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return 0
    return int(mask.sum())

# ═══════════════════════  API pública  ═════════════════════════════

def reconocer_zona(frame_gray: np.ndarray) -> str | None:
    """Devuelve nombre de la zona o **None** si no supera los umbrales."""
    kp_f, des_f = fe.detectAndCompute(frame_gray, None)
    if des_f is None:
        if DEBUG:
            print("[DEBUG] frame sin descriptores")
        return None

    mejor_zona: str | None = None
    mejor_score: tuple[int, int] = (0, 0)  # (good, inliers)

    for zona, datos in db.items():
        des_db = datos["des"]
        kp_db  = datos["kp"]
        if des_db is None:
            continue

        good = _good_matches(des_f, des_db)
        ngood = len(good)

        if ngood == 0:
            if DEBUG:
                print(f"[DEBUG] {zona}: 0 good matches")
            continue

        inliers = _validate_homography(kp_f, kp_db, good)

        if DEBUG:
            print(f"[DEBUG] {zona}: good={ngood:3d}, inliers={inliers:3d}")

        # escoger la zona con más *inliers* (primario) y luego más *good*
        if inliers > mejor_score[1] or (inliers == mejor_score[1] and ngood > mejor_score[0]):
            mejor_score = (ngood, inliers)
            mejor_zona = zona

    if mejor_zona is None:
        return None

    ngood, inliers = mejor_score
    if sift_mode:
        if ngood < MIN_GOOD_SIFT or inliers < MIN_INLIERS_SIFT:
            return None
    else:
        if ngood < MIN_GOOD_ORB or inliers < MIN_INLIERS_ORB:
            return None

    # razón inliers / good
    if inliers / ngood < RATIO_INLIERS:
        return None

    if DEBUG:
        print(f"[INFO] Zona detectada → {mejor_zona}  (good={ngood}, inliers={inliers})")
    return mejor_zona

# ════════════════  Detección ArUco   ═════════════════=
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector   = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

def detectar_marcadores(frame_gray: np.ndarray):
    """Wrapper para que *museo_ar.py* reciba (esquinas, ids)."""
    corners, ids, _ = detector.detectMarkers(frame_gray)
    return corners, ids
