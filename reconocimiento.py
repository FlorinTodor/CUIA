# reconocimiento.py  – v8 (debug detallado + umbral adaptable)
"""Reconoce zona (tanques / armas / aviones) comparando la imagen captada con
las fotos de referencia.

▶ NOTAS IMPORTANTE PARA DEPURAR
1. Activa `DEBUG=True` y mira la terminal: se imprimen nº de keypoints, matches,
   good matches y si supera o no el umbral.
2. Si tu impresora / móvil produce pocas coincidencias, puedes bajar
   `MIN_GOOD_SIFT` y `MIN_GOOD_ORB` en tiempo real.
3. Se puede cambiar el detector a ORB manualmente poniendo
   `FORCE_ORB = True`.
"""
from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np

# ---------- Parámetros globales ----------
DEBUG           = False     # ← pon False cuando funcione
FORCE_ORB       = False    # fuerza ORB aunque SIFT esté disponible
MIN_GOOD_SIFT   = 8        # umbral mínimo para SIFT
MIN_GOOD_ORB    = 15       # umbral mínimo para ORB

# ---------- Inicializar detector y matcher ----------
if not FORCE_ORB:
    try:
        fe = cv2.SIFT_create(nfeatures=800)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
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
    fe = cv2.ORB_create(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    if DEBUG and FORCE_ORB:
        print("[INFO] ORB forzado por configuración")

# ---------- Cargar imágenes de referencia ----------
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

# ---------- API pública ----------

def reconocer_zona(frame_gray: np.ndarray) -> str | None:
    """Devuelve nombre de la zona o None si no supera el umbral."""
    kp_f, des_f = fe.detectAndCompute(frame_gray, None)
    if des_f is None:
        if DEBUG:
            print("[DEBUG] frame sin descriptores")
        return None

    mejor_zona: str | None = None
    mejor_good: int = 0

    for zona, datos in db.items():
        des_db = datos["des"]
        if des_db is None:
            continue

        # Dirección frame→db (como en cuaderno del profesor)
        try:
            matches = matcher.knnMatch(des_f, des_db, k=2)
        except cv2.error as e:
            if DEBUG:
                print("[DEBUG] knnMatch error:", e)
            continue

        # Lowe ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        ngood = len(good)
        if DEBUG:
            print(f"[DEBUG] {zona}: keyF={len(kp_f):3d}  matches={len(matches):3d}  good={ngood:3d}")

        if ngood > mejor_good:
            mejor_good = ngood
            mejor_zona = zona

    # ¿supera umbral?
    umbral = MIN_GOOD_SIFT if sift_mode else MIN_GOOD_ORB
    if mejor_good >= umbral:
        if DEBUG:
            print(f"[INFO] Zona detectada → {mejor_zona}  (good={mejor_good})")
        return mejor_zona
    else:
        return None


# ---------- Detección ArUco (sin cambios) ----------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector   = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

def detectar_marcadores(frame_gray: np.ndarray):
    """Wrapper para que museo_ar.py reciba (esquinas, ids)."""
    corners, ids, _ = detector.detectMarkers(frame_gray)
    return corners, ids
