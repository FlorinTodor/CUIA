"""
models_pygfx.py
===============
Cargador y normalizador de modelos **GLB** para emplearlos con *Pygfx*
en el museo AR, respetando exactamente la API docente contenida en
`cuia.py` (clase `modeloGLTF`).

Flujo de normalización cuando se llama a `cargar(mid)`:
1. **Centrar** → traslada el centro del *bounding‑box* al origen.
2. **Rotar 180 ° eje‑X** → convención "Z‑up"; usa `modelo.rotar((π,0,0))`.
3. **Escalar uniforme** → que el mayor eje ≤ `SCALE_MAX` (5 cm).
4. **Flotar** → desplaza en +Z para que la base quede en Z = 0.

El módulo mantiene una caché en RAM para evitar recargas de disco.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import cuia

# ──────────────────────────────────────────────────────────────────────
# Constantes y datos
# ──────────────────────────────────────────────────────────────────────
SCALE_MAX = 0.05  # metros (≈ 5 cm)

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "museo_data.json", encoding="utf-8") as f:
    _DATA = json.load(f)

# Diccionario {id_str: ruta_relativa}
_MODEL_PATHS: Dict[str, str] = {
    str(k): v for k, v in _DATA.get("model_paths", {}).items()
}

# Caché en RAM de modelos ya cargados y normalizados
_CACHE: Dict[str, cuia.modeloGLTF] = {}

# ──────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ──────────────────────────────────────────────────────────────────────

def _centrar(modelo: cuia.modeloGLTF) -> None:
    """Traslada el centro del *bounding‑box* al origen."""
    bbox_min, bbox_max = modelo.model_obj.get_world_bounding_box()
    centro = (bbox_min + bbox_max) * 0.5  # numpy array (3,)
    modelo.trasladar(tuple(-centro))


def _rotar_eje_x(modelo: cuia.modeloGLTF) -> None:
    """Gira 180 ° sobre el eje X → convención Z‑up."""
    modelo.rotar((np.pi, 0.0, 0.0))


def _escalar(modelo: cuia.modeloGLTF) -> None:
    """Escala uniformemente para que el mayor eje ≤ SCALE_MAX."""
    bbox_min, bbox_max = modelo.model_obj.get_world_bounding_box()
    size = (bbox_max - bbox_min).max()  # lado mayor (float)
    if size > 1e-8:
        s = SCALE_MAX / size
        modelo.escalar(s)


def _flotar(modelo: cuia.modeloGLTF) -> None:
    """Desplaza +Z de forma que la base del modelo quede en Z = 0."""
    bbox_min, _ = modelo.model_obj.get_world_bounding_box()
    min_z = float(bbox_min[2])
    if abs(min_z) > 1e-6:
        modelo.trasladar((0.0, 0.0, -min_z))

# ──────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────

def cargar(mid: Union[int, str]) -> cuia.modeloGLTF:
    """Devuelve un objeto **modeloGLTF** normalizado y cacheado.

    Parameters
    ----------
    mid : int | str
        Identificador de modelo (marcador ArUco, etc.).

    Returns
    -------
    cuia.modeloGLTF
        Modelo listo para añadirse a `escenaPYGFX.scene.add()`.
    """
    mid_str = str(mid)

    # 1. Comprobar caché
    if mid_str in _CACHE:
        return _CACHE[mid_str]

    # 2. Obtener ruta de GLB según JSON
    rel_path = _MODEL_PATHS.get(mid_str)
    if rel_path is None:
        raise KeyError(f"ID de modelo desconocido en museo_data.json: {mid_str}")

    ruta_glb = (BASE_DIR / rel_path).resolve()

    # 3. Cargar con helper del profesor
    modelo = cuia.modeloGLTF(ruta_glb)

    # 4. Normalización geométrica
    _centrar(modelo)
    _rotar_eje_x(modelo)
    _escalar(modelo)
    _flotar(modelo)

    # 5. Guardar en caché y devolver
    _CACHE[mid_str] = modelo
    return modelo
