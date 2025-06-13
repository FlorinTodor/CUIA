import os
import pickle
from typing import Dict

import numpy as np
import trimesh
import pyrender

"""
models.py  – versión robusta
---------------------------------
• Carga modelos .glb/.gltf (trimesh) y los deja listos como pyrender.Mesh
• Primera vez los procesa (centrado, escalado, rotado) y los guarda en
  cache_models/<nombre>.pkl. Las siguientes ejecuciones van a disco.
• Robusto ante escenas cuyas geometrías no están conectadas al grafo
  principal (ej.: "No path from world->Object_0"), usando Identidad.
"""

# ---------------------------------------------------------------------------
# CONFIGURACIÓN ----------------------------------------------------------------
MODELOS: Dict[int, str] = {
    0: "CUIA_models/pzkpfw_vi_tiger_1.glb",        # tanque Tiger I
    1: "CUIA_models/consolidated_b-24_liberator.glb",  # bombardero B‑24
    2: "CUIA_models/mp_40_submachine_gun.glb",     # MP‑40
    # Añade más pares {id: ruta} si fuera necesario
}

SCALE_MAX = 0.05  # metros (≈ 5 cm) → tamaño máximo del modelo
CACHE_DIR = "cache_models"
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# FUNCIONES AUXILIARES --------------------------------------------------------

def _procesar_trimesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centra en el origen, uniformiza escala y alinea ejes."""
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)
    # Escala uniforme al tamaño máximo indicado
    scale_factor = SCALE_MAX / np.max(mesh.extents)
    mesh.apply_scale(scale_factor)
    # Rotaciones para que "mire" al eje Z
    Rx = trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0))
    Ry = trimesh.transformations.rotation_matrix(np.pi, (0, 1, 0))
    mesh.apply_transform(Rx @ Ry)
    return mesh


def _cargar_trimesh(ruta: str) -> trimesh.Trimesh:
    """Carga un archivo .glb/.gltf y lo devuelve como único Trimesh."""
    escena_o_malla = trimesh.load(ruta, force="scene")  # fuerza Scene si existe

    if isinstance(escena_o_malla, trimesh.Scene):
        # Recorre cada geometría; si el transform falla, usa identidad
        mallas = []
        for nombre, geom in escena_o_malla.geometry.items():
            try:
                transform = escena_o_malla.graph.get(nombre)[0]
            except ValueError:
                # Geometría sin camino en el grafo → identidad
                transform = np.eye(4)
            copia = geom.copy()
            copia.apply_transform(transform)
            mallas.append(copia)
        if not mallas:
            raise ValueError(f"[ERROR] Escena vacía en {ruta}")
        mesh = trimesh.util.concatenate(mallas) if len(mallas) > 1 else mallas[0]
    else:  # ya es Trimesh
        mesh = escena_o_malla

    return _procesar_trimesh(mesh)


# ---------------------------------------------------------------------------
# API PÚBLICA -----------------------------------------------------------------

def cargar_modelo(marcador_id: int) -> pyrender.Mesh:
    """Devuelve pyrender.Mesh listo para pintar; con caché persistente."""
    ruta = MODELOS[marcador_id]
    cache_file = os.path.join(CACHE_DIR, os.path.basename(ruta) + ".pkl")

    # 1) Intentar leer caché ---------------------------------------------------
    if os.path.isfile(cache_file):
        try:
            with open(cache_file, "rb") as f:
                mesh = pickle.load(f)
                print(f"[CACHE] {ruta} cargado desde {cache_file}")
                return mesh
        except Exception as e:
            print(f"[WARN] Caché corrupta en {cache_file}: {e}. Reproceso…")

    # 2) Procesado completo ----------------------------------------------------
    print(f"[INFO] Procesando {ruta}…")
    mesh_trimesh = _cargar_trimesh(ruta)
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh)

    # Guardar caché (mejor protocolo)
    with open(cache_file, "wb") as f:
        pickle.dump(mesh_pyrender, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Caché guardada ← {cache_file}")

    return mesh_pyrender


# Precarga al importar ---------------------------------------------------------
modelos_precargados: Dict[int, pyrender.Mesh] = {
    idx: cargar_modelo(idx) for idx in MODELOS
}

__all__ = ["modelos_precargados", "cargar_modelo"]
