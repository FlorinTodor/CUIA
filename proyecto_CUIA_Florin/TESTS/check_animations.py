'''
check_animations.py – Script rápido para saber si tus modelos .glb/.gltf
contienen animaciones.

▶ Uso:
    python3 check_animations.py            # recorre CUIA_models/*
    python3 check_animations.py ruta1.glb ruta2.glb ...

Requisitos:
    pip install pygltflib

La librería «pygltflib» lee el glTF de forma íntegra y expone la lista
`gltf.animations`. Si el modelo es una escena binaria (.glb), funciona igual.
'''

import sys
from pathlib import Path
from typing import List

try:
    from pygltflib import GLTF2
except ImportError:
    print("[ERROR] Debes instalar pygltflib primero →  pip install pygltflib")
    sys.exit(1)

# ---------------- Utilidades ----------------

def tiene_animaciones(ruta: Path) -> bool:
    try:
        gltf = GLTF2().load_binary(ruta)
        return bool(gltf.animations)  # lista vacía ⇒ sin animaciones
    except Exception as e:
        print(f"[WARN] {ruta.name}: no se pudo analizar → {e}")
        return False


def analizar_rutas(rutas: List[Path]):
    print("⏳ Analizando modelos…\n")
    for ruta in rutas:
        if not ruta.exists():
            print(f"[SKIP] {ruta} no existe")
            continue
        tiene = tiene_animaciones(ruta)
        estado = "✔ Tiene animaciones" if tiene else "✘ Sin animaciones"
        print(f"{ruta.name:40}  {estado}")


# ---------------- Main ----------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        rutas = [Path(p) for p in sys.argv[1:]]
    else:
        rutas = list(Path("CUIA_models").glob("*.glb"))
        if not rutas:
            print("[ERROR] No se encontraron .glb en CUIA_models/ y no se pasaron rutas")
            sys.exit(1)

    analizar_rutas(rutas)
