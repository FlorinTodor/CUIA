import sys
import pyrender
import trimesh

ruta = sys.argv[1]

try:
    print(f"[INFO] Cargando modelo: {ruta}")
    mesh = trimesh.load(ruta)
    if not isinstance(mesh, trimesh.Trimesh):
     mesh = mesh.geometry[list(mesh.geometry.keys())[0]]


    render_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(render_mesh)

    print("[INFO] Mostrando modelo en visor 3D...")
    pyrender.Viewer(scene, use_raymond_lighting=True)

except Exception as e:
    print(f"[ERROR] No se pudo abrir el modelo: {e}")