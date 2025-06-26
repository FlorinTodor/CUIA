import os
import trimesh
import pyrender
import numpy as np
import cv2

os.environ["PYOPENGL_PLATFORM"] = "egl"

# Cargar escena .glb
scene_or_mesh = trimesh.load("CUIA_models/pzkpfw_vi_tiger_1.glb")

# Si es escena, combinar toda la geometría con sus transformaciones
if isinstance(scene_or_mesh, trimesh.Scene):
    print("[DEBUG] Modelo es escena. Unificando geometría...")
    geometries = []
    for name, g in scene_or_mesh.geometry.items():
        transform, _ = scene_or_mesh.graph.get(name)
        g_copy = g.copy()
        g_copy.apply_transform(transform)
        geometries.append(g_copy)
    mesh = trimesh.util.concatenate(geometries)
else:
    mesh = scene_or_mesh

# Centrar y escalar
center = (mesh.bounds[0] + mesh.bounds[1]) / 2
mesh.apply_translation(-center)

scale = 1.0 / np.max(mesh.extents)
mesh.apply_scale(scale * 0.9)

print("[DEBUG] Bounds:", mesh.bounds)
print("[DEBUG] Extents:", mesh.extents)

# Corregir orientación
R = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
mesh.apply_transform(R)

# Convertir a pyrender
render_mesh = pyrender.Mesh.from_trimesh(mesh)

# Crear escena
scene = pyrender.Scene()
scene.add(render_mesh, pose=np.eye(4))
light = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)
scene.add(light, pose=np.eye(4))

# Cámara
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
cam_pose = np.eye(4)
cam_pose[2, 3] = 2.5
scene.add(camera, pose=cam_pose)

# Render
renderer = pyrender.OffscreenRenderer(640, 480)
color, _ = renderer.render(scene)
color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)
cv2.imshow("Modelo .glb Renderizado", color)
cv2.waitKey(0)
cv2.destroyAllWindows()
