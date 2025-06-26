import os
os.environ["PYOPENGL_PLATFORM"] = "egl"  # o "osmesa"

import pyrender
import trimesh
import numpy as np

# Crear escena
mesh = trimesh.primitives.Sphere(radius=1.0)
render_mesh = pyrender.Mesh.from_trimesh(mesh)
scene = pyrender.Scene()
scene.add(render_mesh)

# Añadir cámara
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
cam_pose = np.array([
    [1.0, 0.0,  0.0, 0.0],
    [0.0, 1.0,  0.0, 0.0],
    [0.0, 0.0,  1.0, 3.0],  # Aleja la cámara para que vea el objeto
    [0.0, 0.0,  0.0, 1.0]
])
scene.add(camera, pose=cam_pose)

# Renderizado
renderer = pyrender.OffscreenRenderer(640, 480)
color, depth = renderer.render(scene)
renderer.delete()

print(f"Imagen renderizada con forma: {color.shape}")
