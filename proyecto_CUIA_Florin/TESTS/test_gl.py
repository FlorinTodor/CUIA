import os, pyrender, trimesh
os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # mismo backend

mesh = pyrender.Mesh.from_trimesh(trimesh.creation.icosphere())
r = pyrender.OffscreenRenderer(640,480)      # <- ¿segfault aquí?
color, _ = r.render(pyrender.Scene(meshes=[mesh]))
print("OK, se pudo renderizar un triángulo")
