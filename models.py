import pyrender
import trimesh
import numpy as np

marcadores_modelos = {
    0: "CUIA_models/pzkpfw_vi_tiger_1.glb",
    1: "CUIA_models/consolidated_b-24_liberator.glb",
    2: "CUIA_models/mp_40_submachine_gun.glb",
}


def cargar_modelo_glb(ruta):
    try:
        scene_or_mesh = trimesh.load(ruta)
        if isinstance(scene_or_mesh, trimesh.Scene):
            geometries = []
            for name, g in scene_or_mesh.geometry.items():
                try:
                    transform, _ = scene_or_mesh.graph.get(name)
                    g_copy = g.copy()
                    g_copy.apply_transform(transform)
                    geometries.append(g_copy)
                except Exception:
                    pass
            if not geometries:
                raise ValueError("La escena no contiene geometría válida.")
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = scene_or_mesh

        center = (mesh.bounds[0] + mesh.bounds[1]) / 2
        mesh.apply_translation(-center)

        target_size = 0.05
        scale_factor = target_size / np.max(mesh.extents)
        mesh.apply_scale(scale_factor)

        Ry = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
        Rx = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(Rx @ Ry)

        render_mesh = pyrender.Mesh.from_trimesh(mesh)
        return render_mesh
    except Exception as e:
        print(f"[ERROR] Fallo al cargar modelo {ruta}: {e}")
        return None


modelos_precargados = {mid: cargar_modelo_glb(path)
                       for mid, path in marcadores_modelos.items()}
