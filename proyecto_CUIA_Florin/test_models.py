import models, numpy as np

def test_modelos_flotan():
    for idx, mesh in models.modelos_precargados.items():
        min_z = mesh.bounds[0][2]
        assert np.isclose(min_z, 0.0, atol=1e-4), f"Modelo {idx} se hunde {min_z:.4f} m"
