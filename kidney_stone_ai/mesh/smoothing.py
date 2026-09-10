from __future__ import annotations

import numpy as np

from mesh.marching_cubes import MeshData


def smooth_mesh(mesh: MeshData, iterations: int = 15) -> MeshData:
    """
    Basit Laplacian benzeri yumuşatma (PyVista Taubin).
    iterations=0 ise dokunulmaz.
    """
    if iterations <= 0:
        return mesh

    import pyvista as pv

    faces = np.hstack(
        [
            np.full((len(mesh.faces), 1), 3, dtype=np.int64),
            mesh.faces.astype(np.int64),
        ]
    ).ravel()
    poly = pv.PolyData(mesh.vertices, faces)
    poly = poly.smooth_taubin(n_iter=iterations, pass_band=0.1)
    # PyVista faces: [3, i,j,k, 3, ...]
    f = poly.faces.reshape(-1, 4)[:, 1:]
    return MeshData(
        vertices=np.asarray(poly.points, dtype=np.float64),
        faces=np.asarray(f, dtype=np.int64),
    )