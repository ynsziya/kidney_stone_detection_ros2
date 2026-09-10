from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from mesh.marching_cubes import MeshData


def export_mesh(mesh: MeshData, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tm = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        process=False,
    )
    # STL için normals düzelt
    tm.fix_normals()
    tm.export(path)
    return path.resolve()