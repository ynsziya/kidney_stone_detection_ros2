from __future__ import annotations

import numpy as np
import pyvista as pv

from mesh.marching_cubes import MeshData


def _to_poly(mesh: MeshData) -> pv.PolyData:
    faces = np.hstack(
        [
            np.full((len(mesh.faces), 1), 3, dtype=np.int64),
            mesh.faces.astype(np.int64),
        ]
    ).ravel()
    return pv.PolyData(mesh.vertices, faces)


def show_meshes(
    kidney_meshes: list[tuple[str, MeshData]],
    stone_meshes: list[tuple[str, MeshData]],
    title: str = "Kidney + Stones",
) -> None:
    """
    Böbrek: yarı saydam. Taş: opak sarı/turuncu.
    """
    pl = pv.Plotter()
    pl.add_title(title, font_size=12)

    kidney_colors = {"left": "lightcoral", "right": "lightskyblue"}
    for name, mesh in kidney_meshes:
        color = kidney_colors.get(name, "lightgray")
        pl.add_mesh(
            _to_poly(mesh),
            color=color,
            opacity=0.35,
            smooth_shading=True,
            name=f"kidney_{name}",
        )

    for name, mesh in stone_meshes:
        pl.add_mesh(
            _to_poly(mesh),
            color="gold",
            opacity=1.0,
            smooth_shading=True,
            name=f"stone_{name}",
        )

    pl.add_axes()
    pl.show()