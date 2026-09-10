from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import measure


@dataclass
class MeshData:
    vertices: np.ndarray  # (N, 3) dünya mm: X, Y, Z
    faces: np.ndarray     # (M, 3) int


def _mask_to_verts_faces_zyx(
    mask_zyx: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    mask shape (z,y,x). skimage spacing sırası volume eksenleriyle aynı: (sz, sy, sx).
    Dönen vertices geçici olarak (z,y,x) mm (array köşesinden).
    """
    binary = (mask_zyx > 0).astype(np.float32)
    if binary.sum() < 8:
        raise ValueError("Mask too small for marching cubes")

    sx, sy, sz = spacing_xyz
    verts_zyx, faces, *_ = measure.marching_cubes(
        binary,
        level=0.5,
        spacing=(sz, sy, sx),
    )
    return verts_zyx.astype(np.float64), faces.astype(np.int64)


def verts_zyx_to_world(
    verts_zyx: np.ndarray,
    origin: tuple[float, float, float],
    direction: tuple[float, ...],
) -> np.ndarray:
    """
    marching_cubes (z,y,x) mm → dünya (X,Y,Z) mm.

    SimpleITK: world = origin + direction @ (x, y, z)
    """
    direction_matrix = np.array(direction, dtype=np.float64).reshape(3, 3)
    # verts columns: z, y, x  →  physical offset x,y,z
    offset_xyz = np.column_stack(
        [verts_zyx[:, 2], verts_zyx[:, 1], verts_zyx[:, 0]]
    )
    world = np.asarray(origin, dtype=np.float64) + (direction_matrix @ offset_xyz.T).T
    return world


def create_mesh_from_mask(
    mask_zyx: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    direction: tuple[float, ...],
) -> MeshData:
    verts_zyx, faces = _mask_to_verts_faces_zyx(mask_zyx, spacing)
    vertices = verts_zyx_to_world(verts_zyx, origin, direction)
    return MeshData(vertices=vertices, faces=faces)