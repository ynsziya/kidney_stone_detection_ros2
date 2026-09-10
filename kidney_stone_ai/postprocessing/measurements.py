from __future__ import annotations

import numpy as np


def voxel_volume_mm3(spacing: tuple[float, float, float]) -> float:
    """Bir voxel'in mm³ hacmi. spacing = (sx, sy, sz)."""
    sx, sy, sz = spacing
    return float(sx * sy * sz)


def volume_mm3(n_voxels: int, spacing: tuple[float, float, float]) -> float:
    return float(n_voxels) * voxel_volume_mm3(spacing)


def equivalent_diameter_mm(volume_mm3_value: float) -> float:
    """
    Aynı hacimli kürenin çapı (mm).

    V = 4/3 * pi * r^3  →  d = 2 * (3V / 4pi)^(1/3)
    """
    if volume_mm3_value <= 0:
        return 0.0
    radius = ((3.0 * volume_mm3_value) / (4.0 * np.pi)) ** (1.0 / 3.0)
    return float(2.0 * radius)


def bbox_from_indices(
    zs: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """((z0,z1), (y0,y1), (x0,x1)) inclusive."""
    return (
        (int(zs.min()), int(zs.max())),
        (int(ys.min()), int(ys.max())),
        (int(xs.min()), int(xs.max())),
    )