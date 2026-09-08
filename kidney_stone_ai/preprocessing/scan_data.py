from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ScanData:
    """Tek bir CT hacmi + geometri metadata."""

    volume: np.ndarray
    """HU değerleri. Shape: (z, y, x) — SimpleITK GetArrayFromImage düzeni."""

    spacing: tuple[float, float, float]
    """(sx, sy, sz) mm — fiziksel X, Y, Z eksenleri."""

    origin: tuple[float, float, float]
    """Dünya koordinatında origin (mm)."""

    direction: tuple[float, ...]
    """3x3 yön matrisi, satır satır 9 float (SimpleITK)."""

    affine: np.ndarray
    """4x4 affine: voxel (i,j,k) ~ X,Y,Z indeks → dünya mm."""

    path: Path
    """Yüklenen dosya veya DICOM klasör yolu."""


def sitk_image_to_scan(image, path: Path) -> ScanData:
    """SimpleITK Image → ScanData."""
    import SimpleITK as sitk

    volume = sitk.GetArrayFromImage(image).astype(np.float32)  # (z, y, x)
    spacing = tuple(float(v) for v in image.GetSpacing())  # (sx, sy, sz)
    origin = tuple(float(v) for v in image.GetOrigin())
    direction = tuple(float(v) for v in image.GetDirection())

    affine = build_affine(spacing, origin, direction)

    return ScanData(
        volume=volume,
        spacing=spacing,
        origin=origin,
        direction=direction,
        affine=affine,
        path=path,
    )


def build_affine(
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    direction: tuple[float, ...],
) -> np.ndarray:
    """
    SimpleITK geometrisinden 4x4 affine üretir.

    Voxel indeksi (i, j, k) = (x_index, y_index, z_index) için:
    world = origin + direction @ (i*sx, j*sy, k*sz)
    """
    direction_matrix = np.array(direction, dtype=np.float64).reshape(3, 3)
    scale = np.diag(np.array(spacing, dtype=np.float64))
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction_matrix @ scale
    affine[:3, 3] = np.array(origin, dtype=np.float64)
    return affine