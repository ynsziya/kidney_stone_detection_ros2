from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preprocessing.scan_data import ScanData, build_affine


@dataclass
class KidneyROI:
    """Tek böbrek için kırpılmış CT + maske."""

    laterality: str
    """'left' veya 'right'."""

    volume: np.ndarray
    """float32, shape (z,y,x) — ROI CT."""

    mask: np.ndarray
    """uint8, shape ROI ile aynı; bu böbrek = 1, dışı = 0."""

    bbox: tuple[slice, slice, slice]
    """Tam CT üzerinde (z_slice, y_slice, x_slice)."""

    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    affine: np.ndarray
    """ROI'nin kendi 4x4 affine'i (dünya mm)."""

    label: int
    """Tam maskede 1=left, 2=right."""


def _bbox_from_mask(
    binary: np.ndarray,
    margin: int,
) -> tuple[slice, slice, slice] | None:
    """
    binary: (z,y,x) bool/0-1
    margin: her yönde ek voxel
    """
    coords = np.argwhere(binary > 0)
    if coords.size == 0:
        return None

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1  # Python slice: exclusive end

    z, y, x = binary.shape
    z0 = max(0, int(z0) - margin)
    y0 = max(0, int(y0) - margin)
    x0 = max(0, int(x0) - margin)
    z1 = min(z, int(z1) + margin)
    y1 = min(y, int(y1) + margin)
    x1 = min(x, int(x1) + margin)

    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def _origin_for_crop(
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    direction: tuple[float, ...],
    x0: int,
    y0: int,
    z0: int,
) -> tuple[float, float, float]:
    """
    Tam volume'daki voxel indeksi (x0,y0,z0) köşesinin dünya koordinatı.
    SimpleITK: physical = origin + direction @ (i*sx, j*sy, k*sz)
    """
    direction_matrix = np.array(direction, dtype=np.float64).reshape(3, 3)
    sx, sy, sz = spacing
    offset = direction_matrix @ np.array(
        [x0 * sx, y0 * sy, z0 * sz],
        dtype=np.float64,
    )
    new_origin = np.array(origin, dtype=np.float64) + offset
    return tuple(float(v) for v in new_origin)


def extract_kidney_rois(
    scan: ScanData,
    kidney_mask: np.ndarray,
    margin: int = 25,
) -> list[KidneyROI]:
    """
    Sol ve sağ böbrek için ayrı ROI üretir.

    kidney_mask: 0=bg, 1=left, 2=right — scan.volume ile aynı shape.
    margin: bbox'a eklenecek voxel (öneri 20–30).
    """
    if kidney_mask.shape != scan.volume.shape:
        raise ValueError(
            f"Mask shape {kidney_mask.shape} != volume shape {scan.volume.shape}"
        )

    rois: list[KidneyROI] = []
    label_map = {1: "left", 2: "right"}

    for label, name in label_map.items():
        binary = (kidney_mask == label).astype(np.uint8)
        bbox = _bbox_from_mask(binary, margin=margin)
        if bbox is None:
            continue

        z_sl, y_sl, x_sl = bbox
        vol_crop = np.ascontiguousarray(
            scan.volume[z_sl, y_sl, x_sl], dtype=np.float32
        )
        mask_crop = np.ascontiguousarray(
            binary[z_sl, y_sl, x_sl], dtype=np.uint8
        )

        z0, y0, x0 = z_sl.start, y_sl.start, x_sl.start
        new_origin = _origin_for_crop(
            scan.origin, scan.spacing, scan.direction, x0, y0, z0
        )
        new_affine = build_affine(scan.spacing, new_origin, scan.direction)

        rois.append(
            KidneyROI(
                laterality=name,
                volume=vol_crop,
                mask=mask_crop,
                bbox=bbox,
                spacing=scan.spacing,
                origin=new_origin,
                direction=scan.direction,
                affine=new_affine,
                label=label,
            )
        )

    return rois


def roi_summary(rois: list[KidneyROI]) -> str:
    parts = []
    for r in rois:
        z, y, x = r.volume.shape
        z_sl, y_sl, x_sl = r.bbox
        parts.append(
            f"{r.laterality}: shape=({z},{y},{x}) "
            f"bbox z[{z_sl.start}:{z_sl.stop}] "
            f"y[{y_sl.start}:{y_sl.stop}] "
            f"x[{x_sl.start}:{x_sl.stop}]"
        )
    return " | ".join(parts) if parts else "no kidney ROIs"