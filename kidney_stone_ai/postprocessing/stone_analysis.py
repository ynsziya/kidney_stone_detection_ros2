from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai.roi_crop import KidneyROI
from ai.stone_segmentation import StoneSegResult
from postprocessing.measurements import (
    bbox_from_indices,
    equivalent_diameter_mm,
    volume_mm3,
)


@dataclass
class StoneProperties:
    """Tek bir taş adayının ölçümleri."""

    laterality: str
    stone_id: int
    n_voxels: int
    volume_mm3: float
    diameter_mm: float
    mean_hu: float
    max_hu: float
    min_hu: float
    centroid_zyx: tuple[float, float, float]
    """ROI indeks uzayında (z, y, x)."""

    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    """Inclusive ((z0,z1), (y0,y1), (x0,x1))."""

    hu_threshold: float


def analyze_stone_components(
    roi: KidneyROI,
    seg: StoneSegResult,
) -> list[StoneProperties]:
    """
    Etiketli stone maskesindeki her bileşen için ölçüm üretir.
    """
    volume = np.asarray(roi.volume, dtype=np.float32)
    mask = np.asarray(seg.mask)
    spacing = roi.spacing
    props: list[StoneProperties] = []

    for stone_id in range(1, seg.n_components + 1):
        comp = mask == stone_id
        n_vox = int(comp.sum())
        if n_vox == 0:
            continue

        zs, ys, xs = np.where(comp)
        hu_vals = volume[comp]
        vol = volume_mm3(n_vox, spacing)
        diam = equivalent_diameter_mm(vol)

        props.append(
            StoneProperties(
                laterality=roi.laterality,
                stone_id=stone_id,
                n_voxels=n_vox,
                volume_mm3=vol,
                diameter_mm=diam,
                mean_hu=float(hu_vals.mean()),
                max_hu=float(hu_vals.max()),
                min_hu=float(hu_vals.min()),
                centroid_zyx=(float(zs.mean()), float(ys.mean()), float(xs.mean())),
                bbox_zyx=bbox_from_indices(zs, ys, xs),
                hu_threshold=seg.hu_threshold,
            )
        )

    return props


def analyze_all_rois(
    rois: list[KidneyROI],
    stone_results: dict[str, StoneSegResult],
) -> list[StoneProperties]:
    """Sol/sağ tüm ROI sonuçlarını tek listede birleştirir."""
    all_props: list[StoneProperties] = []
    roi_by_side = {r.laterality: r for r in rois}
    for side, seg in stone_results.items():
        roi = roi_by_side.get(side)
        if roi is None:
            continue
        all_props.extend(analyze_stone_components(roi, seg))
    return all_props


def format_stone_report(props: list[StoneProperties]) -> str:
    if not props:
        return "No stone candidates."
    lines = []
    for p in props:
        z0, z1 = p.bbox_zyx[0]
        lines.append(
            f"{p.laterality} #{p.stone_id}: "
            f"{p.volume_mm3:.1f} mm³, d≈{p.diameter_mm:.1f} mm, "
            f"HU mean/max/min={p.mean_hu:.0f}/{p.max_hu:.0f}/{p.min_hu:.0f}, "
            f"centroid z={p.centroid_zyx[0]:.1f}, z-range=[{z0}-{z1}]"
        )
    return "\n".join(lines)