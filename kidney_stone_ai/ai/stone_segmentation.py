from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ai.roi_crop import KidneyROI


@dataclass
class StoneSegResult:
    """Tek ROI için geçici taş adayı sonucu."""

    mask: np.ndarray
    """uint8, shape ROI ile aynı. 0=bg, 1,2,... = ayrı taş bileşenleri."""

    binary_mask: np.ndarray
    """uint8, 0/1 — tüm taş adayları birleşik."""

    laterality: str
    n_components: int
    hu_threshold: float
    min_voxels: int


def segment_stone(
    roi: KidneyROI,
    *,
    hu_threshold: float = 300.0,
    min_voxels: int = 8,
    dilate_kidney_voxels: int = 2,
) -> StoneSegResult:
    """
    Geçici (heuristik) taş segmentasyonu.

    Arayüz sabit kalsın: ileride nnU-Net/MONAI bu fonksiyonun içini değiştirir.
    """
    volume = np.asarray(roi.volume, dtype=np.float32)
    kidney = np.asarray(roi.mask, dtype=bool)

    if dilate_kidney_voxels > 0:
        structure = ndimage.generate_binary_structure(3, 1)
        kidney = ndimage.binary_dilation(
            kidney,
            structure=structure,
            iterations=dilate_kidney_voxels,
        )

    candidates = (volume >= hu_threshold) & kidney

    labeled, n = ndimage.label(candidates)
    if n == 0:
        empty = np.zeros(volume.shape, dtype=np.uint8)
        return StoneSegResult(
            mask=empty,
            binary_mask=empty.copy(),
            laterality=roi.laterality,
            n_components=0,
            hu_threshold=hu_threshold,
            min_voxels=min_voxels,
        )

    # Küçük bileşenleri at; kalanları 1..K olarak yeniden numarala
    kept = np.zeros(volume.shape, dtype=np.uint8)
    new_id = 0
    for comp_id in range(1, n + 1):
        comp = labeled == comp_id
        if int(comp.sum()) < min_voxels:
            continue
        new_id += 1
        kept[comp] = new_id

    binary = (kept > 0).astype(np.uint8)

    return StoneSegResult(
        mask=kept,
        binary_mask=binary,
        laterality=roi.laterality,
        n_components=new_id,
        hu_threshold=hu_threshold,
        min_voxels=min_voxels,
    )


def segment_stones_in_rois(
    rois: list[KidneyROI],
    **kwargs,
) -> dict[str, StoneSegResult]:
    """Sol/sağ tüm ROI'ler için segment_stone çalıştırır."""
    return {roi.laterality: segment_stone(roi, **kwargs) for roi in rois}