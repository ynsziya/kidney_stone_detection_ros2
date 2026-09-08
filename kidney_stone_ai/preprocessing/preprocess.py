from __future__ import annotations

from preprocessing.hu_clipping import clip_hu
from preprocessing.normalization import normalize_volume
from preprocessing.resampling import resample_volume
from preprocessing.scan_data import ScanData


def preprocess(
    scan: ScanData,
    hu_min: float = -200.0,
    hu_max: float = 1500.0,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    normalize: bool = True,
    norm_method: str = "minmax",
) -> ScanData:
    """
    Standart CT ön işleme zinciri:

        clip_hu → resample → (opsiyonel) normalize
    """
    out = clip_hu(scan, hu_min=hu_min, hu_max=hu_max)
    out = resample_volume(out, target_spacing=target_spacing)
    if normalize:
        out = normalize_volume(out, method=norm_method)
    return out