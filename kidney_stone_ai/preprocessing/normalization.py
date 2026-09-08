from __future__ import annotations

import numpy as np

from preprocessing.scan_data import ScanData


def normalize_volume(
    scan: ScanData,
    method: str = "minmax",
) -> ScanData:
    """
    Intensity normalize eder. Geometri değişmez.

    method:
      - "minmax": [0, 1] aralığına çeker (clip sonrası için uygun)
      - "zscore": mean/std ile standardize eder
    """
    volume = scan.volume.astype(np.float32)

    if method == "minmax":
        v_min = float(volume.min())
        v_max = float(volume.max())
        if v_max > v_min:
            normalized = (volume - v_min) / (v_max - v_min)
        else:
            normalized = np.zeros_like(volume)
    elif method == "zscore":
        mean = float(volume.mean())
        std = float(volume.std())
        if std > 1e-8:
            normalized = (volume - mean) / std
        else:
            normalized = volume - mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return ScanData(
        volume=normalized.astype(np.float32),
        spacing=scan.spacing,
        origin=scan.origin,
        direction=scan.direction,
        affine=scan.affine.copy(),
        path=scan.path,
    )