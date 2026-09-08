from __future__ import annotations

import numpy as np

from preprocessing.scan_data import ScanData


def clip_hu(
    scan: ScanData,
    hu_min: float = -200.0,
    hu_max: float = 1500.0,
) -> ScanData:
    """
    HU değerlerini [hu_min, hu_max] aralığına kırpar.

    Geometri (spacing/origin/affine) değişmez; sadece volume güncellenir.
    """
    clipped = np.clip(scan.volume, hu_min, hu_max).astype(np.float32)

    return ScanData(
        volume=clipped,
        spacing=scan.spacing,
        origin=scan.origin,
        direction=scan.direction,
        affine=scan.affine.copy(),
        path=scan.path,
    )