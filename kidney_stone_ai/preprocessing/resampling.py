from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from preprocessing.scan_data import ScanData, sitk_image_to_scan


def _scan_to_sitk(scan: ScanData) -> sitk.Image:
    """ScanData (z,y,x) → SimpleITK Image."""
    image = sitk.GetImageFromArray(scan.volume)
    image.SetSpacing(scan.spacing)
    image.SetOrigin(scan.origin)
    image.SetDirection(scan.direction)
    return image


def resample_volume(
    scan: ScanData,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator: int = sitk.sitkLinear,
) -> ScanData:
    """
    Volume'u target_spacing'e yeniden örnekler (mm).

    target_spacing: (sx, sy, sz) — fiziksel X,Y,Z.
    Shape otomatik değişir; spacing/origin/direction/affine güncellenir.
    """
    image = _scan_to_sitk(scan)

    original_spacing = np.array(image.GetSpacing(), dtype=np.float64)
    original_size = np.array(image.GetSize(), dtype=np.float64)  # (x, y, z)
    target_spacing_arr = np.array(target_spacing, dtype=np.float64)

    # Yeni boyut: fiziksel uzunluk / yeni spacing
    new_size = np.round(original_size * (original_spacing / target_spacing_arr)).astype(int)
    new_size = [int(max(1, s)) for s in new_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(s) for s in target_spacing_arr))
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(scan.volume.min()))

    resampled = resampler.Execute(image)
    return sitk_image_to_scan(resampled, scan.path)