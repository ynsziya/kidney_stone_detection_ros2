from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from preprocessing.scan_data import ScanData, sitk_image_to_scan

def load_nifti(path: str | Path) -> ScanData:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    suffix = "".join(path.suffixes).lower()
    if suffix not in {".nii", ".nii.gz"} and path.suffix.lower() not in {".nii", ".gz"}:
        raise ValueError(f"Not a NIfTI file: {path}")

    image = sitk.ReadImage(str(path))
    return sitk_image_to_scan(image, path)