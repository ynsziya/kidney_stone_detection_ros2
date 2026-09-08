from __future__ import annotations

from pathlib import Path

from preprocessing.dicom_reader import load_dicom_series
from preprocessing.nifti_reader import load_nifti
from preprocessing.scan_data import ScanData

__all__ = ["ScanData", "load_scan", "load_dicom_series", "load_nifti"]

def load_scan(path: str | Path) -> ScanData:
    """
    DICOM klasörü veya NIfTI dosyası yükler.
    Örnek:
        scan = load_scan("data/nifti/case.nii.gz")
        scan = load_scan("data/dicom/patient_001")
    """
    path = Path(path).expanduser().resolve()

    if path.is_dir():
        return load_dicom_series(path)

    if path.is_file():
        name = path.name.lower()
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            return load_nifti(path)
        raise ValueError(
            f"Unsupported file type: {path}. Use .nii / .nii.gz or a DICOM folder."
        )

    raise FileNotFoundError(f"Path does not exist: {path}")