from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from preprocessing.scan_data import ScanData, sitk_image_to_scan

def load_dicom_series(directory: str | Path) -> ScanData:
    """
    Bir DICOM series klasörünü 3D volume olarak yükler.
    `directory` içinde aynı series'e ait .dcm (veya uzantısız) slice'lar olmalı.
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"DICOM directory not found: {directory}")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(directory))
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in: {directory}")

    # Klasörde birden fazla series varsa ilkini al; uyarı bas.
    if len(series_ids) > 1:
        print(
            f"Warning: {len(series_ids)} DICOM series found; "
            f"using the first one: {series_ids[0]}"
        )

    file_names = reader.GetGDCMSeriesFileNames(str(directory), series_ids[0])
    if not file_names:
        raise FileNotFoundError(f"DICOM series has no files: {directory}")

    reader.SetFileNames(file_names)
    image = reader.Execute()
    return sitk_image_to_scan(image, directory)