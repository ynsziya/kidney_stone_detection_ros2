from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from preprocessing.scan_data import ScanData, sitk_image_to_scan


@dataclass
class KidneyMask:
    """Böbrek maskesi — CT ile aynı geometride."""

    mask: np.ndarray
    """uint8, shape (z,y,x). 0=bg, 1=left, 2=right."""

    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    path: Path
    """Kaynak CT yolu."""


def _scan_to_sitk(scan: ScanData) -> sitk.Image:
    image = sitk.GetImageFromArray(scan.volume.astype(np.float32))
    image.SetSpacing(scan.spacing)
    image.SetOrigin(scan.origin)
    image.SetDirection(scan.direction)
    return image


def _ensure_nifti_input(scan: ScanData, work_dir: Path) -> Path:
    """
    TotalSegmentator dosya yolu ister.
    NIfTI ise orijinal path; değilse (DICOM) temp .nii.gz yaz.
    """
    path = Path(scan.path)
    name = path.name.lower()
    if path.is_file() and (name.endswith(".nii") or name.endswith(".nii.gz")):
        return path

    out = work_dir / "input_ct.nii.gz"
    sitk.WriteImage(_scan_to_sitk(scan), str(out))
    return out


def _load_binary_mask(path: Path, reference: sitk.Image) -> np.ndarray:
    if not path.is_file():
        return np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=np.uint8)
    img = sitk.ReadImage(str(path))
    # Gerekirse referans grid'e yeniden örnekle
    if (
        img.GetSize() != reference.GetSize()
        or img.GetSpacing() != reference.GetSpacing()
        or img.GetOrigin() != reference.GetOrigin()
        or img.GetDirection() != reference.GetDirection()
    ):
        img = sitk.Resample(
            img,
            reference,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            img.GetPixelID(),
        )
    arr = sitk.GetArrayFromImage(img)
    return (arr > 0).astype(np.uint8)


def segment_kidneys(
    scan: ScanData,
    *,
    fast: bool = True,
    device: str = "gpu",
    output_dir: str | Path | None = None,
) -> KidneyMask:
    """
    TotalSegmentator ile sol/sağ böbrek maskesi üretir.

    device: "gpu" | "cpu" | "mps"
    fast=True: düşük çözünürlük, prototip için uygun (özellikle CPU).
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise ImportError(
            "TotalSegmentator yüklü değil. "
            "pip install TotalSegmentator"
        ) from exc

    cleanup_tmp = False
    if output_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="kidney_seg_")
        work = Path(tmp.name)
        cleanup_tmp = True
    else:
        tmp = None
        work = Path(output_dir)
        work.mkdir(parents=True, exist_ok=True)

    try:
        input_path = _ensure_nifti_input(scan, work)
        seg_dir = work / "ts_out"
        seg_dir.mkdir(exist_ok=True)

        # İlk indirmede model download olur
        totalsegmentator(
            str(input_path),
            str(seg_dir),
            roi_subset=["kidney_left", "kidney_right"],
            fast=fast,
            device=device,
            quiet=False,
        )

        reference = _scan_to_sitk(scan)
        left = _load_binary_mask(seg_dir / "kidney_left.nii.gz", reference)
        right = _load_binary_mask(seg_dir / "kidney_right.nii.gz", reference)

        combined = np.zeros(left.shape, dtype=np.uint8)
        combined[left > 0] = 1
        combined[right > 0] = 2
        # Çakışma olursa right öncelikli (nadir)
        combined[right > 0] = 2

        return KidneyMask(
            mask=combined,
            spacing=scan.spacing,
            origin=scan.origin,
            direction=scan.direction,
            path=Path(scan.path),
        )
    finally:
        if cleanup_tmp and tmp is not None:
            tmp.cleanup()


def mask_voxel_counts(mask: np.ndarray) -> dict[str, int]:
    return {
        "left": int(np.sum(mask == 1)),
        "right": int(np.sum(mask == 2)),
        "background": int(np.sum(mask == 0)),
    }