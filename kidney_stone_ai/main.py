from __future__ import annotations

import sys
from pathlib import Path

from preprocessing import load_scan


def print_scan_info(path: Path) -> None:
    scan = load_scan(path)
    vol = scan.volume

    print("=== Scan loaded ===")
    print(f"path:     {scan.path}")
    print(f"shape:    {vol.shape}  (z, y, x)")
    print(f"dtype:    {vol.dtype}")
    print(f"spacing:  {scan.spacing}  (sx, sy, sz) mm")
    print(f"origin:   {scan.origin}")
    print(f"direction:{scan.direction}")
    print(f"affine:\n{scan.affine}")
    print(f"HU min:   {float(vol.min()):.1f}")
    print(f"HU max:   {float(vol.max()):.1f}")
    print(f"HU mean:  {float(vol.mean()):.1f}")


def check_folders() -> None:
    root = Path(__file__).resolve().parent
    print("kidney_stone_ai is ready")
    print(f"project root: {root}")
    expected = [
        "app", "ai", "preprocessing", "postprocessing",
        "mesh", "visualization", "models", "data",
    ]
    missing = [name for name in expected if not (root / name).is_dir()]
    if missing:
        print(f"missing folders: {missing}")
    else:
        print("folder check: OK")


def main() -> None:
    if len(sys.argv) < 2:
        check_folders()
        print()
        print("Usage:")
        print("  python main.py data/nifti/your_file.nii.gz")
        print("  python main.py data/dicom/your_series_folder")
        return

    print_scan_info(Path(sys.argv[1]))


if __name__ == "__main__":
    main()