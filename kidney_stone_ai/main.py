from __future__ import annotations

import sys
from pathlib import Path

from preprocessing import load_scan, preprocess


def print_scan_info(title: str, scan) -> None:
    vol = scan.volume
    print(f"=== {title} ===")
    print(f"path:    {scan.path}")
    print(f"shape:   {vol.shape}  (z, y, x)")
    print(f"spacing: {scan.spacing}  (sx, sy, sz) mm")
    print(f"min:     {float(vol.min()):.4f}")
    print(f"max:     {float(vol.max()):.4f}")
    print(f"mean:    {float(vol.mean()):.4f}")
    print()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <nifti_or_dicom_path>")
        return

    raw = load_scan(Path(sys.argv[1]))
    print_scan_info("RAW", raw)

    processed = preprocess(raw)
    print_scan_info("PREPROCESSED", processed)


if __name__ == "__main__":
    main()