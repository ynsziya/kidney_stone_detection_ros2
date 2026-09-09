from __future__ import annotations

import argparse
from pathlib import Path

from preprocessing import load_scan, preprocess
from visualization import show_orthogonal_slices


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
    parser = argparse.ArgumentParser(description="Kidney stone AI — load & view CT")
    parser.add_argument("path", type=Path, help="NIfTI file or DICOM folder")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Show preprocessed volume instead of raw",
    )
    args = parser.parse_args()

    raw = load_scan(args.path)
    print_scan_info("RAW", raw)

    if args.preprocess:
        scan = preprocess(raw)
        print_scan_info("PREPROCESSED", scan)
        # 0–1 arası; clim otomatik
        show_orthogonal_slices(scan, title="Preprocessed CT")
    else:
        # RAW HU: yumuşak doku / böbrek penceresi
        show_orthogonal_slices(
            raw,
            title="Raw CT",
            clim=(-200.0, 400.0),
        )


if __name__ == "__main__":
    main()