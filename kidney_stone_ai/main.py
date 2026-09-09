from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app import MainWindow
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


def run_cli(path: Path, do_preprocess: bool) -> None:
    raw = load_scan(path)
    print_scan_info("RAW", raw)
    if do_preprocess:
        scan = preprocess(raw)
        print_scan_info("PREPROCESSED", scan)
        show_orthogonal_slices(scan, title="Preprocessed CT")
    else:
        show_orthogonal_slices(raw, title="Raw CT", clim=(-200.0, 400.0))


def run_gui() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def main() -> None:
    parser = argparse.ArgumentParser(description="Kidney stone AI")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Optional: CLI mode — NIfTI file or DICOM folder",
    )
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Force CLI/PyVista mode (no GUI)",
    )
    args = parser.parse_args()

    # Path verildiyse veya --cli ise eski davranış; yoksa GUI
    if args.cli or args.path is not None:
        if args.path is None:
            parser.error("CLI mode requires a path")
        run_cli(args.path, args.preprocess)
    else:
        run_gui()


if __name__ == "__main__":
    main()