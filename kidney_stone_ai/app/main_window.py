from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from preprocessing import ScanData, load_scan, preprocess
from visualization import show_orthogonal_slices


def slice_to_qpixmap(
    volume: np.ndarray,
    z_index: int,
    clim: tuple[float, float] | None = None,
) -> QPixmap:
    """(z,y,x) volume'dan bir axial dilimi QPixmap'e çevirir."""
    plane = np.asarray(volume[z_index], dtype=np.float32)  # (y, x)

    if clim is None:
        v_min = float(plane.min())
        v_max = float(plane.max())
    else:
        v_min, v_max = clim

    if v_max <= v_min:
        norm = np.zeros_like(plane, dtype=np.uint8)
    else:
        clipped = np.clip(plane, v_min, v_max)
        norm = ((clipped - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)

    # QImage: satır satır, grayscale
    y, x = norm.shape
    bytes_per_line = x
    image = QImage(norm.data, x, y, bytes_per_line, QImage.Format.Format_Grayscale8)
    # QImage, numpy buffer'a bağlı kalmasın diye kopyala
    image = image.copy()
    return QPixmap.fromImage(image)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kidney Stone AI")
        self.resize(900, 700)

        self.scan: ScanData | None = None
        self.display_scan: ScanData | None = None

        self.path_label = QLabel("No file loaded")
        self.path_label.setWordWrap(True)

        self.info_label = QLabel("shape / spacing / HU: —")
        self.info_label.setWordWrap(True)

        self.image_label = QLabel("Load a NIfTI file or DICOM folder")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(512, 512)
        self.image_label.setStyleSheet("background-color: #1e1e1e; color: #ccc;")

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)

        self.slice_label = QLabel("Slice: —")

        self.preprocess_check = QCheckBox("Use preprocess (clip → resample → normalize)")
        self.preprocess_check.setChecked(False)
        self.preprocess_check.stateChanged.connect(self.on_preprocess_toggled)

        btn_nifti = QPushButton("Open NIfTI…")
        btn_nifti.clicked.connect(self.open_nifti)

        btn_dicom = QPushButton("Open DICOM folder…")
        btn_dicom.clicked.connect(self.open_dicom)

        btn_3d = QPushButton("Show 3D (PyVista)")
        btn_3d.clicked.connect(self.open_3d)

        top_row = QHBoxLayout()
        top_row.addWidget(btn_nifti)
        top_row.addWidget(btn_dicom)
        top_row.addWidget(btn_3d)

        slice_row = QHBoxLayout()
        slice_row.addWidget(self.slice_label)
        slice_row.addWidget(self.slice_slider)

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(self.preprocess_check)
        layout.addWidget(self.path_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.image_label, stretch=1)
        layout.addLayout(slice_row)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def open_nifti(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select NIfTI",
            "",
            "NIfTI (*.nii *.nii.gz);;All files (*)",
        )
        if path_str:
            self.load_path(Path(path_str))

    def open_dicom(self) -> None:
        path_str = QFileDialog.getExistingDirectory(self, "Select DICOM folder")
        if path_str:
            self.load_path(Path(path_str))

    def load_path(self, path: Path) -> None:
        try:
            self.scan = load_scan(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        self.path_label.setText(f"Loaded: {path}")
        self.refresh_display_scan()

    def refresh_display_scan(self) -> None:
        if self.scan is None:
            return

        try:
            if self.preprocess_check.isChecked():
                self.display_scan = preprocess(self.scan)
            else:
                self.display_scan = self.scan
        except Exception as exc:
            QMessageBox.critical(self, "Preprocess error", str(exc))
            return

        vol = self.display_scan.volume
        z, y, x = vol.shape
        sx, sy, sz = self.display_scan.spacing

        self.info_label.setText(
            f"shape (z,y,x): ({z}, {y}, {x}) | "
            f"spacing mm: ({sx:.3f}, {sy:.3f}, {sz:.3f}) | "
            f"min/max: {float(vol.min()):.1f} / {float(vol.max()):.1f}"
        )

        self.slice_slider.blockSignals(True)
        self.slice_slider.setEnabled(True)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(z - 1)
        self.slice_slider.setValue(z // 2)
        self.slice_slider.blockSignals(False)

        self.update_slice_view(self.slice_slider.value())

    def on_preprocess_toggled(self) -> None:
        if self.scan is not None:
            self.refresh_display_scan()

    def on_slice_changed(self, value: int) -> None:
        self.update_slice_view(value)

    def update_slice_view(self, z_index: int) -> None:
        if self.display_scan is None:
            return

        vol = self.display_scan.volume
        z_index = int(np.clip(z_index, 0, vol.shape[0] - 1))
        self.slice_label.setText(f"Slice: {z_index} / {vol.shape[0] - 1}")

        # RAW HU ise pencerele; normalize [0,1] ise otomatik
        clim: tuple[float, float] | None
        if self.preprocess_check.isChecked():
            clim = None
        else:
            clim = (-200.0, 400.0)

        pix = slice_to_qpixmap(vol, z_index, clim=clim)
        scaled = pix.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.display_scan is not None:
            self.update_slice_view(self.slice_slider.value())

    def open_3d(self) -> None:
        if self.display_scan is None:
            QMessageBox.information(self, "No data", "Load a scan first.")
            return

        if self.preprocess_check.isChecked():
            show_orthogonal_slices(self.display_scan, title="Preprocessed CT")
        else:
            show_orthogonal_slices(
                self.display_scan,
                title="Raw CT",
                clim=(-200.0, 400.0),
            )