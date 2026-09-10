from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
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
    kidney_mask: np.ndarray | None = None,
) -> QPixmap:
    plane = np.asarray(volume[z_index], dtype=np.float32)

    if clim is None:
        v_min = float(plane.min())
        v_max = float(plane.max())
    else:
        v_min, v_max = clim

    if v_max <= v_min:
        gray = np.zeros_like(plane, dtype=np.uint8)
    else:
        clipped = np.clip(plane, v_min, v_max)
        gray = ((clipped - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)

    y, x = gray.shape
    rgb = np.stack([gray, gray, gray], axis=-1)

    if kidney_mask is not None:
        m = kidney_mask[z_index]
        left = m == 1
        right = m == 2
        rgb[left, 0] = np.clip(rgb[left, 0].astype(np.int16) + 120, 0, 255).astype(np.uint8)
        rgb[left, 1] = (rgb[left, 1] * 0.5).astype(np.uint8)
        rgb[left, 2] = (rgb[left, 2] * 0.5).astype(np.uint8)
        rgb[right, 2] = np.clip(rgb[right, 2].astype(np.int16) + 120, 0, 255).astype(np.uint8)
        rgb[right, 0] = (rgb[right, 0] * 0.5).astype(np.uint8)
        rgb[right, 1] = (rgb[right, 1] * 0.5).astype(np.uint8)

    rgb = np.ascontiguousarray(rgb)
    bytes_per_line = 3 * x
    image = QImage(rgb.data, x, y, bytes_per_line, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kidney Stone AI")
        self.resize(900, 700)

        self.scan: ScanData | None = None
        self.display_scan: ScanData | None = None
        self.kidney_mask: np.ndarray | None = None
        self.kidney_rois: list | None = None
        self.view_mode: str = "full"  # "full" | "left" | "right"

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

        btn_kidney = QPushButton("Segment kidneys")
        btn_kidney.clicked.connect(self.run_kidney_segmentation)

        btn_roi = QPushButton("Extract kidney ROIs")
        btn_roi.clicked.connect(self.extract_rois)

        btn_view_full = QPushButton("View full")
        btn_view_full.clicked.connect(lambda: self.set_view_mode("full"))

        btn_view_left = QPushButton("View left ROI")
        btn_view_left.clicked.connect(lambda: self.set_view_mode("left"))

        btn_view_right = QPushButton("View right ROI")
        btn_view_right.clicked.connect(lambda: self.set_view_mode("right"))

        top_row = QHBoxLayout()
        top_row.addWidget(btn_nifti)
        top_row.addWidget(btn_dicom)
        top_row.addWidget(btn_3d)
        top_row.addWidget(btn_kidney)
        top_row.addWidget(btn_roi)

        roi_view_row = QHBoxLayout()
        roi_view_row.addWidget(btn_view_full)
        roi_view_row.addWidget(btn_view_left)
        roi_view_row.addWidget(btn_view_right)

        slice_row = QHBoxLayout()
        slice_row.addWidget(self.slice_label)
        slice_row.addWidget(self.slice_slider)

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addLayout(roi_view_row)
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

        self.kidney_mask = None
        self.kidney_rois = None
        self.view_mode = "full"
        self.path_label.setText(f"Loaded: {path}")
        self.refresh_display_scan()

    def refresh_display_scan(self) -> None:
        if self.scan is None:
            return

        # ROI görünümündeyken preprocess full CT'ye zorlamasın
        if self.view_mode in ("left", "right"):
            self.set_view_mode(self.view_mode)
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
            f"view=full | shape (z,y,x): ({z}, {y}, {x}) | "
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
        if self.scan is not None and self.view_mode == "full":
            self.refresh_display_scan()

    def on_slice_changed(self, value: int) -> None:
        self.update_slice_view(value)

    def update_slice_view(self, z_index: int) -> None:
        if self.display_scan is None:
            return

        vol = self.display_scan.volume
        z_index = int(np.clip(z_index, 0, vol.shape[0] - 1))
        self.slice_label.setText(f"Slice: {z_index} / {vol.shape[0] - 1}")

        clim: tuple[float, float] | None
        if self.view_mode == "full" and self.preprocess_check.isChecked():
            clim = None
        else:
            clim = (-200.0, 400.0)

        mask = None
        if self.view_mode == "full":
            mask = self.kidney_mask
        elif self.kidney_rois:
            roi = next(
                (r for r in self.kidney_rois if r.laterality == self.view_mode),
                None,
            )
            if roi is not None:
                m = np.zeros_like(roi.mask, dtype=np.uint8)
                m[roi.mask > 0] = roi.label
                mask = m

        if mask is not None and mask.shape != vol.shape:
            mask = None

        pix = slice_to_qpixmap(
            vol,
            z_index,
            clim=clim,
            kidney_mask=mask,
        )
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

        if self.view_mode == "full" and self.preprocess_check.isChecked():
            show_orthogonal_slices(self.display_scan, title="Preprocessed CT")
        else:
            show_orthogonal_slices(
                self.display_scan,
                title=f"CT ({self.view_mode})",
                clim=(-200.0, 400.0),
            )

    def run_kidney_segmentation(self) -> None:
        if self.scan is None:
            QMessageBox.information(self, "No data", "Load a scan first.")
            return

        from ai import mask_voxel_counts, segment_kidneys

        self.statusBar().showMessage(
            "Running TotalSegmentator (first run downloads model)..."
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = segment_kidneys(self.scan, fast=True, device="gpu")
            self.kidney_mask = result.mask
            self.kidney_rois = None
            counts = mask_voxel_counts(result.mask)
            self.statusBar().showMessage(
                f"Kidneys done — left voxels: {counts['left']}, right: {counts['right']}"
            )
            if self.preprocess_check.isChecked():
                self.preprocess_check.setChecked(False)
            self.view_mode = "full"
            self.display_scan = self.scan
            self.update_slice_view(self.slice_slider.value())
        except Exception as exc:
            QMessageBox.critical(self, "Segmentation error", str(exc))
            self.statusBar().showMessage("Segmentation failed")
        finally:
            QApplication.restoreOverrideCursor()

    def extract_rois(self) -> None:
        if self.scan is None or self.kidney_mask is None:
            QMessageBox.information(
                self,
                "Need kidneys",
                "Önce CT yükle ve Segment kidneys çalıştır.",
            )
            return

        from ai import extract_kidney_rois, roi_summary

        try:
            self.kidney_rois = extract_kidney_rois(
                self.scan,
                self.kidney_mask,
                margin=25,
            )
        except Exception as exc:
            QMessageBox.critical(self, "ROI error", str(exc))
            return

        if not self.kidney_rois:
            QMessageBox.warning(self, "ROI", "Böbrek ROI bulunamadı.")
            return

        summary = roi_summary(self.kidney_rois)
        self.statusBar().showMessage(summary)
        print("=== Kidney ROIs ===")
        print(summary)
        for r in self.kidney_rois:
            print(
                f"  {r.laterality}: volume {r.volume.shape}, "
                f"mask voxels={int(r.mask.sum())}, origin={r.origin}"
            )

        QMessageBox.information(
            self,
            "ROIs ready",
            summary + "\n\nDetay terminalde. View left/right ile ROI dilimine geç.",
        )

    def set_view_mode(self, mode: str) -> None:
        self.view_mode = mode

        if mode == "full":
            if self.scan is None:
                return
            if self.preprocess_check.isChecked():
                self.display_scan = preprocess(self.scan)
            else:
                self.display_scan = self.scan
        else:
            if not self.kidney_rois:
                QMessageBox.information(self, "No ROI", "Önce Extract kidney ROIs.")
                self.set_view_mode("full")
                return
            roi = next((r for r in self.kidney_rois if r.laterality == mode), None)
            if roi is None:
                QMessageBox.information(self, "No ROI", f"{mode} ROI yok.")
                self.set_view_mode("full")
                return
            self.display_scan = ScanData(
                volume=roi.volume,
                spacing=roi.spacing,
                origin=roi.origin,
                direction=roi.direction,
                affine=roi.affine,
                path=self.scan.path if self.scan else Path("."),
            )

        if self.display_scan is None:
            return

        vol = self.display_scan.volume
        z = vol.shape[0]
        sx, sy, sz = self.display_scan.spacing

        self.slice_slider.blockSignals(True)
        self.slice_slider.setEnabled(True)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(max(0, z - 1))
        self.slice_slider.setValue(z // 2)
        self.slice_slider.blockSignals(False)

        self.info_label.setText(
            f"view={self.view_mode} | shape (z,y,x)={vol.shape} | "
            f"spacing mm: ({sx:.3f}, {sy:.3f}, {sz:.3f})"
        )
        self.update_slice_view(self.slice_slider.value())