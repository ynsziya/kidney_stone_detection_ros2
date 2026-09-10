from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from preprocessing import ScanData, load_scan
from visualization import show_orthogonal_slices


def slice_to_qpixmap(
    volume: np.ndarray,
    z_index: int,
    clim: tuple[float, float] | None = None,
    kidney_mask: np.ndarray | None = None,
    stone_mask: np.ndarray | None = None,
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

    if stone_mask is not None:
        s = stone_mask[z_index] > 0
        rgb[s, 0] = 255
        rgb[s, 1] = 220
        rgb[s, 2] = 40

    rgb = np.ascontiguousarray(rgb)
    bytes_per_line = 3 * x
    image = QImage(rgb.data, x, y, bytes_per_line, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class _TeeStream(io.TextIOBase):
    """stdout/stderr'i orijinale + buffer'a yazar; satır callback ile GUI günceller."""

    def __init__(self, original, buffer: io.StringIO, on_text=None) -> None:
        super().__init__()
        self._original = original
        self._buffer = buffer
        self._on_text = on_text
        self._line_buf = ""

    def write(self, data: str) -> int:  # type: ignore[override]
        if not data:
            return 0
        self._original.write(data)
        self._buffer.write(data)
        if self._on_text is not None:
            self._line_buf += data
            while True:
                if "\r" in self._line_buf and "\n" not in self._line_buf.split("\r")[-1]:
                    # tqdm progress satırı (\r)
                    parts = self._line_buf.split("\r")
                    self._line_buf = parts[-1]
                    for part in parts[:-1]:
                        text = part.strip()
                        if text:
                            self._on_text(text, is_progress=True)
                elif "\n" in self._line_buf:
                    line, self._line_buf = self._line_buf.split("\n", 1)
                    line = line.replace("\r", "").strip()
                    if line:
                        self._on_text(line, is_progress=False)
                else:
                    break
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._buffer.flush()


def _clean_pipeline_log(raw: str) -> str:
    """tqdm çubuklarını temizleyip okunabilir satırlar bırakır."""
    lines: list[str] = []
    for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        text = line.strip()
        if not text:
            continue
        if text.startswith("100%|") or "|█" in text or text.startswith("0%|"):
            continue
        if text.startswith("If you use this tool please cite"):
            continue
        lines.append(text)
    return "\n".join(lines)


def _progress_from_line(line: str) -> tuple[int | None, str | None]:
    """TotalSegmentator satırından kabaca yüzde + kısa durum metni çıkarır."""
    lower = line.lower()
    if "using 'fast'" in lower or "using \"fast\"" in lower:
        return 8, "Fast mode (3 mm)"
    if "generating rough" in lower:
        return 18, "Rough segmentation for cropping"
    if "cropping from" in lower:
        return 55, line
    if "predicting" in lower:
        return 45, "Predicting…"
    if "resampling" in lower:
        return 30, "Resampling…"
    if "saving segmentation" in lower:
        return 85, "Saving segmentations…"
    if "saved in" in lower:
        return 95, "Saved"
    if line.startswith("100%"):
        return None, None
    if "%" in line and "|" in line:
        # tqdm: " 45%|...."
        try:
            pct = int(line.strip().split("%", 1)[0].strip().split()[-1])
            return min(99, max(1, pct)), f"Progress {pct}%"
        except ValueError:
            return None, line
    return None, None


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kidney Stone AI")
        self.resize(1280, 860)

        self.scan: ScanData | None = None
        self.display_scan: ScanData | None = None
        self.kidney_mask: np.ndarray | None = None
        self.kidney_rois: list | None = None
        self.stone_results: dict | None = None
        self.stone_properties: list | None = None
        self.view_mode: str = "full"
        self._stone_row_meta: list[tuple[str, int, int]] = []

        self.path_label = QLabel("No file loaded")
        self.path_label.setWordWrap(True)

        self.info_label = QLabel("shape / spacing / HU: —")
        self.info_label.setWordWrap(True)

        self.image_label = QLabel("Load a NIfTI file or DICOM folder")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(480, 480)
        self.image_label.setStyleSheet(
            "background-color: #1e1e1e; color: #ccc; border-radius: 6px;"
        )

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        self.slice_label = QLabel("Slice: —")

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
        btn_stones = QPushButton("Detect stones (HU)")
        btn_stones.clicked.connect(self.detect_stones)
        btn_export_mesh = QPushButton("Export STL")
        btn_export_mesh.clicked.connect(self.export_meshes)
        btn_show_mesh = QPushButton("Show 3D meshes")
        btn_show_mesh.clicked.connect(self.show_mesh_scene)
        btn_view_full = QPushButton("View full")
        btn_view_full.clicked.connect(lambda: self.set_view_mode("full"))
        btn_view_left = QPushButton("View left ROI")
        btn_view_left.clicked.connect(lambda: self.set_view_mode("left"))
        btn_view_right = QPushButton("View right ROI")
        btn_view_right.clicked.connect(lambda: self.set_view_mode("right"))

        top_row = QHBoxLayout()
        for btn in (
            btn_nifti,
            btn_dicom,
            btn_3d,
            btn_kidney,
            btn_roi,
            btn_stones,
            btn_export_mesh,
            btn_show_mesh,
        ):
            top_row.addWidget(btn)

        roi_view_row = QHBoxLayout()
        for btn in (btn_view_full, btn_view_left, btn_view_right):
            roi_view_row.addWidget(btn)
        roi_view_row.addStretch(1)

        # --- Sol: görüntü ---
        slice_row = QHBoxLayout()
        slice_row.addWidget(self.slice_label)
        slice_row.addWidget(self.slice_slider)

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.addWidget(self.path_label)
        viewer_layout.addWidget(self.info_label)
        viewer_layout.addWidget(self.image_label, stretch=1)
        viewer_layout.addLayout(slice_row)

        # --- Sağ: bilgilendirme ---
        info_title = QLabel("Pipeline info")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        info_title.setFont(title_font)

        btn_clear_log = QPushButton("Clear")
        btn_clear_log.setFixedWidth(72)
        btn_clear_log.clicked.connect(self.clear_info_log)

        info_header = QHBoxLayout()
        info_header.addWidget(info_title)
        info_header.addStretch(1)
        info_header.addWidget(btn_clear_log)

        self.info_log = QPlainTextEdit()
        self.info_log.setReadOnly(True)
        self.info_log.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.info_log.setMinimumWidth(360)
        mono = QFont("monospace")
        mono.setPointSize(10)
        self.info_log.setFont(mono)
        self.info_log.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #161a20;
                color: #d7dde5;
                border: 1px solid #2a313c;
                border-radius: 8px;
                padding: 10px;
            }
            """
        )

        self.progress_label = QLabel("Idle")
        self.progress_label.setStyleSheet("color: #aeb6c2;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                background-color: #1a1f27;
                border: 1px solid #2a313c;
                border-radius: 6px;
                color: #e8ecf1;
                text-align: center;
                min-height: 18px;
            }
            QProgressBar::chunk {
                background-color: #3d8bfd;
                border-radius: 5px;
            }
            """
        )

        self.results_label = QLabel("Stone candidates")
        results_font = QFont()
        results_font.setBold(True)
        self.results_label.setFont(results_font)

        self.stone_table = QTableWidget(0, 8)
        self.stone_table.setHorizontalHeaderLabels(
            [
                "Side",
                "#",
                "Vol (mm³)",
                "Diam (mm)",
                "Mean HU",
                "Max HU",
                "Min HU",
                "Centroid Z",
            ]
        )
        self.stone_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.stone_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.stone_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.stone_table.setAlternatingRowColors(True)
        self.stone_table.verticalHeader().setVisible(False)
        self.stone_table.setMinimumHeight(160)
        self.stone_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.stone_table.setStyleSheet(
            """
            QTableWidget {
                background-color: #1a1f27;
                alternate-background-color: #252b36;
                color: #e8ecf1;
                gridline-color: #3a4352;
                border: 1px solid #2a313c;
                border-radius: 6px;
            }
            QTableWidget::item {
                color: #e8ecf1;
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #c45c26;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #2a313c;
                color: #e8ecf1;
                padding: 6px;
                border: none;
                border-right: 1px solid #1a1f27;
                font-weight: 600;
            }
            """
        )
        self.stone_table.cellClicked.connect(self.on_stone_table_clicked)
        self.clear_stone_table()

        info_panel = QFrame()
        info_panel.setFrameShape(QFrame.Shape.NoFrame)
        info_panel.setStyleSheet(
            """
            QFrame {
                background-color: #12151a;
                border: 1px solid #2a313c;
                border-radius: 10px;
            }
            QLabel { color: #e8ecf1; }
            """
        )
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.setSpacing(8)
        info_layout.addLayout(info_header)
        info_layout.addWidget(self.progress_label)
        info_layout.addWidget(self.progress_bar)
        info_layout.addWidget(self.info_log, stretch=3)
        info_layout.addWidget(self.results_label)
        info_layout.addWidget(self.stone_table, stretch=2)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(viewer_panel)
        splitter.addWidget(info_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([760, 480])

        root = QVBoxLayout()
        root.addLayout(top_row)
        root.addLayout(roi_view_row)
        root.addWidget(splitter, stretch=1)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self.append_info(
            "Ready",
            "CT yükle → Segment kidneys → Extract ROIs → Detect stones.\n"
            "Sağ panel pipeline çıktısını gösterir; taş satırına tıklayınca dilime gider.",
        )

    # --- Info panel helpers ---

    def clear_info_log(self) -> None:
        self.info_log.clear()
        self.set_progress(0, "Idle")

    def set_progress(self, value: int, text: str | None = None) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int(np.clip(value, 0, 100)))
        if text is not None:
            self.progress_label.setText(text)
        QApplication.processEvents()

    def set_progress_busy(self, text: str) -> None:
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_label.setText(text)
        QApplication.processEvents()

    def append_info(self, title: str, body: str = "") -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        block = f"[{stamp}] {title}"
        body = body.strip()
        if body:
            block = f"{block}\n{body}"
        block = f"{block}\n" + ("─" * 42) + "\n"
        self.info_log.moveCursor(QTextCursor.MoveOperation.End)
        self.info_log.insertPlainText(block)
        self.info_log.moveCursor(QTextCursor.MoveOperation.End)

    def append_live_line(self, line: str) -> None:
        self.info_log.moveCursor(QTextCursor.MoveOperation.End)
        self.info_log.insertPlainText(line + "\n")
        self.info_log.moveCursor(QTextCursor.MoveOperation.End)
        QApplication.processEvents()

    def _on_pipeline_stream(self, text: str, is_progress: bool = False) -> None:
        pct, status = _progress_from_line(text)
        if pct is not None:
            self.set_progress(pct, status or text)
        elif status:
            self.progress_label.setText(status)
            QApplication.processEvents()

        # tqdm satırlarını log'a basma; anlamlı satırları canlı yaz
        if is_progress and ("%" in text and "|" in text):
            return
        if text.startswith("If you use this tool please cite"):
            return
        self.append_live_line(text)

    def _run_capturing_output(self, fn, live: bool = True):
        """fn() çalıştırırken stdout/stderr'i yakala (terminale + GUI)."""
        buffer = io.StringIO()
        on_text = self._on_pipeline_stream if live else None
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(old_out, buffer, on_text)  # type: ignore[assignment]
        sys.stderr = _TeeStream(old_err, buffer, on_text)  # type: ignore[assignment]
        try:
            result = fn()
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
        return result, _clean_pipeline_log(buffer.getvalue())

    # --- File / view ---

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
            self.append_info("Load error", str(exc))
            return

        self.kidney_mask = None
        self.kidney_rois = None
        self.stone_results = None
        self.stone_properties = None
        self.view_mode = "full"
        self.clear_stone_table()
        self.path_label.setText(f"Loaded: {path}")
        self.refresh_display_scan()

        vol = self.scan.volume
        sx, sy, sz = self.scan.spacing
        self.append_info(
            "CT loaded",
            f"path: {path.name}\n"
            f"shape (z,y,x): {vol.shape}\n"
            f"spacing mm: ({sx:.4f}, {sy:.4f}, {sz:.4f})\n"
            f"HU min/max: {float(vol.min()):.1f} / {float(vol.max()):.1f}",
        )

    def refresh_display_scan(self) -> None:
        if self.scan is None:
            return

        if self.view_mode in ("left", "right"):
            self.set_view_mode(self.view_mode)
            return

        self.display_scan = self.scan

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

    def on_slice_changed(self, value: int) -> None:
        self.update_slice_view(value)

    def update_slice_view(self, z_index: int) -> None:
        if self.display_scan is None:
            return

        vol = self.display_scan.volume
        z_index = int(np.clip(z_index, 0, vol.shape[0] - 1))
        self.slice_label.setText(f"Slice: {z_index} / {vol.shape[0] - 1}")

        clim: tuple[float, float] | None = (-200.0, 400.0)

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

        stone = None
        if (
            self.view_mode in ("left", "right")
            and self.stone_results
            and self.view_mode in self.stone_results
        ):
            stone = self.stone_results[self.view_mode].mask
            if stone is not None and stone.shape != vol.shape:
                stone = None

        pix = slice_to_qpixmap(
            vol,
            z_index,
            clim=clim,
            kidney_mask=mask,
            stone_mask=stone,
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

        show_orthogonal_slices(
            self.display_scan,
            title=f"CT ({self.view_mode})",
            clim=(-200.0, 400.0),
        )

    # --- Pipeline steps ---

    def run_kidney_segmentation(self) -> None:
        if self.scan is None:
            QMessageBox.information(self, "No data", "Load a scan first.")
            return

        from ai import mask_voxel_counts, segment_kidneys

        self.statusBar().showMessage("Running TotalSegmentator...")
        self.set_progress(2, "Starting TotalSegmentator…")
        self.append_info(
            "Kidney segmentation started",
            "TotalSegmentator fast mode (≈3 mm). First run may download models.\n"
            "Canlı çıktı aşağıda akacak…",
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            def _run():
                return segment_kidneys(self.scan, fast=True, device="gpu")

            result, _captured = self._run_capturing_output(_run, live=True)
            self.kidney_mask = result.mask
            self.kidney_rois = None
            self.stone_results = None
            self.stone_properties = None
            self.clear_stone_table()
            counts = mask_voxel_counts(result.mask)

            self.append_info(
                "Kidney segmentation done",
                f"left voxels: {counts['left']}\n"
                f"right voxels: {counts['right']}",
            )
            self.set_progress(
                100,
                f"Done — left {counts['left']}, right {counts['right']} voxels",
            )
            self.statusBar().showMessage(
                f"Kidneys done — left: {counts['left']}, right: {counts['right']}"
            )

            self.view_mode = "full"
            self.display_scan = self.scan
            self.update_slice_view(self.slice_slider.value())
        except Exception as exc:
            QMessageBox.critical(self, "Segmentation error", str(exc))
            self.append_info("Kidney segmentation failed", str(exc))
            self.set_progress(0, "Failed")
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

        self.set_progress_busy("Extracting kidney ROIs…")
        try:
            self.kidney_rois = extract_kidney_rois(
                self.scan,
                self.kidney_mask,
                margin=25,
            )
            self.stone_results = None
            self.stone_properties = None
            self.clear_stone_table()
        except Exception as exc:
            QMessageBox.critical(self, "ROI error", str(exc))
            self.append_info("ROI error", str(exc))
            self.set_progress(0, "ROI failed")
            return

        if not self.kidney_rois:
            QMessageBox.warning(self, "ROI", "Böbrek ROI bulunamadı.")
            self.append_info("Kidney ROIs", "No ROI found.")
            self.set_progress(0, "No ROI")
            return

        summary = roi_summary(self.kidney_rois)
        detail_lines = [summary, ""]
        for r in self.kidney_rois:
            detail_lines.append(
                f"{r.laterality}: volume {r.volume.shape}, "
                f"mask voxels={int(r.mask.sum())}, origin={r.origin}"
            )

        text = "\n".join(detail_lines)
        self.append_info("Kidney ROIs", text)
        self.set_progress(100, f"ROIs ready ({len(self.kidney_rois)})")
        self.statusBar().showMessage(summary)
        print("=== Kidney ROIs ===")
        print(text)

    def set_view_mode(self, mode: str) -> None:
        self.view_mode = mode

        if mode == "full":
            if self.scan is None:
                return
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

    def clear_stone_table(self) -> None:
        self._stone_row_meta.clear()
        self.stone_table.setRowCount(0)
        self.results_label.setText("Stone analysis — satıra tıkla → dilime git")

    def fill_stone_table(self) -> None:
        self._stone_row_meta.clear()
        self.stone_table.setRowCount(0)
        if not self.stone_properties:
            self.results_label.setText("Stone analysis — henüz yok")
            return

        props = self.stone_properties
        self.stone_table.setRowCount(len(props))
        for row_idx, p in enumerate(props):
            values = [
                p.laterality,
                str(p.stone_id),
                f"{p.volume_mm3:.1f}",
                f"{p.diameter_mm:.1f}",
                f"{p.mean_hu:.0f}",
                f"{p.max_hu:.0f}",
                f"{p.min_hu:.0f}",
                f"{p.centroid_zyx[0]:.1f}",
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setForeground(QBrush(QColor("#e8ecf1")))
                self.stone_table.setItem(row_idx, col, item)
            z_jump = int(round(p.centroid_zyx[0]))
            self._stone_row_meta.append((p.laterality, p.stone_id, z_jump))

        self.results_label.setText(
            f"Stone analysis ({len(props)}) — satıra tıkla → ROI + dilim"
        )

    def on_stone_table_clicked(self, row: int, _column: int) -> None:
        if row < 0 or row >= len(self._stone_row_meta):
            return
        side, comp_id, z_jump = self._stone_row_meta[row]
        if self.view_mode != side:
            self.set_view_mode(side)
        self.slice_slider.setValue(z_jump)
        self.statusBar().showMessage(f"Showing {side} stone #{comp_id} at z={z_jump}")

    def detect_stones(self) -> None:
        if not self.kidney_rois:
            QMessageBox.information(
                self,
                "Need ROIs",
                "Önce Segment kidneys + Extract kidney ROIs.",
            )
            return

        from ai import segment_stones_in_rois
        from postprocessing import analyze_all_rois, format_stone_report

        self.set_progress_busy("Detecting + analyzing stones (HU)…")
        try:
            self.stone_results = segment_stones_in_rois(
                self.kidney_rois,
                hu_threshold=300.0,
                min_voxels=8,
            )
            self.stone_properties = analyze_all_rois(
                self.kidney_rois,
                self.stone_results,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Stone error", str(exc))
            self.append_info("Stone detection failed", str(exc))
            self.set_progress(0, "Stone detection failed")
            return

        summary_lines = [
            f"{side}: {res.n_components} stone(s)"
            for side, res in self.stone_results.items()
        ]
        summary = " | ".join(summary_lines)
        report = format_stone_report(self.stone_properties)
        self.append_info("Stone analysis", summary + "\n\n" + report)
        print(report)

        self.fill_stone_table()
        self.set_progress(
            100,
            f"Analysis done — {len(self.stone_properties or [])} stones",
        )
        self.statusBar().showMessage(summary)

        target_side = (
            self.view_mode
            if self.view_mode in ("left", "right")
            else next(iter(self.stone_results))
        )
        if self.view_mode != target_side:
            self.set_view_mode(target_side)
        res = self.stone_results[target_side]
        zs = np.where(res.binary_mask.any(axis=(1, 2)))[0]
        if len(zs) > 0:
            self.slice_slider.setValue(int(zs[len(zs) // 2]))
        else:
            self.update_slice_view(self.slice_slider.value())

    def _build_meshes(self):
        if self.scan is None or self.kidney_mask is None:
            raise RuntimeError("Önce CT yükle ve Segment kidneys çalıştır.")

        from mesh import build_kidney_meshes, build_stone_meshes

        kidney_meshes = build_kidney_meshes(
            self.kidney_mask,
            self.scan.spacing,
            self.scan.origin,
            self.scan.direction,
        )
        stone_meshes = []
        if self.kidney_rois and self.stone_results:
            stone_meshes = build_stone_meshes(
                self.kidney_rois,
                self.stone_results,
            )
        return kidney_meshes, stone_meshes

    def export_meshes(self) -> None:
        from mesh import export_all

        try:
            kidney_meshes, stone_meshes = self._build_meshes()
        except Exception as exc:
            QMessageBox.critical(self, "Mesh error", str(exc))
            self.append_info("Mesh error", str(exc))
            return

        if not kidney_meshes and not stone_meshes:
            QMessageBox.information(self, "Mesh", "Export edilecek mesh yok.")
            return

        default_dir = Path(__file__).resolve().parent.parent / "outputs" / "meshes"
        default_dir.mkdir(parents=True, exist_ok=True)
        out = QFileDialog.getExistingDirectory(
            self, "STL klasörü seç", str(default_dir)
        )
        if not out:
            return

        self.set_progress_busy("Exporting STL…")
        try:
            paths = export_all(kidney_meshes, stone_meshes, out)
            listing = "\n".join(str(p) for p in paths)
            self.append_info("STL export", listing)
            self.set_progress(100, f"Exported {len(paths)} file(s)")
            self.statusBar().showMessage(f"Exported {len(paths)} STL file(s)")
            QMessageBox.information(
                self,
                "Export OK",
                f"{len(paths)} dosya yazıldı:\n{listing}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))
            self.append_info("Export error", str(exc))
            self.set_progress(0, "Export failed")

    def show_mesh_scene(self) -> None:
        from visualization import show_meshes

        try:
            kidney_meshes, stone_meshes = self._build_meshes()
        except Exception as exc:
            QMessageBox.critical(self, "Mesh error", str(exc))
            self.append_info("Mesh error", str(exc))
            return

        if not kidney_meshes and not stone_meshes:
            QMessageBox.information(self, "Mesh", "Gösterilecek mesh yok.")
            return

        if not stone_meshes:
            self.append_info(
                "3D meshes",
                "Taş mesh'i yok — sadece böbrek gösterilecek. "
                "Taş için Extract ROIs + Detect stones çalıştır.",
            )

        self.set_progress_busy("Building 3D meshes…")
        QApplication.processEvents()
        try:
            show_meshes(kidney_meshes, stone_meshes)
            self.set_progress(
                100,
                f"Meshes shown — kidneys={len(kidney_meshes)}, stones={len(stone_meshes)}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Viewer error", str(exc))
            self.append_info("Mesh viewer error", str(exc))
            self.set_progress(0, "Mesh viewer failed")
