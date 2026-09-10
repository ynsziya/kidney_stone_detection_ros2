from __future__ import annotations

from pathlib import Path

from mesh.export import export_mesh
from mesh.marching_cubes import MeshData, create_mesh_from_mask
from mesh.smoothing import smooth_mesh


def build_kidney_meshes(
    kidney_mask,
    spacing,
    origin,
    direction,
    *,
    smooth_iter: int = 15,
) -> list[tuple[str, MeshData]]:
    out: list[tuple[str, MeshData]] = []
    for label, name in ((1, "left"), (2, "right")):
        binary = (kidney_mask == label).astype("uint8")
        if binary.sum() == 0:
            continue
        mesh = create_mesh_from_mask(binary, spacing, origin, direction)
        mesh = smooth_mesh(mesh, iterations=smooth_iter)
        out.append((name, mesh))
    return out


def build_stone_meshes(
    kidney_rois,
    stone_results,
    *,
    smooth_iter: int = 10,
) -> list[tuple[str, MeshData]]:
    """Her ROI'deki birleşik stone binary'den bir mesh (veya bileşen bazlı)."""
    out: list[tuple[str, MeshData]] = []
    roi_by = {r.laterality: r for r in kidney_rois}
    for side, seg in stone_results.items():
        roi = roi_by.get(side)
        if roi is None or seg.n_components == 0:
            continue
        # Tüm taşlar bir mesh (basit). İstersen stone_id döngüsü yap.
        mesh = create_mesh_from_mask(
            seg.binary_mask,
            roi.spacing,
            roi.origin,
            roi.direction,
        )
        mesh = smooth_mesh(mesh, iterations=smooth_iter)
        out.append((side, mesh))
    return out


def export_all(
    kidney_meshes: list[tuple[str, MeshData]],
    stone_meshes: list[tuple[str, MeshData]],
    out_dir: str | Path,
) -> list[Path]:
    out_dir = Path(out_dir)
    written: list[Path] = []
    for name, mesh in kidney_meshes:
        written.append(export_mesh(mesh, out_dir / f"kidney_{name}.stl"))
    for name, mesh in stone_meshes:
        written.append(export_mesh(mesh, out_dir / f"stone_{name}.stl"))
    return written