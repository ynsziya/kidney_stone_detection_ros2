from __future__ import annotations

import numpy as np
import pyvista as pv

from preprocessing.scan_data import ScanData


def scan_to_grid(scan: ScanData, scalars_name: str = "values") -> pv.ImageData:
    """
    ScanData (z, y, x) → PyVista ImageData (x, y, z noktaları).

    spacing/origin ScanData'dan gelir; slice'lar mm cinsinden doğru ölçeklenir.
    """
    volume = np.asarray(scan.volume, dtype=np.float32)
    nz, ny, nx = volume.shape
    sx, sy, sz = scan.spacing

    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)  # nokta sayısı: x, y, z
    grid.spacing = (sx, sy, sz)
    grid.origin = scan.origin

    # (z,y,x) → (x,y,z), sonra Fortran flatten (PyVista beklediği düzen)
    vol_xyz = np.transpose(volume, (2, 1, 0))
    grid.point_data[scalars_name] = vol_xyz.flatten(order="F")
    grid.set_active_scalars(scalars_name)
    return grid


def show_orthogonal_slices(
    scan: ScanData,
    title: str = "CT Viewer",
    clim: tuple[float, float] | None = None,
) -> None:
    """
    Merkezde axial / coronal / sagittal üç dilim gösterir.

    clim:
      - RAW HU için örn. (-200, 400) yumuşak doku penceresi
      - normalize [0,1] için None bırak (otomatik)
    """
    grid = scan_to_grid(scan)
    nx, ny, nz = grid.dimensions
    sx, sy, sz = grid.spacing
    ox, oy, oz = grid.origin

    # Merkez dünya koordinatı (mm)
    cx = ox + (nx - 1) * sx * 0.5
    cy = oy + (ny - 1) * sy * 0.5
    cz = oz + (nz - 1) * sz * 0.5

    slices = grid.slice_orthogonal(x=cx, y=cy, z=cz)

    plotter = pv.Plotter()
    plotter.add_title(title, font_size=12)
    mesh_kwargs = {
        "cmap": "gray",
        "show_scalar_bar": True,
        "scalar_bar_args": {"title": "intensity"},
    }
    if clim is not None:
        mesh_kwargs["clim"] = clim

    plotter.add_mesh(slices, **mesh_kwargs)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()


def show_volume(
    scan: ScanData,
    title: str = "CT Volume",
    clim: tuple[float, float] | None = None,
) -> None:
    """
    Basit 3D volume rendering (GPU ister; yavaş olabilir).
    İlk denemede orthogonal slices yeter; bu opsiyonel.
    """
    grid = scan_to_grid(scan)
    plotter = pv.Plotter()
    plotter.add_title(title, font_size=12)
    plotter.add_volume(
        grid,
        cmap="gray",
        opacity="sigmoid",
        clim=clim,
        show_scalar_bar=True,
    )
    plotter.add_axes()
    plotter.show()