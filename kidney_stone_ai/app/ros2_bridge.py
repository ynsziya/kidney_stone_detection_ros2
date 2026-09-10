"""Loose ROS2 bridge: launch / reload via subprocess (no rclpy dependency)."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


def default_mesh_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "outputs" / "meshes"


def find_ros_setup() -> tuple[Path | None, Path | None]:
    """Return (/opt/ros/<distro>/setup.bash, ~/ros2_ws/install/setup.bash)."""
    distro = None
    for name in ("jazzy", "humble", "iron", "rolling"):
        p = Path(f"/opt/ros/{name}/setup.bash")
        if p.is_file():
            distro = p
            break
    ws = Path.home() / "ros2_ws" / "install" / "setup.bash"
    if not ws.is_file():
        ws = None
    return distro, ws


def _bash_with_ros(command: str) -> list[str]:
    distro, ws = find_ros_setup()
    parts: list[str] = ["set -e"]
    if distro is not None:
        parts.append(f'source "{distro}"')
    if ws is not None:
        parts.append(f'source "{ws}"')
    parts.append(command)
    return ["bash", "-lc", " && ".join(parts)]


@dataclass
class Ros2Bridge:
    """Tracks launched RViz / Gazebo processes; reloads if already running."""

    _procs: dict[str, subprocess.Popen] = field(default_factory=dict)

    def ros_available(self) -> bool:
        distro, ws = find_ros_setup()
        return distro is not None and ws is not None and shutil.which("bash") is not None

    def _alive(self, key: str) -> bool:
        proc = self._procs.get(key)
        return proc is not None and proc.poll() is None

    def _try_reload(self, service: str) -> bool:
        cmd = (
            f"ros2 service call {service} std_srvs/srv/Trigger '{{}}'"
        )
        try:
            result = subprocess.run(
                _bash_with_ros(cmd),
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            return result.returncode == 0 and "success=True" in (result.stdout or "")
        except (subprocess.TimeoutExpired, OSError):
            return False

    def open_rviz(self, mesh_dir: Path) -> str:
        mesh_dir = mesh_dir.resolve()
        if self._alive("rviz"):
            if self._try_reload("/mesh_marker_publisher/reload"):
                return f"RViz zaten açık — mesh yenilendi ({mesh_dir})"
            return (
                f"RViz süreci çalışıyor (pid={self._procs['rviz'].pid}); "
                "henüz hazır değilse birkaç saniye sonra tekrar bas.\n"
                f"{mesh_dir}"
            )

        launch = (
            "ros2 launch kidney_stone_viz_ros2 view_meshes.launch.py "
            f'mesh_dir:="{mesh_dir}"'
        )
        self._procs["rviz"] = subprocess.Popen(
            _bash_with_ros(launch),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
        )
        return f"RViz launch başlatıldı (pid={self._procs['rviz'].pid})\n{mesh_dir}"

    def open_gazebo(self, mesh_dir: Path) -> str:
        mesh_dir = mesh_dir.resolve()
        if self._alive("gazebo"):
            if self._try_reload("/gazebo_mesh_spawner/reload"):
                return f"Gazebo zaten açık — mesh yenilendi ({mesh_dir})"
            return (
                f"Gazebo süreci çalışıyor (pid={self._procs['gazebo'].pid}); "
                "henüz hazır değilse birkaç saniye sonra tekrar bas.\n"
                f"{mesh_dir}"
            )

        launch = (
            "ros2 launch kidney_stone_viz_ros2 view_gazebo.launch.py "
            f'mesh_dir:="{mesh_dir}"'
        )
        self._procs["gazebo"] = subprocess.Popen(
            _bash_with_ros(launch),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
        )
        return f"Gazebo launch başlatıldı (pid={self._procs['gazebo'].pid})\n{mesh_dir}"
