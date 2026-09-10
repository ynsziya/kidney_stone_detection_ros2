import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("kidney_stone_viz_ros2")
    world_path = os.path.join(pkg_share, "worlds", "empty.sdf")

    mesh_dir = LaunchConfiguration("mesh_dir")
    mesh_scale = LaunchConfiguration("mesh_scale")
    world_name = LaunchConfiguration("world_name")
    model_name = LaunchConfiguration("model_name")
    spawn_delay_sec = LaunchConfiguration("spawn_delay_sec")

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={
            "gz_args": f"-r {world_path}",
        }.items(),
    )

    spawner = Node(
        package="kidney_stone_viz_ros2",
        executable="gazebo_mesh_spawner",
        name="gazebo_mesh_spawner",
        output="screen",
        parameters=[
            {
                "mesh_dir": mesh_dir,
                "output_dir": "/tmp/kidney_stone_viz_gazebo",
                "world_name": world_name,
                "model_name": model_name,
                "mesh_scale": mesh_scale,
                "auto_spawn": True,
                "spawn_delay_sec": spawn_delay_sec,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mesh_dir",
                default_value="",
                description="Directory containing kidney_*.stl / stone_*.stl",
            ),
            DeclareLaunchArgument(
                "mesh_scale",
                default_value="0.001",
                description="STL unit scale (0.001 if mesh is in mm)",
            ),
            DeclareLaunchArgument("world_name", default_value="empty"),
            DeclareLaunchArgument(
                "model_name", default_value="kidney_stone_meshes"
            ),
            DeclareLaunchArgument(
                "spawn_delay_sec",
                default_value="3.0",
                description="Wait for gz-sim before spawning",
            ),
            gz_sim,
            # Extra cushion so UserCommands is up before create runs
            TimerAction(period=0.5, actions=[spawner]),
        ]
    )
