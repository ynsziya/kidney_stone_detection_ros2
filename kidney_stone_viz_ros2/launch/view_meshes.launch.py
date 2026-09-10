from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    mesh_dir = LaunchConfiguration("mesh_dir")
    frame_id = LaunchConfiguration("frame_id")
    mesh_scale = LaunchConfiguration("mesh_scale")
    use_rviz = LaunchConfiguration("use_rviz")

    rviz_config = PathJoinSubstitution(
        [FindPackageShare("kidney_stone_viz_ros2"), "rviz", "meshes.rviz"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mesh_dir",
                default_value="",
                description="Directory containing kidney_*.stl / stone_*.stl",
            ),
            DeclareLaunchArgument("frame_id", default_value="map"),
            DeclareLaunchArgument(
                "mesh_scale",
                default_value="0.001",
                description="STL unit scale (0.001 if mesh is in mm)",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Also start RViz2 with meshes.rviz",
            ),
            Node(
                package="kidney_stone_viz_ros2",
                executable="mesh_marker_publisher",
                name="mesh_marker_publisher",
                output="screen",
                parameters=[
                    {
                        "mesh_dir": mesh_dir,
                        "frame_id": frame_id,
                        "mesh_scale": mesh_scale,
                        "publish_rate_hz": 1.0,
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(use_rviz),
            ),
        ]
    )
