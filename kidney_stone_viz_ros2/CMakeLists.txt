cmake_minimum_required(VERSION 3.8)
project(kidney_stone_viz_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_executable(mesh_marker_publisher src/mesh_marker_publisher.cpp)
ament_target_dependencies(
  mesh_marker_publisher
  rclcpp
  std_msgs
  geometry_msgs
  visualization_msgs
  std_srvs
)

add_executable(gazebo_mesh_spawner src/gazebo_mesh_spawner.cpp)
ament_target_dependencies(
  gazebo_mesh_spawner
  rclcpp
  std_srvs
)

install(TARGETS mesh_marker_publisher gazebo_mesh_spawner
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch rviz worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
