#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace fs = std::filesystem;

namespace
{

std::string to_lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std_msgs::msg::ColorRGBA color_for_name(const std::string & name)
{
  const std::string lower = to_lower(name);
  float r = 0.8f, g = 0.8f, b = 0.8f, a = 0.8f;

  if (lower.find("kidney_left") != std::string::npos) {
    r = 1.0f; g = 0.55f; b = 0.55f; a = 0.45f;
  } else if (lower.find("kidney_right") != std::string::npos) {
    r = 0.55f; g = 0.75f; b = 1.0f; a = 0.45f;
  } else if (lower.find("stone") != std::string::npos) {
    r = 1.0f; g = 0.85f; b = 0.2f; a = 1.0f;
  }

  std_msgs::msg::ColorRGBA c;
  c.r = r;
  c.g = g;
  c.b = b;
  c.a = a;
  return c;
}

std::string path_to_file_uri(const fs::path & path)
{
  // Absolute path → file:///...
  const fs::path abs = fs::weakly_canonical(path);
  return std::string("file://") + abs.string();
}

double param_as_double(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter p = node.get_parameter(name);
  if (p.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
    return p.as_double();
  }
  if (p.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
    return static_cast<double>(p.as_int());
  }
  return std::stod(p.as_string());
}

}  // namespace

class MeshMarkerPublisher : public rclcpp::Node
{
public:
  MeshMarkerPublisher()
  : Node("mesh_marker_publisher")
  {
    declare_parameter<std::string>("mesh_dir", "");
    declare_parameter<std::string>("frame_id", "map");
    declare_parameter<double>("publish_rate_hz", 1.0);
    declare_parameter<double>("mesh_scale", 0.001);

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/kidney_stone/markers", 10);

    reload_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/reload",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> res)
      {
        const bool ok = reload_meshes();
        res->success = ok;
        size_t count = 0;
        for (const auto & m : markers_.markers) {
          if (m.action != visualization_msgs::msg::Marker::DELETEALL) {
            ++count;
          }
        }
        res->message = "markers=" + std::to_string(count);
      });

    double rate = param_as_double(*this, "publish_rate_hz");
    if (rate < 0.1) {
      rate = 0.1;
    }
    const auto period = std::chrono::duration<double>(1.0 / rate);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      [this]() { tick(); });

    reload_meshes();
    RCLCPP_INFO(get_logger(), "mesh_marker_publisher ready");
  }

private:
  bool reload_meshes()
  {
    const std::string mesh_dir = get_parameter("mesh_dir").as_string();
    const std::string frame_id = get_parameter("frame_id").as_string();
    const double scale = param_as_double(*this, "mesh_scale");

    markers_.markers.clear();

    if (mesh_dir.empty()) {
      RCLCPP_WARN(get_logger(), "mesh_dir empty — set mesh_dir:=/path/to/stls");
      return false;
    }

    const fs::path directory = fs::absolute(fs::path(mesh_dir));
    if (!fs::is_directory(directory)) {
      RCLCPP_ERROR(get_logger(), "mesh_dir not found: %s", directory.c_str());
      return false;
    }

    std::vector<fs::path> stls;
    for (const auto & entry : fs::directory_iterator(directory)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      const auto ext = entry.path().extension().string();
      if (to_lower(ext) == ".stl") {
        stls.push_back(entry.path());
      }
    }
    std::sort(stls.begin(), stls.end());

    if (stls.empty()) {
      RCLCPP_WARN(get_logger(), "No STL in %s", directory.c_str());
      return false;
    }

    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    markers_.markers.push_back(clear);

    int idx = 0;
    for (const auto & stl : stls) {
      const std::string uri = path_to_file_uri(stl);

      visualization_msgs::msg::Marker m;
      m.header.frame_id = frame_id;
      m.ns = "kidney_stone_meshes";
      m.id = idx++;
      m.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.mesh_resource = uri;
      m.mesh_use_embedded_materials = false;
      m.pose.orientation.w = 1.0;
      m.scale.x = scale;
      m.scale.y = scale;
      m.scale.z = scale;
      m.color = color_for_name(stl.stem());
      markers_.markers.push_back(m);

      RCLCPP_INFO(get_logger(), "Loaded %s → %s", stl.filename().c_str(), uri.c_str());
    }

    return true;
  }

  void tick()
  {
    const auto now = get_clock()->now();
    for (auto & m : markers_.markers) {
      if (m.action != visualization_msgs::msg::Marker::DELETEALL) {
        m.header.stamp = now;
      }
    }
    marker_pub_->publish(markers_);
  }

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reload_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
  visualization_msgs::msg::MarkerArray markers_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MeshMarkerPublisher>());
  rclcpp::shutdown();
  return 0;
}
