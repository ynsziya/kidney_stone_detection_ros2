#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace fs = std::filesystem;

namespace
{

std::string to_lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string sanitize(std::string name)
{
  for (char & c : name) {
    if (!std::isalnum(static_cast<unsigned char>(c))) {
      c = '_';
    }
  }
  return name;
}

struct Rgba
{
  float r{0.8f};
  float g{0.8f};
  float b{0.8f};
  float a{0.8f};
};

Rgba color_for_name(const std::string & name)
{
  const std::string lower = to_lower(name);
  if (lower.find("kidney_left") != std::string::npos) {
    return {1.0f, 0.55f, 0.55f, 0.45f};
  }
  if (lower.find("kidney_right") != std::string::npos) {
    return {0.55f, 0.75f, 1.0f, 0.45f};
  }
  if (lower.find("stone") != std::string::npos) {
    return {1.0f, 0.85f, 0.2f, 1.0f};
  }
  return {};
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

std::string shell_quote(const std::string & s)
{
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

}  // namespace

class GazeboMeshSpawner : public rclcpp::Node
{
public:
  GazeboMeshSpawner()
  : Node("gazebo_mesh_spawner")
  {
    declare_parameter<std::string>("mesh_dir", "");
    declare_parameter<std::string>("output_dir", "/tmp/kidney_stone_viz_gazebo");
    declare_parameter<std::string>("world_name", "empty");
    declare_parameter<std::string>("model_name", "kidney_stone_meshes");
    declare_parameter<double>("mesh_scale", 0.001);
    declare_parameter<bool>("auto_spawn", true);
    declare_parameter<double>("spawn_delay_sec", 2.0);

    reload_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/reload",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> res)
      {
        const bool ok = build_and_spawn();
        res->success = ok;
        res->message = ok ? "spawned" : "failed";
      });

    const double delay = param_as_double(*this, "spawn_delay_sec");
    startup_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(std::max(0.0, delay))),
      [this]() {
        startup_timer_->cancel();
        build_and_spawn();
        RCLCPP_INFO(get_logger(), "gazebo_mesh_spawner ready");
      });
  }

private:
  bool build_and_spawn()
  {
    const std::string mesh_dir = get_parameter("mesh_dir").as_string();
    const std::string output_dir = get_parameter("output_dir").as_string();
    const std::string model_name = sanitize(get_parameter("model_name").as_string());
    const double scale = param_as_double(*this, "mesh_scale");

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
      if (to_lower(entry.path().extension().string()) == ".stl") {
        stls.push_back(entry.path());
      }
    }
    std::sort(stls.begin(), stls.end());

    if (stls.empty()) {
      RCLCPP_WARN(get_logger(), "No STL in %s", directory.c_str());
      return false;
    }

    const fs::path model_dir = fs::path(output_dir) / model_name;
    std::error_code ec;
    fs::create_directories(model_dir, ec);
    if (ec) {
      RCLCPP_ERROR(get_logger(), "Cannot create %s: %s", model_dir.c_str(), ec.message().c_str());
      return false;
    }

    const fs::path sdf_path = model_dir / "model.sdf";
    const fs::path config_path = model_dir / "model.config";

    if (!write_sdf(sdf_path, model_name, stls, scale)) {
      return false;
    }
    write_model_config(config_path, model_name);
    RCLCPP_INFO(get_logger(), "SDF written: %s (%zu meshes)", sdf_path.c_str(), stls.size());

    if (!get_parameter("auto_spawn").as_bool()) {
      return true;
    }

    remove_model(model_name);
    return spawn_model(sdf_path, model_name);
  }

  bool write_sdf(
    const fs::path & path, const std::string & model_name,
    const std::vector<fs::path> & stls, double scale) const
  {
    std::ostringstream sdf;
    sdf << "<?xml version=\"1.0\"?>\n"
        << "<sdf version=\"1.9\">\n"
        << "  <model name=\"" << model_name << "\">\n"
        << "    <static>true</static>\n";

    for (const auto & stl : stls) {
      const std::string link = sanitize(stl.stem().string());
      const auto c = color_for_name(stl.stem().string());
      const float transparency = 1.0f - c.a;
      const fs::path abs = fs::weakly_canonical(stl);
      const std::string uri = std::string("file://") + abs.string();

      sdf << "    <link name=\"" << link << "\">\n"
          << "      <visual name=\"" << link << "_visual\">\n"
          << "        <geometry>\n"
          << "          <mesh>\n"
          << "            <uri>" << uri << "</uri>\n"
          << "            <scale>" << scale << " " << scale << " " << scale << "</scale>\n"
          << "          </mesh>\n"
          << "        </geometry>\n"
          << "        <transparency>" << transparency << "</transparency>\n"
          << "        <material>\n"
          << "          <ambient>" << c.r << " " << c.g << " " << c.b << " 1</ambient>\n"
          << "          <diffuse>" << c.r << " " << c.g << " " << c.b << " 1</diffuse>\n"
          << "        </material>\n"
          << "      </visual>\n"
          << "    </link>\n";
    }

    sdf << "  </model>\n</sdf>\n";

    std::ofstream out(path);
    if (!out) {
      RCLCPP_ERROR(get_logger(), "Failed to write %s", path.c_str());
      return false;
    }
    out << sdf.str();
    return true;
  }

  void write_model_config(const fs::path & path, const std::string & model_name) const
  {
    std::ofstream out(path);
    out << "<?xml version=\"1.0\"?>\n"
        << "<model>\n"
        << "  <name>" << model_name << "</name>\n"
        << "  <version>1.0</version>\n"
        << "  <sdf version=\"1.9\">model.sdf</sdf>\n"
        << "  <description>Kidney / stone meshes for Gazebo visualization.</description>\n"
        << "</model>\n";
  }

  void remove_model(const std::string & model_name) const
  {
    const std::string world = get_parameter("world_name").as_string();
    const std::string cmd =
      "ros2 run ros_gz_sim remove -world " + shell_quote(world) +
      " -name " + shell_quote(model_name) + " >/dev/null 2>&1";
    std::system(cmd.c_str());
  }

  bool spawn_model(const fs::path & sdf_path, const std::string & model_name) const
  {
    const std::string world = get_parameter("world_name").as_string();
    const std::string cmd =
      "ros2 run ros_gz_sim create -world " + shell_quote(world) +
      " -file " + shell_quote(sdf_path.string()) +
      " -name " + shell_quote(model_name) +
      " -x 0 -y 0 -z 0";

    RCLCPP_INFO(get_logger(), "Spawning into gz-sim: %s", model_name.c_str());
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
      RCLCPP_ERROR(get_logger(), "ros_gz_sim create failed (code %d)", rc);
      return false;
    }
    RCLCPP_INFO(get_logger(), "Spawn OK: %s", model_name.c_str());
    return true;
  }

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reload_srv_;
  rclcpp::TimerBase::SharedPtr startup_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GazeboMeshSpawner>());
  rclcpp::shutdown();
  return 0;
}
