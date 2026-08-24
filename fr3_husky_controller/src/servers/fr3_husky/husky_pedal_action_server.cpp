#include <fr3_husky_controller/servers/fr3_husky/husky_pedal_action_server.hpp>

#include <algorithm>
#include <stdexcept>

namespace fr3_husky_controller::servers::fr3_husky
{

namespace
{
FR3HuskyModelUpdater& getFR3HuskyModelUpdater(ModelUpdaterBase& model_updater, const std::string& server_name)
{
    auto* fr3_husky_model_updater = dynamic_cast<FR3HuskyModelUpdater*>(&model_updater);
    if (!fr3_husky_model_updater)
    {
        throw std::runtime_error("[" + server_name + "] requires FR3HuskyModelUpdater");
    }
    return *fr3_husky_model_updater;
}

double clamp(double value, double lo, double hi)
{
    return std::max(lo, std::min(hi, value));
}

}  // namespace

HuskyPedal::HuskyPedal(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_husky_model_updater_(getFR3HuskyModelUpdater(model_updater, name))
{
    mode_ = ServerMode::CONTROLLER;
    cmd_vel_.setZero();
    q_hold_.setZero(model_updater.manipulator_dof_);

    // 중복 선언을 방지하기 위해 이미 선언된 파라미터가 있으면 가져오고, 없으면 선언하는 안전한 함수 정의
    auto safe_get_int = [this](const std::string& param_name, int default_val) -> int {
        if (node_->has_parameter(param_name)) {
            return static_cast<int>(node_->get_parameter(param_name).as_int());
        }
        return static_cast<int>(node_->declare_parameter<int>(param_name, default_val));
    };

    auto safe_get_double = [this](const std::string& param_name, double default_val) -> double {
        if (node_->has_parameter(param_name)) {
            return node_->get_parameter(param_name).as_double();
        }
        return node_->declare_parameter<double>(param_name, default_val);
    };

    auto safe_get_string = [this](const std::string& param_name, const std::string& default_val) -> std::string {
        if (node_->has_parameter(param_name)) {
            return node_->get_parameter(param_name).as_string();
        }
        return node_->declare_parameter<std::string>(param_name, default_val);
    };

    // 안전하게 파라미터 값 할당 (중복 선언 에러 완벽 해결)
    axis_left_        = safe_get_int(name_ + ".axis_left", 0);
    axis_right_       = safe_get_int(name_ + ".axis_right", 1);
    axis_yaw_mag_     = safe_get_int(name_ + ".axis_yaw_mag", 2);
    scale_linear_     = safe_get_double(name_ + ".scale_linear", 0.5);
    scale_angular_    = safe_get_double(name_ + ".scale_angular", 0.5);
    deadzone_pedal_   = safe_get_double(name_ + ".deadzone_pedal", 0.05);
    deadzone_yaw_mag_ = safe_get_double(name_ + ".deadzone_yaw_mag", 0.05);
    enable_button_    = safe_get_int(name_ + ".enable_button", -1);
    pedal_topic_      = safe_get_string(name_ + ".pedal_topic", "joy");


    RCLCPP_INFO(node_->get_logger(), "[%s] HuskyPedal created (topic: %s)", name_.c_str(), pedal_topic_.c_str());
}

bool HuskyPedal::acceptGoal(const ActionT::Goal& goal)
{
    (void)goal;
    return true;
}

void HuskyPedal::onGoalAccepted(const ActionT::Goal& goal)
{
    const bool latch_hold = goal.enable && !enabled_;
    enabled_ = goal.enable;
    goal_update_pending_ = true;
    applyEnabledState(latch_hold);
}

void HuskyPedal::onStart()
{
    applyEnabledState(enabled_);
}

HuskyPedal::ComputeResult HuskyPedal::compute(const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/)
{
    if (goal_update_pending_)
    {
        goal_update_pending_ = false;
        return ComputeResult::SUCCEEDED;
    }

    if (!enabled_)
    {
        resetCommand();
        fr3_husky_model_updater_.wheel_vel_desired_.setZero();
        return ComputeResult::RUNNING;
    }

    Eigen::Vector3d cmd_vel;
    {
        std::lock_guard<std::mutex> lock(cmd_vel_mutex_);
        cmd_vel = cmd_vel_;
    }

    fr3_husky_model_updater_.wheel_vel_desired_ =
        fr3_husky_model_updater_.robot_controller_->MobileVelocityCommand(cmd_vel);

    fr3_husky_model_updater_.writeCommand(q_hold_, fr3_husky_model_updater_.wheel_vel_desired_);
    return ComputeResult::RUNNING;
}

void HuskyPedal::onStop(StopReason reason)
{
    if (!enabled_ || reason == StopReason::CANCELED || reason == StopReason::ABORTED)
    {
        pedal_sub_.reset();
        resetCommand();
        fr3_husky_model_updater_.wheel_vel_desired_.setZero();
        fr3_husky_model_updater_.haltCommands();
    }

    const char* reason_str = "none";
    if (reason == StopReason::CANCELED)
    {
        reason_str = "canceled";
    }
    else if (reason == StopReason::SUCCEEDED)
    {
        reason_str = "succeeded";
    }
    else if (reason == StopReason::ABORTED)
    {
        reason_str = "aborted";
    }

    RCLCPP_INFO(node_->get_logger(), "[%s] stopped (%s)", name_.c_str(), reason_str);
}

HuskyPedal::ResultPtr HuskyPedal::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();
    result->is_completed = (reason != StopReason::ABORTED);
    return result;
}

void HuskyPedal::subPedalCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    if (!msg)
    {
        return;
    }

    const Eigen::Vector3d cmd_vel = computeBaseVelocityFromPedal(*msg);
    std::lock_guard<std::mutex> lock(cmd_vel_mutex_);
    cmd_vel_ = cmd_vel;
}

void HuskyPedal::applyEnabledState(bool latch_hold)
{
    if (enabled_)
    {
        if (latch_hold)
        {
            resetCommand();
            q_hold_ = fr3_husky_model_updater_.q_total_;
            warned_axes_ = false;
            warned_buttons_ = false;
        }
        if (!pedal_sub_)
        {
            pedal_sub_ = node_->create_subscription<sensor_msgs::msg::Joy>(
                pedal_topic_, rclcpp::SensorDataQoS(),
                std::bind(&HuskyPedal::subPedalCallback, this, std::placeholders::_1));
        }
        RCLCPP_INFO(node_->get_logger(), "[%s] %s pedal control",
                    name_.c_str(),
                    latch_hold ? "started" : "kept");
        return;
    }

    pedal_sub_.reset();
    resetCommand();
    fr3_husky_model_updater_.wheel_vel_desired_.setZero();
    RCLCPP_INFO(node_->get_logger(), "[%s] pedal control disable requested", name_.c_str());
}

Eigen::Vector3d HuskyPedal::computeBaseVelocityFromPedal(const sensor_msgs::msg::Joy& msg)
{
    const int max_axis = std::max({axis_left_, axis_right_, axis_yaw_mag_});
    if (max_axis < 0 || static_cast<int>(msg.axes.size()) <= max_axis)
    {
        if (!warned_axes_)
        {
            RCLCPP_ERROR(node_->get_logger(),
                         "[%s] Pedal axes too short: need index %d, got %zu",
                         name_.c_str(), max_axis, msg.axes.size());
            warned_axes_ = true;
        }
        return Eigen::Vector3d::Zero();
    }

    if (enable_button_ >= 0)
    {
        if (static_cast<int>(msg.buttons.size()) <= enable_button_)
        {
            if (!warned_buttons_)
            {
                RCLCPP_ERROR(node_->get_logger(),
                             "[%s] Pedal buttons too short: need index %d, got %zu",
                             name_.c_str(), enable_button_, msg.buttons.size());
                warned_buttons_ = true;
            }
            return Eigen::Vector3d::Zero();
        }
        if (msg.buttons[enable_button_] == 0)
        {
            return Eigen::Vector3d::Zero();
        }
    }

    const double raw_left = static_cast<double>(msg.axes[axis_left_]);
    const double raw_right = static_cast<double>(msg.axes[axis_right_]);
    const double raw_yaw = static_cast<double>(msg.axes[axis_yaw_mag_]);

    double left = (raw_left + 1.0) * 0.25;
    double right = (raw_right + 1.0) * 0.25;
    double yaw_mag = (raw_yaw + 1.0) * 0.25;

    if (left < deadzone_pedal_) left = 0.0;
    if (right < deadzone_pedal_) right = 0.0;
    if (yaw_mag < deadzone_yaw_mag_) yaw_mag = 0.0;

    const bool turning = yaw_mag >= deadzone_yaw_mag_;
    const bool turning_right = turning && right >= deadzone_pedal_;

    double linear = 0.0;
    if (turning)
    {
        linear = scale_linear_ * -left;
    }
    else
    {
        linear = scale_linear_ * (right - left);
    }
    linear = clamp(linear, -1.0, 1.0);

    double angular = 0.0;
    if (turning)
    {
        const double yaw_sign = turning_right ? -1.0 : 1.0;
        angular = scale_angular_ * clamp(yaw_mag, 0.0, 1.0) * yaw_sign;
        angular = clamp(angular, -1.0, 1.0);
    }

    return Eigen::Vector3d(linear, 0.0, angular);
}

void HuskyPedal::resetCommand()
{
    std::lock_guard<std::mutex> lock(cmd_vel_mutex_);
    cmd_vel_.setZero();
}

REGISTER_FR3_HUSKY_ACTION_SERVER(HuskyPedal, "fr3_husky_pedal")

}  // namespace fr3_husky_controller::servers::fr3_husky
