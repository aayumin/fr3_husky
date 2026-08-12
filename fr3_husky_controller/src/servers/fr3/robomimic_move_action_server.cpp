#include <fr3_husky_controller/servers/fr3_husky/robomimic_move_action_server.hpp>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <algorithm>

namespace fr3_husky_controller::servers::fr3
{

namespace
{

FR3ModelUpdater& getFR3ModelUpdater(
    ModelUpdaterBase& model_updater,
    const std::string& server_name)
{
    auto* p = dynamic_cast<FR3ModelUpdater*>(&model_updater);

    if (!p)
        throw std::runtime_error("[" + server_name + "] requires FR3ModelUpdater");

    return *p;
}


geometry_msgs::msg::PoseStamped affineToPoseStamped(
    const Eigen::Affine3d& T,
    const rclcpp::Time& stamp)
{
    geometry_msgs::msg::PoseStamped msg;

    msg.header.stamp = stamp;
    msg.header.frame_id = "base_link";

    msg.pose.position.x = T.translation().x();
    msg.pose.position.y = T.translation().y();
    msg.pose.position.z = T.translation().z();

    Eigen::Quaterniond q(T.linear());
    q.normalize();

    msg.pose.orientation.x = q.x();
    msg.pose.orientation.y = q.y();
    msg.pose.orientation.z = q.z();
    msg.pose.orientation.w = q.w();

    return msg;
}

}  // namespace


RobomimicMove::RobomimicMove(
    const std::string& name,
    const NodePtr& node,
    ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_model_updater_(getFR3ModelUpdater(model_updater, name))
{
    mode_ = ServerMode::TASK;

    left_arm_.robot_name = "left";
    left_arm_.controller_ee_name = "left_fr3_hand_tcp";

    right_arm_.robot_name = "right";
    right_arm_.controller_ee_name = "right_fr3_hand_tcp";

    delta_action_sub_ = node_->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/robomimic/absolute_action",
        rclcpp::QoS(1),
        std::bind(&RobomimicMove::onDeltaAction, this, std::placeholders::_1));

    auto active_qos = rclcpp::QoS(1).reliable().transient_local();

    inference_active_pub_ = node_->create_publisher<std_msgs::msg::Bool>(
        "/robomimic/inference_active",
        active_qos);

    left_eef_pose_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/robomimic/obs/left_eef_pose",
        10);

    right_eef_pose_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/robomimic/obs/right_eef_pose",
        10);

    publishInferenceActive(false);

    RCLCPP_INFO(node_->get_logger(), "[%s] RobomimicMove created", name_.c_str());
}


bool RobomimicMove::acceptGoal(const ActionT::Goal& goal)
{
    if (!fr3_model_updater_.HasEffortCommandInterface())
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] Reject: effort command interface is required",
            name_.c_str());

        return false;
    }

    if (goal.arm != "left" && goal.arm != "right" && goal.arm != "dual")
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] Reject: arm must be left, right, or dual",
            name_.c_str());

        return false;
    }

    if (goal.mode < 0 || goal.mode > 3)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: mode must be 0-3", name_.c_str());
        return false;
    }

    if ((goal.arm == "left" || goal.arm == "dual") &&
        !fr3_model_updater_.robot_data_->hasLinkFrame(left_arm_.controller_ee_name))
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] Reject: unknown EE name [%s]",
            name_.c_str(),
            left_arm_.controller_ee_name.c_str());

        return false;
    }

    if ((goal.arm == "right" || goal.arm == "dual") &&
        !fr3_model_updater_.robot_data_->hasLinkFrame(right_arm_.controller_ee_name))
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] Reject: unknown EE name [%s]",
            name_.c_str(),
            right_arm_.controller_ee_name.c_str());

        return false;
    }

    return true;
}


void RobomimicMove::onGoalAccepted(const ActionT::Goal& goal)
{
    if (goal.arm == "left")
        arm_mode_ = ArmMode::LEFT;
    else if (goal.arm == "right")
        arm_mode_ = ArmMode::RIGHT;
    else
        arm_mode_ = ArmMode::DUAL;

    control_mode_ = goal.mode;

    position_scale_ = goal.position_scale > 0.0 ? goal.position_scale : 1.0;
    rotation_scale_ = goal.rotation_scale > 0.0 ? goal.rotation_scale : 1.0;
    command_timeout_ = goal.command_timeout > 0.0 ? goal.command_timeout : 1.0;

    requestActivate();
}


void RobomimicMove::onStart()
{
    ee_data_.clear();

    auto initialize_arm = [this](ArmState& arm)
    {
        auto& ee_data = ee_data_[arm.controller_ee_name];
        ee_data = drc::TaskSpaceData::Zero();

        ee_data.x = fr3_model_updater_.robot_data_->getPose(arm.controller_ee_name);
        ee_data.xdot = fr3_model_updater_.robot_data_->getVelocity(arm.controller_ee_name);
        ee_data.xddot.setZero();

        ee_data.setInit();
        ee_data.setDesired();
        ee_data.xdot_desired.setZero();

        arm.x_goal = ee_data.x;
        arm.x_target = ee_data.x;
    };

    initialize_arm(left_arm_);
    initialize_arm(right_arm_);

    {
        std::lock_guard<std::mutex> lock(command_mutex_);

        latest_action_.clear();
        has_command_ = false;
        has_new_command_ = false;
        last_command_time_ = 0.0;
    }

    step_count_ = 0;

    publishInferenceActive(true);

    const char* arm_name = "dual";

    if (arm_mode_ == ArmMode::LEFT)
        arm_name = "left";
    else if (arm_mode_ == ArmMode::RIGHT)
        arm_name = "right";

    RCLCPP_INFO(
        node_->get_logger(),
        "[%s] started: arm=%s, mode=%d",
        name_.c_str(),
        arm_name,
        control_mode_);
}


void RobomimicMove::onDeltaAction(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (!msg)
        return;

    const std::size_t expected_dim = arm_mode_ == ArmMode::DUAL ? 14 : 7;

    if (msg->data.size() != expected_dim)
    {
        RCLCPP_WARN_THROTTLE(
            node_->get_logger(),
            *node_->get_clock(),
            1000,
            "[%s] Invalid absolute action dimension: received=%zu, expected=%zu",
            name_.c_str(),
            msg->data.size(),
            expected_dim);

        return;
    }

    for (const double value : msg->data)
    {
        if (!std::isfinite(value))
        {
            RCLCPP_WARN(node_->get_logger(), "[%s] Received NaN / Inf action", name_.c_str());
            return;
        }
    }

    std::lock_guard<std::mutex> lock(command_mutex_);

    latest_action_ = msg->data;
    has_command_ = true;
    has_new_command_ = true;
    last_command_time_ = nowSec();
}


void RobomimicMove::publishInferenceActive(bool active)
{
    std_msgs::msg::Bool msg;
    msg.data = active;

    inference_active_pub_->publish(msg);
}


void RobomimicMove::publishObservation(
    const ArmState& arm,
    const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr& publisher)
{
    if (arm.controller_ee_name.empty() || !publisher)
        return;

    const Eigen::Affine3d current_pose =
        fr3_model_updater_.robot_data_->getPose(arm.controller_ee_name);

    publisher->publish(affineToPoseStamped(current_pose, node_->now()));
}


RobomimicMove::ComputeResult
RobomimicMove::compute(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/)
{
    auto update_arm_state = [this](ArmState& arm)
    {
        auto& ee_data = ee_data_[arm.controller_ee_name];

        ee_data.x = fr3_model_updater_.robot_data_->getPose(arm.controller_ee_name);
        ee_data.xdot = fr3_model_updater_.robot_data_->getVelocity(arm.controller_ee_name);
        ee_data.xddot.setZero();
    };

    update_arm_state(left_arm_);
    update_arm_state(right_arm_);

    publishObservation(left_arm_, left_eef_pose_pub_);
    publishObservation(right_arm_, right_eef_pose_pub_);


    std::vector<double> action;
    bool new_command = false;
    bool command_timeout = false;

    {
        std::lock_guard<std::mutex> lock(command_mutex_);

        if (has_new_command_)
        {
            action = latest_action_;
            has_new_command_ = false;
            new_command = true;
        }

        if (has_command_ && nowSec() - last_command_time_ > command_timeout_)
        {
            has_command_ = false;
            has_new_command_ = false;
            command_timeout = true;
        }
    }

    auto hold_arm = [this](ArmState& arm)
    {
        auto& ee_data = ee_data_[arm.controller_ee_name];

        arm.x_goal = ee_data.x;
        arm.x_target = ee_data.x;
    };

    if (command_timeout)
    {
        if (arm_mode_ == ArmMode::LEFT || arm_mode_ == ArmMode::DUAL)
            hold_arm(left_arm_);

        if (arm_mode_ == ArmMode::RIGHT || arm_mode_ == ArmMode::DUAL)
            hold_arm(right_arm_);

        RCLCPP_WARN_THROTTLE(
            node_->get_logger(),
            *node_->get_clock(),
            1000,
            "[%s] Absolute action timeout. Holding current pose.",
            name_.c_str());
    }

    auto apply_absolute_action = [this, &action](ArmState& arm, std::size_t offset)
    {
        const Eigen::Vector3d target_pos(
            action[offset + 0],
            action[offset + 1],
            action[offset + 2]);

        Eigen::Quaterniond target_quat(
            action[offset + 6],
            action[offset + 3],
            action[offset + 4],
            action[offset + 5]);

        const double quat_norm = target_quat.norm();

        if (quat_norm < 1e-8)
        {
            RCLCPP_WARN_THROTTLE(
                node_->get_logger(),
                *node_->get_clock(),
                1000,
                "[%s] Invalid quaternion received for %s arm",
                name_.c_str(),
                arm.robot_name.c_str());

            return;
        }

        target_quat.normalize();

        arm.x_goal.translation() = target_pos;
        arm.x_goal.linear() = target_quat.toRotationMatrix();
    };

    if (new_command)
    {
        if (arm_mode_ == ArmMode::LEFT)
        {
            apply_absolute_action(left_arm_, 0);
        }
        else if (arm_mode_ == ArmMode::RIGHT)
        {
            apply_absolute_action(right_arm_, 0);
        }
        else
        {
            apply_absolute_action(left_arm_, 0);
            apply_absolute_action(right_arm_, 7);
        }
    }


    const double max_linear_velocity = 0.10;   // m/s
    const double max_angular_velocity = 0.5;   // rad/s
    const double dt = fr3_model_updater_.dt_;

    auto update_target = [this, max_linear_velocity, max_angular_velocity, dt](ArmState& arm)
    {
        auto& ee_data = ee_data_[arm.controller_ee_name];

        Eigen::Vector3d position_error =
            arm.x_goal.translation() - arm.x_target.translation();

        const double distance = position_error.norm();
        const double max_position_step = max_linear_velocity * dt;

        if (distance > max_position_step && distance > 1e-9)
            arm.x_target.translation() += position_error * (max_position_step / distance);
        else
            arm.x_target.translation() = arm.x_goal.translation();

        Eigen::Quaterniond q_target(arm.x_target.linear());
        Eigen::Quaterniond q_goal(arm.x_goal.linear());

        q_target.normalize();
        q_goal.normalize();

        if (q_target.dot(q_goal) < 0.0)
            q_goal.coeffs() *= -1.0;

        double dot = std::clamp(q_target.dot(q_goal), -1.0, 1.0);
        const double angle = 2.0 * std::acos(dot);
        const double max_angle_step = max_angular_velocity * dt;

        if (angle > max_angle_step && angle > 1e-9)
        {
            const double ratio = max_angle_step / angle;
            arm.x_target.linear() = q_target.slerp(ratio, q_goal).toRotationMatrix();
        }
        else
        {
            arm.x_target.linear() = q_goal.toRotationMatrix();
        }

        ee_data.x_desired = arm.x_target;
        ee_data.xdot_desired.setZero();
    };


    update_target(left_arm_);
    update_target(right_arm_);

    Eigen::VectorXd qdot_mobile =
        Eigen::VectorXd::Zero(fr3_model_updater_.mobile_dof_);

    switch (control_mode_)
    {
        case 0:
        {
            Eigen::VectorXd null_qdot = Eigen::VectorXd::Zero(
                fr3_model_updater_.robot_data_->getActuatorDof());

            fr3_model_updater_.robot_controller_->CLIKStep(
                ee_data_,
                qdot_mobile,
                fr3_model_updater_.qdot_desired_total_,
                null_qdot);

            fr3_model_updater_.q_desired_total_ =
                fr3_model_updater_.q_total_ +
                fr3_model_updater_.dt_ *
                fr3_model_updater_.qdot_desired_total_;

            fr3_model_updater_.torque_desired_total_ =
                fr3_model_updater_.robot_controller_->moveManipulatorJointTorqueStep(
                    fr3_model_updater_.q_desired_total_,
                    fr3_model_updater_.qdot_desired_total_,
                    false);

            fr3_model_updater_.wheel_vel_desired_.setZero();

            break;
        }

        case 1:
        {
            Eigen::VectorXd null_torque = Eigen::VectorXd::Zero(
                fr3_model_updater_.robot_data_->getActuatorDof());

            Eigen::VectorXd wheel_acc_desired =
                Eigen::VectorXd::Zero(fr3_model_updater_.mobile_dof_);

            fr3_model_updater_.robot_controller_->OSFStep(
                ee_data_,
                wheel_acc_desired,
                fr3_model_updater_.torque_desired_total_,
                null_torque);

            fr3_model_updater_.wheel_vel_desired_ =
                fr3_model_updater_.wheel_vel_ +
                wheel_acc_desired * fr3_model_updater_.dt_;

            break;
        }

        case 2:
        {
            std::string time_verbose;

            const bool solved =
                fr3_model_updater_.robot_controller_->QPIKStep(
                    ee_data_,
                    qdot_mobile,
                    fr3_model_updater_.qdot_desired_total_,
                    time_verbose);

            if (!solved)
            {
                fr3_model_updater_.qdot_desired_total_.setZero();
                fr3_model_updater_.wheel_vel_desired_.setZero();
            }

            fr3_model_updater_.q_desired_total_ =
                fr3_model_updater_.q_total_ +
                fr3_model_updater_.dt_ *
                fr3_model_updater_.qdot_desired_total_;

            fr3_model_updater_.torque_desired_total_ =
                fr3_model_updater_.robot_controller_->moveManipulatorJointTorqueStep(
                    fr3_model_updater_.q_desired_total_,
                    fr3_model_updater_.qdot_desired_total_,
                    false);

            break;
        }

        case 3:
        {
            std::string time_verbose;

            Eigen::VectorXd wheel_acc_desired =
                Eigen::VectorXd::Zero(fr3_model_updater_.mobile_dof_);

            const bool solved =
                fr3_model_updater_.robot_controller_->QPIDStep(
                    ee_data_,
                    wheel_acc_desired,
                    fr3_model_updater_.torque_desired_total_,
                    time_verbose);

            if (!solved)
            {
                fr3_model_updater_.torque_desired_total_ =
                    fr3_model_updater_.g_total_;

                wheel_acc_desired.setZero();
            }

            fr3_model_updater_.wheel_vel_desired_ =
                fr3_model_updater_.wheel_vel_ +
                wheel_acc_desired * fr3_model_updater_.dt_;

            break;
        }

        default:
        {
            fr3_model_updater_.qdot_desired_total_.setZero();
            fr3_model_updater_.torque_desired_total_.setZero();
            fr3_model_updater_.wheel_vel_desired_.setZero();

            break;
        }
    }

    fr3_model_updater_.writeCommand(
        fr3_model_updater_.torque_desired_total_ -
            fr3_model_updater_.g_total_,
        fr3_model_updater_.wheel_vel_desired_);

    ++step_count_;

    return ComputeResult::RUNNING;
}


void RobomimicMove::onStop(StopReason reason)
{
    publishInferenceActive(false);

    {
        std::lock_guard<std::mutex> lock(command_mutex_);

        latest_action_.clear();
        has_command_ = false;
        has_new_command_ = false;
        last_command_time_ = 0.0;
    }

    fr3_model_updater_.haltCommands();

    const char* reason_str = "none";

    if (reason == StopReason::CANCELED)
        reason_str = "canceled";
    else if (reason == StopReason::SUCCEEDED)
        reason_str = "succeeded";
    else if (reason == StopReason::ABORTED)
        reason_str = "aborted";

    RCLCPP_INFO(
        node_->get_logger(),
        "[%s] stopped (%s)",
        name_.c_str(),
        reason_str);
}


RobomimicMove::ResultPtr
RobomimicMove::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();

    result->success = reason != StopReason::ABORTED;

    if (reason == StopReason::SUCCEEDED)
        result->message = "RobomimicMove succeeded";
    else if (reason == StopReason::CANCELED)
        result->message = "RobomimicMove canceled";
    else if (reason == StopReason::ABORTED)
        result->message = "RobomimicMove aborted";
    else
        result->message = "RobomimicMove stopped";

    return result;
}


REGISTER_FR3_ACTION_SERVER(
    RobomimicMove,
    "fr3_robomimic_move")


}  // namespace fr3_husky_controller::servers::fr3