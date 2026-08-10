#include <fr3_husky_controller/servers/fr3_husky/robomimic_move_action_server.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>


namespace fr3_husky_controller::servers::fr3_husky
{

namespace
{

FR3HuskyModelUpdater& getFR3HuskyModelUpdater(
    ModelUpdaterBase& model_updater,
    const std::string& server_name)
{
    auto* p = dynamic_cast<FR3HuskyModelUpdater*>(&model_updater);

    if (!p)
        throw std::runtime_error("[" + server_name + "] requires FR3HuskyModelUpdater");

    return *p;
}


std::string getRobotNameFromEEName(const std::string& ee_name)
{
    if (ee_name.rfind("left_", 0) == 0) return "left";
    if (ee_name.rfind("right_", 0) == 0) return "right";
    return "";
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
  fr3_husky_model_updater_(
      getFR3HuskyModelUpdater(model_updater, name))
{
    mode_ = ServerMode::TASK;

    delta_action_sub_ =
        node_->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/robomimic/delta_action",
            rclcpp::QoS(1),
            std::bind(
                &RobomimicMove::onDeltaAction,
                this,
                std::placeholders::_1));

    auto active_qos =
        rclcpp::QoS(1).reliable().transient_local();

    inference_active_pub_ =
        node_->create_publisher<std_msgs::msg::Bool>(
            "/robomimic/inference_active",
            active_qos);

    eef_pose_pub_ =
        node_->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/robomimic/obs/eef_pose",
            10);

    publishInferenceActive(false);

    RCLCPP_INFO(
        node_->get_logger(),
        "[%s] RobomimicMove created",
        name_.c_str());
}


bool RobomimicMove::acceptGoal(
    const ActionT::Goal& goal)
{
    if (!fr3_husky_model_updater_.HasEffortCommandInterface())
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] Reject: effort command interface is required",
            name_.c_str());

        return false;
    }

    if (goal.controller_ee_name.empty())
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] Reject: controller_ee_name is empty",
            name_.c_str());

        return false;
    }

    if (!fr3_husky_model_updater_.robot_data_->hasLinkFrame(
            goal.controller_ee_name))
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] Reject: unknown EE name [%s]",
            name_.c_str(),
            goal.controller_ee_name.c_str());

        return false;
    }

    if (goal.mode < 0 || goal.mode > 3)
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] Reject: mode must be 0-3",
            name_.c_str());

        return false;
    }

    return true;
}


void RobomimicMove::onGoalAccepted(
    const ActionT::Goal& goal)
{
    controller_ee_name_ = goal.controller_ee_name;
    robot_name_ = getRobotNameFromEEName(controller_ee_name_);

    control_mode_ = goal.mode;

    position_scale_ =
        goal.position_scale > 0.0
            ? goal.position_scale
            : 1.0;

    rotation_scale_ =
        goal.rotation_scale > 0.0
            ? goal.rotation_scale
            : 1.0;

    command_timeout_ =
        goal.command_timeout > 0.0
            ? goal.command_timeout
            : 0.5;

    requestActivate();
}


void RobomimicMove::onStart()
{
    ee_data_.clear();

    ee_data_[controller_ee_name_] =
        drc::TaskSpaceData::Zero();

    auto& ee_data =
        ee_data_[controller_ee_name_];

    ee_data.x =
        fr3_husky_model_updater_.robot_data_->getPose(
            controller_ee_name_);

    ee_data.xdot =
        fr3_husky_model_updater_.robot_data_->getVelocity(
            controller_ee_name_);

    ee_data.xddot.setZero();

    ee_data.setInit();
    ee_data.setDesired();
    ee_data.xdot_desired.setZero();

    x_goal_ = ee_data.x;
    x_target_ = ee_data.x;

    {
        std::lock_guard<std::mutex> lock(command_mutex_);

        latest_action_.clear();

        has_command_ = false;
        has_new_command_ = false;

        last_command_time_ = 0.0;

        previous_gripper_command_ = 0.0;
        gripper_closed_ = false;
    }

    step_count_ = 0;

    publishInferenceActive(true);

    RCLCPP_INFO(
        node_->get_logger(),
        "[%s] started: ee=%s, mode=%d",
        name_.c_str(),
        controller_ee_name_.c_str(),
        control_mode_);
}


void RobomimicMove::onDeltaAction(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (!msg)
        return;

    if (msg->data.size() < 6)
    {
        RCLCPP_WARN_THROTTLE(
            node_->get_logger(),
            *node_->get_clock(),
            1000,
            "[%s] Invalid delta action dimension: %zu",
            name_.c_str(),
            msg->data.size());

        return;
    }

    for (const double value : msg->data)
    {
        if (!std::isfinite(value))
        {
            RCLCPP_WARN(
                node_->get_logger(),
                "[%s] Received NaN / Inf action",
                name_.c_str());

            return;
        }
    }

    std::lock_guard<std::mutex> lock(command_mutex_);

    latest_action_ = msg->data;

    has_command_ = true;
    has_new_command_ = true;

    last_command_time_ = nowSec();
}


void RobomimicMove::publishInferenceActive(
    bool active)
{
    std_msgs::msg::Bool msg;
    msg.data = active;

    inference_active_pub_->publish(msg);
}


void RobomimicMove::publishObservation()
{
    if (controller_ee_name_.empty())
        return;

    const Eigen::Affine3d current_pose =
        fr3_husky_model_updater_.robot_data_->getPose(
            controller_ee_name_);

    eef_pose_pub_->publish(
        affineToPoseStamped(
            current_pose,
            node_->now()));
}


RobomimicMove::ComputeResult
RobomimicMove::compute(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/)
{
    auto& ee_data =
        ee_data_[controller_ee_name_];

    ee_data.x =
        fr3_husky_model_updater_.robot_data_->getPose(
            controller_ee_name_);

    ee_data.xdot =
        fr3_husky_model_updater_.robot_data_->getVelocity(
            controller_ee_name_);

    ee_data.xddot.setZero();

    publishObservation();

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

        if (
            has_command_ &&
            nowSec() - last_command_time_ > command_timeout_)
        {
            has_command_ = false;
            has_new_command_ = false;

            command_timeout = true;
        }
    }

    if (command_timeout)
    {
        x_goal_ = ee_data.x;
        x_target_ = ee_data.x;

        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] Delta action timeout. Holding current pose.",
            name_.c_str());
    }


    if (new_command)
    {
        Eigen::Vector3d delta_pos(
            action[0],
            action[1],
            action[2]);

        Eigen::Vector3d delta_rot(
            action[3],
            action[4],
            action[5]);

        delta_pos *= position_scale_;
        delta_rot *= rotation_scale_;

        x_goal_.translation() += delta_pos;

        const double angle =
            delta_rot.norm();

        if (angle > 1e-9)
        {
            const Eigen::Vector3d axis =
                delta_rot / angle;

            x_goal_.linear() =
                x_goal_.linear() *
                Eigen::AngleAxisd(
                    angle,
                    axis)
                    .toRotationMatrix();
        }


        if (
            action.size() >= 7 &&
            !robot_name_.empty())
        {
            const double gripper_command =
                action[6];

            if (
                gripper_command > 0.0 &&
                previous_gripper_command_ <= 0.0)
            {
                fr3_husky_model_updater_.GripperGrasp(
                    robot_name_,
                    0.0,
                    0.1,
                    100.0);

                gripper_closed_ = true;
            }
            else if (
                gripper_command <= 0.0 &&
                previous_gripper_command_ > 0.0)
            {
                fr3_husky_model_updater_.GripperOpen(
                    robot_name_,
                    0.1);

                gripper_closed_ = false;
            }

            previous_gripper_command_ =
                gripper_command;
        }
    }


    const double alpha = 0.25;

    x_target_.translation() =
        (1.0 - alpha) *
            x_target_.translation() +
        alpha *
            x_goal_.translation();

    Eigen::Quaterniond q_target(
        x_target_.linear());

    Eigen::Quaterniond q_goal(
        x_goal_.linear());

    q_target.normalize();
    q_goal.normalize();

    if (q_target.dot(q_goal) < 0.0)
        q_goal.coeffs() *= -1.0;

    x_target_.linear() =
        q_target
            .slerp(alpha, q_goal)
            .toRotationMatrix();


    ee_data.x_desired =
        x_target_;

    ee_data.xdot_desired.setZero();


    Eigen::VectorXd qdot_mobile =
        Eigen::VectorXd::Zero(
            fr3_husky_model_updater_.mobile_dof_);


    switch (control_mode_)
    {
        case 0:
        {
            Eigen::VectorXd null_qdot =
                Eigen::VectorXd::Zero(
                    fr3_husky_model_updater_
                        .robot_data_
                        ->getActuatorDof());

            fr3_husky_model_updater_
                .robot_controller_
                ->CLIKStep(
                    ee_data_,
                    qdot_mobile,
                    fr3_husky_model_updater_
                        .qdot_desired_total_,
                    null_qdot);

            fr3_husky_model_updater_
                .q_desired_total_ =
                fr3_husky_model_updater_
                    .q_total_ +
                fr3_husky_model_updater_
                    .dt_ *
                fr3_husky_model_updater_
                    .qdot_desired_total_;

            fr3_husky_model_updater_
                .torque_desired_total_ =
                fr3_husky_model_updater_
                    .robot_controller_
                    ->moveManipulatorJointTorqueStep(
                        fr3_husky_model_updater_
                            .q_desired_total_,
                        fr3_husky_model_updater_
                            .qdot_desired_total_,
                        false);

            fr3_husky_model_updater_
                .wheel_vel_desired_
                .setZero();

            break;
        }


        case 1:
        {
            Eigen::VectorXd null_torque =
                Eigen::VectorXd::Zero(
                    fr3_husky_model_updater_
                        .robot_data_
                        ->getActuatorDof());

            Eigen::VectorXd wheel_acc_desired =
                Eigen::VectorXd::Zero(
                    fr3_husky_model_updater_
                        .mobile_dof_);

            fr3_husky_model_updater_
                .robot_controller_
                ->OSFStep(
                    ee_data_,
                    wheel_acc_desired,
                    fr3_husky_model_updater_
                        .torque_desired_total_,
                    null_torque);

            fr3_husky_model_updater_
                .wheel_vel_desired_ =
                fr3_husky_model_updater_
                    .wheel_vel_ +
                wheel_acc_desired *
                    fr3_husky_model_updater_
                        .dt_;

            break;
        }


        case 2:
        {
            std::string time_verbose;

            const bool solved =
                fr3_husky_model_updater_
                    .robot_controller_
                    ->QPIKStep(
                        ee_data_,
                        qdot_mobile,
                        fr3_husky_model_updater_
                            .qdot_desired_total_,
                        time_verbose);

            if (!solved)
            {
                fr3_husky_model_updater_
                    .qdot_desired_total_
                    .setZero();

                fr3_husky_model_updater_
                    .wheel_vel_desired_
                    .setZero();
            }

            fr3_husky_model_updater_
                .q_desired_total_ =
                fr3_husky_model_updater_
                    .q_total_ +
                fr3_husky_model_updater_
                    .dt_ *
                fr3_husky_model_updater_
                    .qdot_desired_total_;

            fr3_husky_model_updater_
                .torque_desired_total_ =
                fr3_husky_model_updater_
                    .robot_controller_
                    ->moveManipulatorJointTorqueStep(
                        fr3_husky_model_updater_
                            .q_desired_total_,
                        fr3_husky_model_updater_
                            .qdot_desired_total_,
                        false);

            break;
        }


        case 3:
        {
            std::string time_verbose;

            Eigen::VectorXd wheel_acc_desired =
                Eigen::VectorXd::Zero(
                    fr3_husky_model_updater_
                        .mobile_dof_);

            const bool solved =
                fr3_husky_model_updater_
                    .robot_controller_
                    ->QPIDStep(
                        ee_data_,
                        wheel_acc_desired,
                        fr3_husky_model_updater_
                            .torque_desired_total_,
                        time_verbose);

            if (!solved)
            {
                fr3_husky_model_updater_
                    .torque_desired_total_ =
                    fr3_husky_model_updater_
                        .g_total_;

                wheel_acc_desired.setZero();
            }

            fr3_husky_model_updater_
                .wheel_vel_desired_ =
                fr3_husky_model_updater_
                    .wheel_vel_ +
                wheel_acc_desired *
                    fr3_husky_model_updater_
                        .dt_;

            break;
        }


        default:
        {
            fr3_husky_model_updater_
                .qdot_desired_total_
                .setZero();

            fr3_husky_model_updater_
                .torque_desired_total_
                .setZero();

            fr3_husky_model_updater_
                .wheel_vel_desired_
                .setZero();

            break;
        }
    }


    fr3_husky_model_updater_.writeCommand(
        fr3_husky_model_updater_
                .torque_desired_total_ -
            fr3_husky_model_updater_
                .g_total_,
        fr3_husky_model_updater_
            .wheel_vel_desired_);


    ++step_count_;

    return ComputeResult::RUNNING;
}


void RobomimicMove::onStop(
    StopReason reason)
{
    publishInferenceActive(false);

    fr3_husky_model_updater_.haltCommands();

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
RobomimicMove::makeResult(
    StopReason /*reason*/)
{
    auto result =
        std::make_shared<ActionT::Result>();

    return result;
}


REGISTER_FR3_HUSKY_ACTION_SERVER(
    RobomimicMove,
    "fr3_husky_robomimic_move")


}  // namespace fr3_husky_controller::servers::fr3_husky