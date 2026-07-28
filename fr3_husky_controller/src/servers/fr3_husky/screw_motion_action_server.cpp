#include <fr3_husky_controller/servers/fr3_husky/screw_motion_action_server.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fr3_husky_controller/utils/dyros_math.h>

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

constexpr double kEps = 1e-9;
constexpr double kDefaultOffset = 0.1;
constexpr double kDefaultAngle = M_PI / 2.0;
constexpr double kDefaultDuration = 10.0;
constexpr double kDefaultPressDepth = 0.001;
constexpr double kDefaultPrepressDuration = 0.5;
constexpr double kDefaultPosTolerance = 0.01;
constexpr double kDefaultOriTolerance = 0.05;

}  // namespace

ScrewMotionBase::ScrewMotionBase(
    const std::string& name,
    const NodePtr& node,
    ModelUpdaterBase& model_updater,
    bool force_base_z_axis)
: Base(name, node, model_updater),
  fr3_husky_model_updater_(getFR3HuskyModelUpdater(model_updater, name)),
  force_base_z_axis_(force_base_z_axis)
{
    mode_ = ServerMode::TASK;
    RCLCPP_INFO(node_->get_logger(), "[%s] ScrewMotion created", name_.c_str());
}

bool ScrewMotionBase::acceptGoal(const ActionT::Goal& goal)
{
    const std::string mode = goal.mode.empty() ? "normal" : goal.mode;
    if (mode != "normal" && mode != "press")
    {
        RCLCPP_WARN(node_->get_logger(),
                    "[%s] Reject: mode must be normal or press, got %s",
                    name_.c_str(), mode.c_str());
        return false;
    }

    if (goal.press_depth < 0.0 || goal.prepress_duration < 0.0)
    {
        RCLCPP_WARN(node_->get_logger(),
                    "[%s] Reject: press_depth and prepress_duration must be non-negative",
                    name_.c_str());
        return false;
    }

    if (mode == "press")
    {
        const double effective_offset = goal.offset > 0.0 ? goal.offset : kDefaultOffset;
        const double effective_press_depth =
            goal.press_depth > 0.0 ? goal.press_depth : kDefaultPressDepth;
        if (effective_press_depth >= effective_offset)
        {
            RCLCPP_WARN(node_->get_logger(),
                        "[%s] Reject: press_depth (%.4f) must be smaller than offset (%.4f)",
                        name_.c_str(), effective_press_depth, effective_offset);
            return false;
        }
    }

    std::vector<std::string> ee_names;
    try
    {
        ee_names = eeNamesFromArm(goal.arm.empty() ? "left" : goal.arm);
    }
    catch (const std::exception& e)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: %s", name_.c_str(), e.what());
        return false;
    }
    for (const auto& ee_name : ee_names)
    {
        if (!fr3_husky_model_updater_.robot_data_->hasLinkFrame(ee_name))
        {
            RCLCPP_WARN(node_->get_logger(), "[%s] Reject: unknown ee_name: %s", name_.c_str(), ee_name.c_str());
            return false;
        }
    }
    return true;
}

void ScrewMotionBase::onGoalAccepted(const ActionT::Goal& goal)
{
    arm_ = goal.arm.empty() ? "left" : goal.arm;
    ee_names_ = eeNamesFromArm(arm_);

    Eigen::Vector3d center_direction = vectorMsgToEigen(goal.center_direction_ee);
    if (center_direction.norm() < kEps) center_direction = Eigen::Vector3d::UnitX();
    center_direction_ee_ = center_direction.normalized();

    Eigen::Vector3d axis = vectorMsgToEigen(goal.axis_base);
    if (axis.norm() < kEps) axis = Eigen::Vector3d::UnitZ();
    axis_base_ = force_base_z_axis_ ? Eigen::Vector3d::UnitZ() : axis.normalized();

    offset_ = goal.offset > 0.0 ? goal.offset : kDefaultOffset;
    const double raw_angle = std::abs(goal.angle) > kEps ? goal.angle : kDefaultAngle;
    angle_ = goal.angle_in_degrees ? raw_angle * DEG2RAD : raw_angle;
    pitch_ = goal.pitch;
    motion_mode_ = goal.mode.empty() ? "normal" : goal.mode;
    press_depth_ = goal.press_depth > 0.0 ? goal.press_depth : kDefaultPressDepth;
    prepress_duration_ =
        goal.prepress_duration > 0.0 ? goal.prepress_duration : kDefaultPrepressDuration;
    duration_ = goal.duration > 0.0 ? goal.duration : kDefaultDuration;
    pos_tolerance_ = goal.pos_tolerance > 0.0 ? goal.pos_tolerance : kDefaultPosTolerance;
    ori_tolerance_ = goal.ori_tolerance > 0.0 ? goal.ori_tolerance : kDefaultOriTolerance;

    start_time_set_ = false;
    last_progress_ = 0.0;
    last_position_errors_.assign(ee_names_.size(), 0.0);
    last_orientation_errors_.assign(ee_names_.size(), 0.0);
    result_error_code_ = 0;

    requestActivate();

    RCLCPP_INFO(node_->get_logger(),
                "[%s] goal accepted: arm=%s offset=%.4f angle=%.4f rad pitch=%.6f m/rev "
                "mode=%s press_depth=%.4f prepress_duration=%.3f duration=%.3f "
                "axis=[%.3f %.3f %.3f]",
                name_.c_str(), arm_.c_str(), offset_, angle_, pitch_,
                motion_mode_.c_str(), press_depth_, prepress_duration_, duration_,
                axis_base_.x(), axis_base_.y(), axis_base_.z());
}

void ScrewMotionBase::onStart()
{
    fr3_husky_model_updater_.setInitFromCurrent();

    ee_data_.clear();
    start_poses_.clear();
    centers_base_.clear();

    for (const auto& ee_name : ee_names_)
    {
        ee_data_[ee_name] = drc::TaskSpaceData::Zero();
        ee_data_[ee_name].x = fr3_husky_model_updater_.robot_data_->getPose(ee_name);
        ee_data_[ee_name].xdot = fr3_husky_model_updater_.robot_data_->getVelocity(ee_name);
        ee_data_[ee_name].xddot.setZero();
        ee_data_[ee_name].setInit();
        ee_data_[ee_name].setDesired();
        ee_data_[ee_name].xdot_desired.setZero();

        const Eigen::Affine3d start_pose = ee_data_[ee_name].x;
        start_poses_[ee_name] = start_pose;

        Eigen::Vector3d center_direction_base = start_pose.linear() * center_direction_ee_;
        center_direction_base -= axis_base_ * center_direction_base.dot(axis_base_);
        if (center_direction_base.norm() < kEps)
        {
            result_error_code_ = 2;
            throw std::runtime_error("center_direction_ee is parallel to the rotation axis after frame conversion");
        }
        center_direction_base.normalize();
        centers_base_[ee_name] = start_pose.translation() + offset_ * center_direction_base;

        RCLCPP_INFO(node_->get_logger(),
                    "[%s] started: ee=%s center_base=[%.4f %.4f %.4f] start=[%.4f %.4f %.4f]",
                    name_.c_str(), ee_name.c_str(),
                    centers_base_[ee_name].x(), centers_base_[ee_name].y(), centers_base_[ee_name].z(),
                    start_pose.translation().x(), start_pose.translation().y(), start_pose.translation().z());
    }

    last_position_errors_.assign(ee_names_.size(), 0.0);
    last_orientation_errors_.assign(ee_names_.size(), 0.0);
    start_time_set_ = false;
}

ScrewMotionBase::ComputeResult ScrewMotionBase::compute(
    const rclcpp::Time& time,
    const rclcpp::Duration& /*period*/)
{
    for (const auto& ee_name : ee_names_)
    {
        auto& ee_data = ee_data_[ee_name];
        ee_data.x = fr3_husky_model_updater_.robot_data_->getPose(ee_name);
        ee_data.xdot = fr3_husky_model_updater_.robot_data_->getVelocity(ee_name);
        ee_data.xddot.setZero();
    }

    if (!start_time_set_)
    {
        start_time_ = time;
        start_time_set_ = true;
    }

    const double elapsed = (time - start_time_).seconds();
    const bool press_mode = motion_mode_ == "press";
    const double prepress_time = press_mode ? prepress_duration_ : 0.0;
    const double total_duration = prepress_time + duration_;
    const double screw_time = std::clamp(elapsed - prepress_time, 0.0, duration_);
    const double theta =
        dyros_math::cubic(screw_time, 0.0, duration_, 0.0, angle_, 0.0, 0.0);
    const double theta_dot =
        dyros_math::cubicDot(screw_time, 0.0, duration_, 0.0, angle_, 0.0, 0.0);
    const double progress = std::clamp(elapsed / total_duration, 0.0, 1.0);

    double press_scale = 0.0;
    double press_scale_dot = 0.0;
    if (press_mode)
    {
        const double ramp_time = std::clamp(elapsed, 0.0, prepress_duration_);
        press_scale = dyros_math::cubic(
            ramp_time, 0.0, prepress_duration_, 0.0, 1.0, 0.0, 0.0);
        press_scale_dot = elapsed < prepress_duration_
            ? dyros_math::cubicDot(
                ramp_time, 0.0, prepress_duration_, 0.0, 1.0, 0.0, 0.0)
            : 0.0;
    }

    const Eigen::Matrix3d R_axis = force_base_z_axis_
        ? dyros_math::rotateWithZ(theta)
        : rotationFromAxisAngle(axis_base_, theta);

    std::vector<geometry_msgs::msg::Pose> desired_poses;
    std::vector<double> position_errors;
    std::vector<double> orientation_errors;
    bool all_reached = true;

    for (const auto& ee_name : ee_names_)
    {
        auto& ee_data = ee_data_[ee_name];
        const Eigen::Affine3d& start_pose = start_poses_.at(ee_name);
        const Eigen::Vector3d& center_base = centers_base_.at(ee_name);

        const Eigen::Vector3d start_radius = start_pose.translation() - center_base;
        const double axial_displacement = pitch_ * theta / (2.0 * M_PI);
        const Eigen::Vector3d center_on_axis =
            center_base + axis_base_ * axial_displacement;
        const Eigen::Vector3d nominal_radius = R_axis * start_radius;
        const Eigen::Vector3d p_nominal = center_on_axis + nominal_radius;
        const Eigen::Vector3d inward_direction = -nominal_radius.normalized();
        const Eigen::Vector3d p_control =
            p_nominal + press_scale * press_depth_ * inward_direction;
        const Eigen::Matrix3d R_des = R_axis * start_pose.linear();

        Eigen::Affine3d T_des = Eigen::Affine3d::Identity();
        T_des.linear() = R_des;
        T_des.translation() = p_control;

        const Eigen::Vector3d omega_des = axis_base_ * theta_dot;
        const double axial_velocity = pitch_ * theta_dot / (2.0 * M_PI);
        const Eigen::Vector3d control_radius = p_control - center_on_axis;
        const Eigen::Vector3d v_des =
            omega_des.cross(control_radius)
            + axis_base_ * axial_velocity
            + press_scale_dot * press_depth_ * inward_direction;

        ee_data.x_desired = T_des;
        ee_data.xdot_desired.setZero();
        ee_data.xdot_desired.head<3>() = v_des;
        ee_data.xdot_desired.tail<3>() = omega_des;

        // The radial preload is intentional virtual penetration. Use the
        // nominal path for completion when contact prevents reaching p_control.
        const double p_err = (p_nominal - ee_data.x.translation()).norm();
        const double o_err = orientationError(R_des, ee_data.x.linear());
        desired_poses.push_back(affineToPoseMsg(T_des));
        position_errors.push_back(p_err);
        orientation_errors.push_back(o_err);
        if (p_err > pos_tolerance_ || o_err > ori_tolerance_) all_reached = false;
    }

    last_progress_ = progress;
    last_position_errors_ = position_errors;
    last_orientation_errors_ = orientation_errors;

    writeTaskSpaceCommand();

    auto fb = std::make_shared<ActionT::Feedback>();
    fb->progress = progress;
    fb->status_message =
        press_mode && elapsed < prepress_duration_
            ? "Building radial preload"
            : "Executing screw motion";
    fb->desired_poses = desired_poses;
    fb->position_errors = position_errors;
    fb->orientation_errors = orientation_errors;
    publishFeedback(fb);

    if (progress >= 1.0)
    {
        // TODO: Add a post-screw sequence that releases radial contact, returns
        // to the initial angular pose while preserving pitch displacement, and
        // repeats the tighten/reset cycle N times. Until then, keep preload.
        if (all_reached) return ComputeResult::SUCCEEDED;
        result_error_code_ = 3;
        return ComputeResult::ABORTED;
    }

    return ComputeResult::RUNNING;
}

void ScrewMotionBase::writeTaskSpaceCommand()
{
    Eigen::VectorXd qdot_mobile = Eigen::VectorXd::Zero(fr3_husky_model_updater_.mobile_dof_);
    Eigen::VectorXd null_qdot_mani = Eigen::VectorXd::Zero(fr3_husky_model_updater_.manipulator_dof_);
    Eigen::VectorXd null_qdot_mobile = Eigen::VectorXd::Zero(fr3_husky_model_updater_.mobile_dof_);

    Eigen::VectorXd null_qdot = Eigen::VectorXd::Zero(fr3_husky_model_updater_.robot_data_->getActuatorDof());
    const auto& act_idx = fr3_husky_model_updater_.robot_data_->getActuatorIndex();
    null_qdot.segment(act_idx.mobi_start, fr3_husky_model_updater_.mobile_dof_) = null_qdot_mobile;
    null_qdot.segment(act_idx.mani_start, fr3_husky_model_updater_.manipulator_dof_) = null_qdot_mani;

    fr3_husky_model_updater_.robot_controller_->CLIKStep(
        ee_data_,
        qdot_mobile,
        fr3_husky_model_updater_.qdot_desired_total_,
        null_qdot);

    fr3_husky_model_updater_.q_desired_total_ =
        fr3_husky_model_updater_.q_total_ +
        fr3_husky_model_updater_.dt_ * fr3_husky_model_updater_.qdot_desired_total_;

    fr3_husky_model_updater_.torque_desired_total_ =
        fr3_husky_model_updater_.robot_controller_->moveManipulatorJointTorqueStep(
            fr3_husky_model_updater_.q_desired_total_,
            fr3_husky_model_updater_.qdot_desired_total_,
            false);

    fr3_husky_model_updater_.wheel_vel_desired_.setZero();
    fr3_husky_model_updater_.writeCommand(
        fr3_husky_model_updater_.torque_desired_total_ - fr3_husky_model_updater_.g_total_,
        fr3_husky_model_updater_.wheel_vel_desired_);
}

void ScrewMotionBase::onStop(StopReason reason)
{
    if (reason != StopReason::SUCCEEDED)
        model_updater_.haltCommands();

    const char* rs = (reason == StopReason::CANCELED)  ? "canceled"  :
                     (reason == StopReason::SUCCEEDED) ? "succeeded" :
                     (reason == StopReason::ABORTED)   ? "aborted"   : "none";

    RCLCPP_INFO(node_->get_logger(), "[%s] stopped (%s)", name_.c_str(), rs);
}

ScrewMotionBase::ResultPtr ScrewMotionBase::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();

    if (reason == StopReason::SUCCEEDED)
    {
        result->success = true;
        result->message = "Screw motion completed";
        result->error_code = 0;
    }
    else if (reason == StopReason::CANCELED)
    {
        result->success = false;
        result->message = "Cancelled";
        result->error_code = 0;
    }
    else
    {
        result->success = false;
        result->error_code = result_error_code_;

        if (result_error_code_ == 2)
        {
            result->message =
                "Screw motion failed: center_direction_ee is parallel to the rotation axis after frame conversion";
        }
        else if (result_error_code_ == 3)
        {
            const double max_position_error = last_position_errors_.empty()
                ? 0.0
                : *std::max_element(last_position_errors_.begin(), last_position_errors_.end());
            const double max_orientation_error = last_orientation_errors_.empty()
                ? 0.0
                : *std::max_element(last_orientation_errors_.begin(), last_orientation_errors_.end());

            std::ostringstream oss;
            oss << "Screw motion duration elapsed, but final tracking error is outside tolerance"
                << " (max_position_error=" << max_position_error
                << ", pos_tolerance=" << pos_tolerance_
                << ", max_orientation_error=" << max_orientation_error
                << ", ori_tolerance=" << ori_tolerance_ << ")";
            result->message = oss.str();
        }
        else
        {
            result->message = "Aborted";
        }

        RCLCPP_WARN(node_->get_logger(), "[%s] %s", name_.c_str(), result->message.c_str());
    }

    return result;
}

Eigen::Vector3d ScrewMotionBase::vectorMsgToEigen(const geometry_msgs::msg::Vector3& msg)
{
    return Eigen::Vector3d(msg.x, msg.y, msg.z);
}

double ScrewMotionBase::orientationError(const Eigen::Matrix3d& R_des, const Eigen::Matrix3d& R_cur)
{
    Eigen::Matrix3d R_err = R_des * R_cur.transpose();
    Eigen::AngleAxisd aa(R_err);
    return std::abs(aa.angle());
}

geometry_msgs::msg::Pose ScrewMotionBase::affineToPoseMsg(const Eigen::Affine3d& T)
{
    geometry_msgs::msg::Pose msg;
    msg.position.x = T.translation().x();
    msg.position.y = T.translation().y();
    msg.position.z = T.translation().z();

    Eigen::Quaterniond q(T.linear());
    q.normalize();
    msg.orientation.x = q.x();
    msg.orientation.y = q.y();
    msg.orientation.z = q.z();
    msg.orientation.w = q.w();
    return msg;
}

Eigen::Matrix3d ScrewMotionBase::rotationFromAxisAngle(const Eigen::Vector3d& axis, double angle)
{
    return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
}

std::vector<std::string> ScrewMotionBase::eeNamesFromArm(const std::string& arm)
{
    if (arm == "left") return {"left_fr3_hand_tcp"};
    if (arm == "right") return {"right_fr3_hand_tcp"};
    if (arm == "both") return {"left_fr3_hand_tcp", "right_fr3_hand_tcp"};
    throw std::invalid_argument("arm must be one of: left, right, both");
}

REGISTER_FR3_HUSKY_ACTION_SERVER(ScrewMotion, "fr3_husky_screw")
REGISTER_FR3_HUSKY_ACTION_SERVER(ScrewMotionZ, "fr3_husky_screw_z")

}  // namespace fr3_husky_controller::servers::fr3_husky
