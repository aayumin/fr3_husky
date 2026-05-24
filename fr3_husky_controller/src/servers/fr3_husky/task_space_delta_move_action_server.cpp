// task_space_delta_move_action_server.cpp

#include <fr3_husky_controller/servers/fr3_husky/task_space_delta_move_action_server.hpp>

#include <chrono>
#include <stdexcept>
#include <string>

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

}

TaskSpaceDeltaMove::TaskSpaceDeltaMove(
    const std::string& name,
    const NodePtr& node,
    ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_husky_model_updater_(getFR3HuskyModelUpdater(model_updater, name))
{
    mode_ = ServerMode::TASK;
    RCLCPP_INFO(node_->get_logger(), "[%s] TaskSpaceDeltaMove created", name_.c_str());
}

bool TaskSpaceDeltaMove::acceptGoal(const ActionT::Goal& goal)
{
    if (goal.ee_names.empty())
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: ee_names is empty", name_.c_str());
        return false;
    }

    if (goal.ee_names.size() != goal.target_delta_poses.size())
    {
        RCLCPP_WARN(node_->get_logger(),
                    "[%s] Reject: ee_names.size() != target_delta_poses_.size()",
                    name_.c_str());
        return false;
    }

    for (const auto& ee_name : goal.ee_names)
    {
        if (!fr3_husky_model_updater_.robot_data_->hasLinkFrame(ee_name))
        {
            RCLCPP_WARN(node_->get_logger(), "[%s] Reject: unknown ee_name: %s", name_.c_str(), ee_name.c_str());
            return false;
        }
    }

    return true;
}

void TaskSpaceDeltaMove::onGoalAccepted(const ActionT::Goal& goal)
{
    ee_names_ = goal.ee_names;
    target_delta_poses_.clear();

    for (const auto& pose_msg : goal.target_delta_poses)
        target_delta_poses_.push_back(poseMsgToAffine(pose_msg));

    duration_ = goal.duration > 0.0 ? goal.duration : 3.0;
    pos_tolerance_ = goal.pos_tolerance > 0.0 ? goal.pos_tolerance : 0.01;
    ori_tolerance_ = goal.ori_tolerance > 0.0 ? goal.ori_tolerance : 0.05;

    result_error_code_ = 0;
    start_time_set_ = false;

    requestActivate();

    RCLCPP_INFO(node_->get_logger(),
                "[%s] goal accepted: %zu EE targets",
                name_.c_str(), ee_names_.size());
}

void TaskSpaceDeltaMove::onStart()
{
    fr3_husky_model_updater_.setInitFromCurrent();

    ee_data_.clear();
    start_poses_.clear();

    for (const auto& ee_name : ee_names_)
    {
        ee_data_[ee_name] = drc::TaskSpaceData::Zero();
        ee_data_[ee_name].x = fr3_husky_model_updater_.robot_data_->getPose(ee_name);
        ee_data_[ee_name].xdot = fr3_husky_model_updater_.robot_data_->getVelocity(ee_name);
        ee_data_[ee_name].xddot.setZero();
        ee_data_[ee_name].setInit();
        ee_data_[ee_name].setDesired();
        ee_data_[ee_name].xdot_desired.setZero();

        start_poses_.push_back(ee_data_[ee_name].x);
    }

    start_time_set_ = false;
    RCLCPP_INFO(node_->get_logger(), "[%s] started", name_.c_str());
}


Eigen::Affine3d TaskSpaceDeltaMove::poseMsgToAffine(const geometry_msgs::msg::Pose& msg)
{
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.translation() << msg.position.x, msg.position.y, msg.position.z;

    Eigen::Quaterniond q(
        msg.orientation.w,
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z);

    if (q.norm() < 1e-9)
        q = Eigen::Quaterniond::Identity();
    else
        q.normalize();

    T.linear() = q.toRotationMatrix();
    return T;
}

double TaskSpaceDeltaMove::orientationError(const Eigen::Matrix3d& R_des, const Eigen::Matrix3d& R_cur)
{
    Eigen::Matrix3d R_err = R_des * R_cur.transpose();
    Eigen::AngleAxisd aa(R_err);
    return std::abs(aa.angle());
}



Eigen::Vector6d TaskSpaceDeltaMove::computeTargetVelocity(
    const Eigen::Affine3d& prev,
    const Eigen::Affine3d& cur,
    double dt)
{
    Eigen::Vector6d vel;
    vel.setZero();

    if (dt <= 1e-6) {
        return vel;
    }

    vel.head<3>() = (cur.translation() - prev.translation()) / dt;

    Eigen::Matrix3d R_delta = cur.linear() * prev.linear().transpose();
    Eigen::AngleAxisd aa(R_delta);

    double angle = aa.angle();
    if (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }

    if (std::abs(angle) > 1e-9) {
        vel.tail<3>() = aa.axis() * angle / dt;
    }

    return vel;
}


TaskSpaceDeltaMove::ComputeResult TaskSpaceDeltaMove::compute(
    const rclcpp::Time& time,
    const rclcpp::Duration& /*period*/)
{
    for (auto& [ee_name, ee_data] : ee_data_)
    {
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
    const double s = std::clamp(elapsed / duration_, 0.0, 1.0);
    const double dt = fr3_husky_model_updater_.dt_;

    std::vector<double> pos_errors;
    std::vector<double> ori_errors;
    bool all_reached = true;

    for (size_t i = 0; i < ee_names_.size(); ++i)
    {
        const auto& ee_name = ee_names_[i];
        // const Eigen::Affine3d& T0 = start_poses_[i];
        // const Eigen::Affine3d& T1 = T0 * target_poses_[i];
        const Eigen::Affine3d& T0 = start_poses_.at(i);
        const Eigen::Affine3d& T_delta = target_delta_poses_.at(i);
        const Eigen::Affine3d T1 = T0 * T_delta;

        Eigen::Affine3d T_des = Eigen::Affine3d::Identity();

        T_des.translation() = (1.0 - s) * T0.translation() + s * T1.translation();

        Eigen::Quaterniond q0(T0.linear());
        Eigen::Quaterniond q1(T1.linear());
        q0.normalize();
        q1.normalize();
        if (q0.dot(q1) < 0.0) q1.coeffs() *= -1.0;
        T_des.linear() = q0.slerp(s, q1).toRotationMatrix();

        ee_data_[ee_name].x_desired = T_des;
        ee_data_[ee_name].xdot_desired.setZero();


        const double p_err = (T1.translation() - ee_data_[ee_name].x.translation()).norm();
        const double o_err = orientationError(T1.linear(), ee_data_[ee_name].x.linear());

        pos_errors.push_back(p_err);
        ori_errors.push_back(o_err);

        if (p_err > pos_tolerance_ || o_err > ori_tolerance_) all_reached = false;
    }

    Eigen::VectorXd qdot_mobile = Eigen::VectorXd::Zero(fr3_husky_model_updater_.mobile_dof_);
    Eigen::VectorXd null_qdot_mani = Eigen::VectorXd::Zero(fr3_husky_model_updater_.manipulator_dof_);
    Eigen::VectorXd null_qdot_mobile = Eigen::VectorXd::Zero(fr3_husky_model_updater_.mobile_dof_);

    Eigen::VectorXd null_qdot(fr3_husky_model_updater_.robot_data_->getActuatorDof());
    const auto& act_idx = fr3_husky_model_updater_.robot_data_->getActuatorIndex();

    null_qdot.segment(act_idx.mobi_start, fr3_husky_model_updater_.mobile_dof_) = null_qdot_mobile;
    null_qdot.segment(act_idx.mani_start, fr3_husky_model_updater_.manipulator_dof_) = null_qdot_mani;

    fr3_husky_model_updater_.robot_controller_->CLIKStep(
        ee_data_,
        qdot_mobile,
        fr3_husky_model_updater_.qdot_desired_total_,
        null_qdot
    );

    fr3_husky_model_updater_.q_desired_total_ =
        fr3_husky_model_updater_.q_total_ +
        fr3_husky_model_updater_.dt_ * fr3_husky_model_updater_.qdot_desired_total_;

    fr3_husky_model_updater_.torque_desired_total_ =
        fr3_husky_model_updater_.robot_controller_->moveManipulatorJointTorqueStep(
            fr3_husky_model_updater_.q_desired_total_,
            fr3_husky_model_updater_.qdot_desired_total_,
            false
        );

    fr3_husky_model_updater_.wheel_vel_desired_.setZero();

    fr3_husky_model_updater_.writeCommand(
        fr3_husky_model_updater_.torque_desired_total_ - fr3_husky_model_updater_.g_total_,
        fr3_husky_model_updater_.wheel_vel_desired_
    );


    if (s >= 1.0 && all_reached) return ComputeResult::SUCCEEDED;
    return ComputeResult::RUNNING;
}


void TaskSpaceDeltaMove::onStop(StopReason reason)
{
    if (reason != StopReason::SUCCEEDED)
        model_updater_.haltCommands();

    const char* rs = (reason == StopReason::CANCELED)  ? "canceled"  :
                     (reason == StopReason::SUCCEEDED) ? "succeeded" :
                     (reason == StopReason::ABORTED)   ? "aborted"   : "none";

    RCLCPP_INFO(node_->get_logger(), "[%s] stopped (%s)", name_.c_str(), rs);
}

TaskSpaceDeltaMove::ResultPtr TaskSpaceDeltaMove::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();

    if (reason == StopReason::SUCCEEDED)
    {
        result->success = true;
        result->message = "Task-space target reached";
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
        result->message = "Aborted";
        result->error_code = result_error_code_;
    }

    return result;
}

REGISTER_FR3_HUSKY_ACTION_SERVER(TaskSpaceDeltaMove, "fr3_husky_task_space_delta_move")

}  // namespace fr3_husky_controller::servers::fr3_husky