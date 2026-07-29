// contact_guarded_motion_action_server.cpp

#include <fr3_husky_controller/servers/fr3/contact_guarded_motion_action_server.hpp>

#include <chrono>
#include <stdexcept>
#include <string>

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

}

ContactGuardedMotion::ContactGuardedMotion(
    const std::string& name,
    const NodePtr& node,
    ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_model_updater_(getFR3ModelUpdater(model_updater, name))
{
    mode_ = ServerMode::TASK;
    RCLCPP_INFO(node_->get_logger(), "[%s] ContactGuardedMotion created", name_.c_str());
}

bool ContactGuardedMotion::acceptGoal(const ActionT::Goal& goal)
{
    if (goal.ee_names.empty())
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: ee_names is empty", name_.c_str());
        return false;
    }

    if (goal.ee_names.size() != goal.target_poses.size())
    {
        RCLCPP_WARN(node_->get_logger(),
                    "[%s] Reject: ee_names.size() != target_poses.size()",
                    name_.c_str());
        return false;
    }

    for (const auto& ee_name : goal.ee_names)
    {
        if (!fr3_model_updater_.robot_data_->hasLinkFrame(ee_name))
        {
            RCLCPP_WARN(node_->get_logger(), "[%s] Reject: unknown ee_name: %s", name_.c_str(), ee_name.c_str());
            return false;
        }
    }

    return true;
}

void ContactGuardedMotion::onGoalAccepted(const ActionT::Goal& goal)
{
    ee_names_ = goal.ee_names;
    target_poses_.clear();

    for (const auto& pose_msg : goal.target_poses)
        target_poses_.push_back(poseMsgToAffine(pose_msg));

    duration_ = goal.duration > 0.0 ? goal.duration : 3.0;
    pos_tolerance_ = goal.pos_tolerance > 0.0 ? goal.pos_tolerance : 0.01;
    ori_tolerance_ = goal.ori_tolerance > 0.0 ? goal.ori_tolerance : 0.05;

    result_error_code_ = 0;
    start_time_set_ = false;

    contact_torque_bias_set_ = false;
    contact_detected_ = false;
    contact_count_ = 0;

    requestActivate();

    RCLCPP_INFO(node_->get_logger(),
                "[%s] goal accepted: %zu EE targets",
                name_.c_str(), ee_names_.size());
}

void ContactGuardedMotion::onStart()
{
    fr3_model_updater_.setInitFromCurrent();

    ee_data_.clear();
    start_poses_.clear();

    for (const auto& ee_name : ee_names_)
    {
        ee_data_[ee_name] = drc::TaskSpaceData::Zero();
        ee_data_[ee_name].x = fr3_model_updater_.robot_data_->getPose(ee_name);
        ee_data_[ee_name].xdot = fr3_model_updater_.robot_data_->getVelocity(ee_name);
        ee_data_[ee_name].xddot.setZero();
        ee_data_[ee_name].setInit();
        ee_data_[ee_name].setDesired();
        ee_data_[ee_name].xdot_desired.setZero();

        start_poses_.push_back(ee_data_[ee_name].x);
    }

    for (const auto& robot_name : fr3_model_updater_.robot_names_) {
        start_joint_torque_[robot_name] = fr3_model_updater_.getJointTorque(robot_name);
        contact_torque_threshold_[robot_name] = Eigen::VectorXd::Constant(start_joint_torque_[robot_name].size(), 8.0);
    }


    contact_torque_bias_set_ = true;
    contact_detected_ = false;
    contact_count_ = 0;

    start_time_set_ = false;
    RCLCPP_INFO(node_->get_logger(), "[%s] started", name_.c_str());
}



bool ContactGuardedMotion::isContactDetected(const rclcpp::Time& time)
{
    if (!contact_torque_bias_set_ || !start_time_set_) return false;

    const double elapsed = (time - start_time_).seconds();
    if (elapsed < contact_detection_start_time_) return false;

    bool threshold_exceeded = false;
    std::string exceeded_ee_name;
    int exceeded_idx = -1;
    double exceeded_value = 0.0;
    double exceeded_threshold = 0.0;


    for (const auto& robot_name : fr3_model_updater_.robot_names_) 
    {
        const auto start_it = start_joint_torque_.find(robot_name);
        const auto threshold_it = contact_torque_threshold_.find(robot_name);

        bool is_error = false;
        is_error = is_error || start_it == start_joint_torque_.end();
        is_error = is_error || threshold_it == contact_torque_threshold_.end();
        if (is_error) {
            result_error_code_ = 2;
            return false;
        }

        const Eigen::VectorXd current_joint_torque = fr3_model_updater_.getJointTorque(robot_name);
        const Eigen::VectorXd& start_joint_torque = start_it->second;
        const Eigen::VectorXd& contact_torque_threshold = threshold_it->second;


        is_error = is_error || current_joint_torque.size() != start_joint_torque.size();
        is_error = is_error || contact_torque_threshold.size() != start_joint_torque.size();
        if (is_error) {
            result_error_code_ = 2;
            return false;
        }
        

        const Eigen::VectorXd delta_torque = current_joint_torque - start_joint_torque;

        for (int i = 0; i < delta_torque.size(); ++i)
        {
            const double abs_tau = std::abs(delta_torque[i]);
            const double threshold = contact_torque_threshold[i];

            if (abs_tau > threshold)
            {
                threshold_exceeded = true;
                exceeded_ee_name = robot_name;
                exceeded_idx = i;
                exceeded_value = delta_torque[i];
                exceeded_threshold = threshold;
                break;
            }
        }

        if (threshold_exceeded) break;
    }

    contact_count_ = threshold_exceeded ? contact_count_ + 1 : 0;
    if (contact_count_ >= contact_debounce_count_)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] contact detected: ee_name=%s joint=%d delta_tau=%.3f threshold=%.3f", name_.c_str(), exceeded_ee_name.c_str(), exceeded_idx, exceeded_value, exceeded_threshold);
        return true;
    }

    return false;
}



Eigen::Affine3d ContactGuardedMotion::poseMsgToAffine(const geometry_msgs::msg::Pose& msg)
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

double ContactGuardedMotion::orientationError(const Eigen::Matrix3d& R_des, const Eigen::Matrix3d& R_cur)
{
    Eigen::Matrix3d R_err = R_des * R_cur.transpose();
    Eigen::AngleAxisd aa(R_err);
    return std::abs(aa.angle());
}



Eigen::Vector6d ContactGuardedMotion::computeTargetVelocity(
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


ContactGuardedMotion::ComputeResult ContactGuardedMotion::compute(
    const rclcpp::Time& time,
    const rclcpp::Duration& /*period*/)
{
    for (auto& [ee_name, ee_data] : ee_data_)
    {
        ee_data.x = fr3_model_updater_.robot_data_->getPose(ee_name);
        ee_data.xdot = fr3_model_updater_.robot_data_->getVelocity(ee_name);
        ee_data.xddot.setZero();
    }

    if (!start_time_set_)
    {
        start_time_ = time;
        start_time_set_ = true;
    }

    // if (isContactDetected(time))
    // {
    //     contact_detected_ = true;
    //     result_error_code_ = 0;
    //     fr3_model_updater_.haltCommands();
    //     RCLCPP_WARN(node_->get_logger(), "[%s] stopped by contact guard", name_.c_str());
    //     return ComputeResult::SUCCEEDED;
    // }


    const double elapsed = (time - start_time_).seconds();
    const double s = std::clamp(elapsed / duration_, 0.0, 1.0);
    const double dt = fr3_model_updater_.dt_;

    std::vector<double> pos_errors;
    std::vector<double> ori_errors;
    bool all_reached = true;

    for (size_t i = 0; i < ee_names_.size(); ++i)
    {
        const auto& ee_name = ee_names_[i];
        const Eigen::Affine3d& T0 = start_poses_[i];
        const Eigen::Affine3d& T1 = target_poses_[i];

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


    fr3_model_updater_.robot_controller_->CLIKStep(ee_data_, fr3_model_updater_.qdot_desired_total_);

    fr3_model_updater_.q_desired_total_ = fr3_model_updater_.q_total_ +
                                                                fr3_model_updater_.dt_ * fr3_model_updater_.qdot_desired_total_;
    
    fr3_model_updater_.torque_desired_total_ = fr3_model_updater_.robot_controller_->moveJointTorqueStep(fr3_model_updater_.q_desired_total_,
                                                                                                            fr3_model_updater_.qdot_desired_total_,
                                                                                                            false);

    fr3_model_updater_.writeCommand(fr3_model_updater_.torque_desired_total_ - fr3_model_updater_.g_total_); // robot_controller automatically add gravity force


    if (s >= 1.0 && all_reached) return ComputeResult::SUCCEEDED;
    return ComputeResult::RUNNING;
}


void ContactGuardedMotion::onStop(StopReason reason)
{
    if (reason != StopReason::SUCCEEDED)
        fr3_model_updater_.haltCommands();

    const char* rs = (reason == StopReason::CANCELED)  ? "canceled"  :
                     (reason == StopReason::SUCCEEDED) ? "succeeded" :
                     (reason == StopReason::ABORTED)   ? "aborted"   : "none";

    RCLCPP_INFO(node_->get_logger(), "[%s] stopped (%s)", name_.c_str(), rs);
}

ContactGuardedMotion::ResultPtr ContactGuardedMotion::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();

    if (reason == StopReason::SUCCEEDED)
    {
        result->success = true;
        result->message = "Succeeded";
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

REGISTER_FR3_ACTION_SERVER(ContactGuardedMotion, "fr3_contact_guarded_motion")

}  // namespace fr3_controller::servers::fr3