#include <fr3_husky_controller/servers/fr3_husky/husky_move_action_server.hpp>

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

}  // namespace

HuskyMove::HuskyMove(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_husky_model_updater_(getFR3HuskyModelUpdater(model_updater, name))
{
    cmd_vel_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 1, std::bind(&HuskyMove::subCMDVelCallback, this, std::placeholders::_1));
    joy_cmd_vel_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(
        "/joy_teleop/cmd_vel", 1, std::bind(&HuskyMove::subCMDVelCallback, this, std::placeholders::_1));
    fr3_husky_cmd_vel_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(
        "/fr3_husky/cmd_vel", 1, std::bind(&HuskyMove::subCMDVelCallback, this, std::placeholders::_1));

    cmd_vel_.setZero();
    q_hold_.setZero(model_updater.manipulator_dof_);

    RCLCPP_INFO(node_->get_logger(), "[%s] HuskyMove created", name_.c_str());
}

bool HuskyMove::acceptGoal(const ActionT::Goal& goal)
{
    return true;
}

void HuskyMove::onGoalAccepted(const ActionT::Goal& goal)
{
}

void HuskyMove::onStart()
{
    cmd_vel_.setZero();
    q_hold_ = fr3_husky_model_updater_.q_total_;
    RCLCPP_INFO(node_->get_logger(), "[%s] started", name_.c_str());
}

HuskyMove::ComputeResult HuskyMove::compute(const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/)
{
    fr3_husky_model_updater_.wheel_vel_desired_ = fr3_husky_model_updater_.robot_controller_->mobi.VelocityCommand(cmd_vel_);
    if(model_updater_.HasPositionCommandInterface())
    {
        fr3_husky_model_updater_.q_desired_total_ = q_hold_;
        fr3_husky_model_updater_.writeCommand(fr3_husky_model_updater_.q_desired_total_,
                                              fr3_husky_model_updater_.wheel_vel_desired_);
        return ComputeResult::RUNNING;                            
    }
    else if(model_updater_.HasVelocityCommandInterface())
    {
        fr3_husky_model_updater_.qdot_desired_total_.setZero();
        fr3_husky_model_updater_.writeCommand(fr3_husky_model_updater_.qdot_desired_total_,
                                              fr3_husky_model_updater_.wheel_vel_desired_);
        return ComputeResult::RUNNING;   
    }
    else
    {
        fr3_husky_model_updater_.torque_desired_total_
            = fr3_husky_model_updater_.robot_controller_->mani.moveJointTorqueStep(q_hold_,
                                                                                    Eigen::VectorXd::Zero(model_updater_.manipulator_dof_),
                                                                                    false);
        fr3_husky_model_updater_.writeCommand(fr3_husky_model_updater_.torque_desired_total_ - fr3_husky_model_updater_.robot_data_->mani.getGravity(),
                                              fr3_husky_model_updater_.wheel_vel_desired_);
        return ComputeResult::RUNNING;   
    }
}

void HuskyMove::onStop(StopReason reason)
{
    model_updater_.haltCommands();

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

HuskyMove::ResultPtr HuskyMove::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();
    result->is_completed = true;
    return result;
}

void HuskyMove::subCMDVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    Eigen::Vector3d vel(msg->linear.x, msg->linear.y, msg->angular.z);
    cmd_vel_ = dyros_math::lowPassFilter(vel, cmd_vel_, 0.001, 0.002);

}


// Register this server into global registry (executed when this TU is linked)
REGISTER_FR3_HUSKY_ACTION_SERVER(HuskyMove, "fr3_husky_husky_move")

}  // namespace fr3_husky_controller::servers::fr3_husky

/*
# send goal
ros2 action send_goal /fr3_husky_husky_move fr3_husky_msgs/action/MobileMove "{}"
*/
