#include <fr3_husky_controller/servers/fr3/gripper_command_action_server.hpp>

#include <mujoco/mujoco.h>

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <vector>


namespace fr3_husky_controller::servers::fr3
{

namespace
{
FR3ModelUpdater& getFR3ModelUpdater(ModelUpdaterBase& model_updater, const std::string& server_name)
{
    auto* fr3_model_updater = dynamic_cast<FR3ModelUpdater*>(&model_updater);
    if (!fr3_model_updater)
    {
        throw std::runtime_error("[" + server_name + "] requires FR3ModelUpdater");
    }
    return *fr3_model_updater;
}

}  // namespace

GripperCommand::GripperCommand(
    const std::string& name,
    const NodePtr& node,
    ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_model_updater_(getFR3ModelUpdater(model_updater, name))
{
    mode_ = ServerMode::CONTROLLER;

    RCLCPP_INFO(node_->get_logger(), "[%s] GripperCommand created", name_.c_str());
}

bool GripperCommand::acceptGoal(const ActionT::Goal& goal)
{
    if (goal.arm_names != "left" && goal.arm_names != "right" && goal.arm_names != "both")
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: invalid arm_names '%s'", name_.c_str(), goal.arm_names.c_str());
        return false;
    }
    if (goal.command != "open" && goal.command != "grasp" && goal.command != "move")
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: invalid command '%s'", name_.c_str(), goal.command.c_str());
        return false;
    }
    if (goal.width < 0.0 || goal.width > 0.08)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: width %.4f out of range [0.0, 0.08]", name_.c_str(), goal.width);
        return false;
    }
    if (goal.speed <= 0.0 || goal.speed > 0.3)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: speed %.4f out of range (0.0, 0.3]", name_.c_str(), goal.speed);
        return false;
    }
    if (goal.command == "grasp" && (goal.force <= 0.0 || goal.force > 140.0))
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: force %.2f out of range (0.0, 140.0]", name_.c_str(), goal.force);
        return false;
    }
    if (goal.epsilon_inner < 0.0 || goal.epsilon_inner > 0.08 ||
        goal.epsilon_outer < 0.0 || goal.epsilon_outer > 0.08)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: epsilon out of range [0.0, 0.08]", name_.c_str());
        return false;
    }
    if (goal.use_weld && !goal.weld_name.empty() && goal.weld_name != "weld_right_tcp")
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: unsupported weld_name '%s'", name_.c_str(), goal.weld_name.c_str());
        return false;
    }
    return true;
}

void GripperCommand::onGoalAccepted(const ActionT::Goal& goal)
{
    goal_ = goal;
    has_goal_ = true;
    command_sent_ = false;
    command_success_ = false;
    result_message_.clear();
}

GripperCommand::ComputeResult GripperCommand::compute(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/)
{
    if (!has_goal_)
    {
        return ComputeResult::RUNNING;
    }

    if (command_sent_)
    {
        has_goal_ = false;
        return command_success_ ? ComputeResult::SUCCEEDED : ComputeResult::ABORTED;
    }

    command_sent_ = true;

    const std::vector<std::string> arms =
        (goal_.arm_names == "both") ? std::vector<std::string>{"left", "right"}
                                    : std::vector<std::string>{goal_.arm_names};


    for (const auto& arm : arms)
    {
        if (!runCommandForArm(arm))
        {
            command_success_ = false;
            has_goal_ = false;
            return ComputeResult::ABORTED;
        }
    }

    command_success_ = true;
    result_message_ = "Gripper command sent: command=" + goal_.command +
                      ", arm_names=" + goal_.arm_names +
                      ", width=" + std::to_string(goal_.width);
    has_goal_ = false;
    return ComputeResult::SUCCEEDED;
}

bool GripperCommand::runCommandForArm(const std::string& arm)
{
    bool ok = false;

    if (goal_.command == "open")
    {
        ok = fr3_model_updater_.GripperOpen(arm, goal_.speed);
    }
    else if (goal_.command == "grasp")
    {
        ok = fr3_model_updater_.GripperGrasp(
            arm,
            goal_.width,
            goal_.speed,
            goal_.force,
            {goal_.epsilon_inner, goal_.epsilon_outer});

    }
    else
    {
        ok = fr3_model_updater_.GripperMove(arm, goal_.width, goal_.speed);
    }

    if (!ok)
    {
        result_message_ = "Failed to send gripper command for arm=" + arm;
        return false;
    }
    return true;
}


GripperCommand::ResultPtr GripperCommand::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();
    result->success = (reason == StopReason::SUCCEEDED) && command_success_;
    result->message = result_message_;
    if (result->message.empty())
    {
        result->message = result->success ? "Gripper command sent." : "Gripper command failed.";
    }
    return result;
}

REGISTER_FR3_ACTION_SERVER(GripperCommand, "fr3_gripper_command")

}  // namespace fr3_husky_controller::servers::fr3
