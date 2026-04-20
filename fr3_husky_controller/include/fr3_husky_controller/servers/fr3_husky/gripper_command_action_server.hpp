#pragma once

#include <string>

#include <fr3_husky_msgs/action/gripper_command.hpp>

#include <fr3_husky_controller/model/fr3_husky_model_updater.hpp>
#include <fr3_husky_controller/servers/action_server_base.hpp>

namespace fr3_husky_controller::servers::fr3_husky
{

class GripperCommand final : public ActionServerBase<fr3_husky_msgs::action::GripperCommand>
{
public:
    using ActionT       = fr3_husky_msgs::action::GripperCommand;
    using Base          = ActionServerBase<ActionT>;
    using ComputeResult = typename Base::ComputeResult;
    using StopReason    = typename Base::StopReason;
    using ResultPtr     = typename Base::ResultPtr;

    GripperCommand(const std::string& name, const NodePtr& node,
                   ModelUpdaterBase& model_updater);

private:
    bool          acceptGoal(const ActionT::Goal& goal) override;
    void          onGoalAccepted(const ActionT::Goal& goal) override;
    ComputeResult compute(const rclcpp::Time& time,
                          const rclcpp::Duration& period) override;
    ResultPtr     makeResult(StopReason reason) override;

    bool runCommandForArm(const std::string& arm);
    bool setWeldActive(bool active);

    FR3HuskyModelUpdater& fr3_husky_model_updater_;

    ActionT::Goal goal_;
    bool has_goal_{false};
    bool command_sent_{false};
    bool command_success_{false};
    std::string result_message_;
};

}  // namespace fr3_husky_controller::servers::fr3_husky
