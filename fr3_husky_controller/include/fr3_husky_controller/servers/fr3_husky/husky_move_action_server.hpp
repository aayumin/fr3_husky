#pragma once

#include <memory>

#include <fr3_husky_msgs/action/mobile_move.hpp>

#include <fr3_husky_controller/servers/action_server_base.hpp>
#include <fr3_husky_controller/model/fr3_husky_model_updater.hpp>
#include<fr3_husky_controller/utils/dyros_math.h>

#include <geometry_msgs/msg/twist.hpp>

namespace fr3_husky_controller::servers::fr3_husky
{

class HuskyMove final : public ActionServerBase<fr3_husky_msgs::action::MobileMove>
{
public:
    using ActionT = fr3_husky_msgs::action::MobileMove;
    using Base = ActionServerBase<ActionT>;
    using ComputeResult = typename Base::ComputeResult;
    using StopReason = typename Base::StopReason;
    using ResultPtr = typename Base::ResultPtr;

    HuskyMove(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater);
    ~HuskyMove() override = default;

    int priority() const override { return 0; }
    bool allowPreemption() const override { return false; }  // if true, a new incoming goal preempts (aborts) the current one

private:
    bool acceptGoal(const ActionT::Goal& goal) override;
    void onGoalAccepted(const ActionT::Goal& goal) override;
    void onStart() override;
    ComputeResult compute(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    void onStop(StopReason reason) override;
    ResultPtr makeResult(StopReason reason) override;

private:
    FR3HuskyModelUpdater& fr3_husky_model_updater_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr joy_cmd_vel_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr fr3_husky_cmd_vel_sub_;
    Eigen::Vector3d cmd_vel_; // vx, vy, w wrt base frame

    Eigen::VectorXd q_hold_; // initial manipulator joint when start action server

    void subCMDVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);

};

}  // namespace fr3_husky_controller::servers::fr3_husky
