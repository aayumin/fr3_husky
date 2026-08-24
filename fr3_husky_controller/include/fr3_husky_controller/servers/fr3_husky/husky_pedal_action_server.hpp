#pragma once

#include <memory>
#include <mutex>

#include <fr3_husky_msgs/action/husky_pedal.hpp>

#include <fr3_husky_controller/model/fr3_husky_model_updater.hpp>
#include <fr3_husky_controller/servers/action_server_base.hpp>

#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/joy.hpp>

namespace fr3_husky_controller::servers::fr3_husky
{

class HuskyPedal final : public ActionServerBase<fr3_husky_msgs::action::HuskyPedal>
{
public:
    using ActionT = fr3_husky_msgs::action::HuskyPedal;
    using Base = ActionServerBase<ActionT>;
    using ComputeResult = typename Base::ComputeResult;
    using StopReason = typename Base::StopReason;
    using ResultPtr = typename Base::ResultPtr;

    HuskyPedal(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater);
    ~HuskyPedal() override = default;

    int priority() const override { return 0; }
    bool allowPreemption() const override { return true; }
    bool isActive() const override { return enabled_; }

private:
    bool acceptGoal(const ActionT::Goal& goal) override;
    void onGoalAccepted(const ActionT::Goal& goal) override;
    void onStart() override;
    ComputeResult compute(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    void onStop(StopReason reason) override;
    ResultPtr makeResult(StopReason reason) override;

    void subPedalCallback(const sensor_msgs::msg::Joy::SharedPtr msg);
    Eigen::Vector3d computeBaseVelocityFromPedal(const sensor_msgs::msg::Joy& msg);
    void applyEnabledState(bool latch_hold);
    void resetCommand();

    FR3HuskyModelUpdater& fr3_husky_model_updater_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr pedal_sub_;

    mutable std::mutex cmd_vel_mutex_;
    Eigen::Vector3d cmd_vel_;
    Eigen::VectorXd q_hold_;

    bool enabled_{false};
    bool goal_update_pending_{false};

    int axis_left_{0};
    int axis_right_{1};
    int axis_yaw_mag_{2};
    double scale_linear_{0.5};
    double scale_angular_{0.5};
    double deadzone_pedal_{0.05};
    double deadzone_yaw_mag_{0.05};
    int enable_button_{-1};
    std::string pedal_topic_{"joy"};

    bool warned_axes_{false};
    bool warned_buttons_{false};
};

}  // namespace fr3_husky_controller::servers::fr3_husky
