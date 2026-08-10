#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <fr3_husky_msgs/action/robomimic_move.hpp>

#include <fr3_husky_controller/servers/action_server_base.hpp>
#include <fr3_husky_controller/model/fr3_husky_model_updater.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>


namespace fr3_husky_controller::servers::fr3_husky
{

class RobomimicMove final
    : public ActionServerBase<fr3_husky_msgs::action::RobomimicMove>
{
public:
    using ActionT = fr3_husky_msgs::action::RobomimicMove;
    using Base = ActionServerBase<ActionT>;
    using ComputeResult = typename Base::ComputeResult;
    using StopReason = typename Base::StopReason;
    using ResultPtr = typename Base::ResultPtr;

    RobomimicMove(
        const std::string& name,
        const NodePtr& node,
        ModelUpdaterBase& model_updater);

    ~RobomimicMove() override = default;

    int priority() const override { return 7; }
    bool allowPreemption() const override { return true; }

private:
    bool acceptGoal(const ActionT::Goal& goal) override;
    void onGoalAccepted(const ActionT::Goal& goal) override;
    void onStart() override;
    ComputeResult compute(
        const rclcpp::Time& time,
        const rclcpp::Duration& period) override;
    void onStop(StopReason reason) override;
    ResultPtr makeResult(StopReason reason) override;

private:
    void onDeltaAction(
        const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    void publishInferenceActive(bool active);
    void publishObservation();

    static double nowSec()
    {
        using namespace std::chrono;
        return duration<double>(
            steady_clock::now().time_since_epoch()).count();
    }

private:
    FR3HuskyModelUpdater& fr3_husky_model_updater_;

    std::mutex command_mutex_;

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr delta_action_sub_;

    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr inference_active_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr eef_pose_pub_;

    std::map<std::string, drc::TaskSpaceData> ee_data_;

    std::string controller_ee_name_;
    std::string robot_name_;

    int control_mode_{0};

    Eigen::Affine3d x_goal_{Eigen::Affine3d::Identity()};
    Eigen::Affine3d x_target_{Eigen::Affine3d::Identity()};

    std::vector<double> latest_action_;

    bool has_command_{false};
    bool has_new_command_{false};

    double last_command_time_{0.0};
    double command_timeout_{0.5};

    double position_scale_{1.0};
    double rotation_scale_{1.0};

    bool gripper_closed_{false};
    double previous_gripper_command_{0.0};

    int step_count_{0};
};

}  // namespace fr3_husky_controller::servers::fr3_husky