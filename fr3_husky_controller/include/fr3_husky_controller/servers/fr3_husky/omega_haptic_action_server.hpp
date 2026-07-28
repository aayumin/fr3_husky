#pragma once

#include <memory>
#include <vector>

#include <fr3_husky_msgs/action/omega_haptic.hpp>

#include <fr3_husky_controller/servers/action_server_base.hpp>
#include <fr3_husky_controller/model/fr3_husky_model_updater.hpp>
#include <fr3_husky_controller/utils/dyros_math.h>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int8_multi_array.hpp>

/*
Teleoperation modes:
    1) Base Mode       : joystick -> mobile base
    2) Arm Mode        : omega -> manipulator (EE in base frame)
    3) Whole-Body Mode : omega -> mobile base + manipulator (EE in world/task frame)
    4) Dual Mode       : omega + joystick -> simultaneous EE and base control
*/
namespace fr3_husky_controller::servers::fr3_husky
{

class OmegaHaptic final : public ActionServerBase<fr3_husky_msgs::action::OmegaHaptic>
{
public:
    using ActionT = fr3_husky_msgs::action::OmegaHaptic;
    using Base = ActionServerBase<ActionT>;
    using ComputeResult = typename Base::ComputeResult;
    using StopReason = typename Base::StopReason;
    using ResultPtr = typename Base::ResultPtr;

    OmegaHaptic(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater);
    ~OmegaHaptic() override = default;

    int priority() const override { return 7; }
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

private:
    // publishers & subscribers
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr  pose_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr ori_encoder_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr        twist_sub_;
    rclcpp::Subscription<std_msgs::msg::Int8MultiArray>::SharedPtr    button_state_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr        base_vel_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr        joy_base_vel_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr        fr3_husky_base_vel_sub_;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr   wrench_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr         force_pub_;

    void subPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void subOriEncoderCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    void subTwistCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void subButtonCallback(const std_msgs::msg::Int8MultiArray::SharedPtr msg);
    void subBaseVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);


    // haptic state data
    Eigen::Affine3d haptic_pose_;
    Eigen::Vector6d haptic_vel_;
    Eigen::Vector3d haptic_ori_encoder_;
    Eigen::Affine3d haptic_pose_init_;
    bool button0_state_;
    bool is_mouse_mode_on_{false};
    Eigen::Matrix3d haptic_base2robot_base_;

    // joy state data
    Eigen::Vector3d joy_cmd_vel_;

    // robot data
    std::map<std::string, drc::TaskSpaceData> ee_data; // world to EE
    Eigen::Affine3d T_world2mobi_base_;      // world to mobile base
    Eigen::Affine3d T_world2mobi_base_init_; // world to mobile base

    // action goal data
    int control_mode_; // 0: CLIK, 1: OSF, 2:QPIK, 3:QPID
    int teleop_mode_;  // 0: Base, 1: Arm, 2: Wholebody, 3:Dual
    std::string control_ee_name_; // EE name for tracking haptic twist 
    bool move_ori_;
    double haptic_pos_multiplier_;
    double haptic_ori_multiplier_;
    double haptic_lin_vel_multiplier_;
    double haptic_ang_vel_multiplier_;
    double dual_mobile_null_torque_gain_{100.0};

    std::mutex haptic_pose_mutex_;
    std::mutex haptic_ori_encoder_mutex_;
    std::mutex haptic_vel_mutex_;
    std::mutex button0_state_mutex_;
    std::mutex joy_cmd_mutex_;
};

}  // namespace fr3_husky_controller::servers::fr3_husky
