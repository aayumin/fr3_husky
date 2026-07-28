#pragma once

#include <memory>

#include <action_msgs/msg/goal_status.hpp>
#include <action_msgs/msg/goal_status_array.hpp>
#include <fr3_husky_msgs/action/move_to_joint.hpp>
#include <fr3_husky_msgs/action/vive_tracker.hpp>

#include <fr3_husky_controller/servers/action_server_base.hpp>
#include <fr3_husky_controller/model/fr3_husky_model_updater.hpp>
#include <fr3_husky_controller/utils/dyros_math.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/joy.hpp>

#define NUM_TRACKERS    3 // left, right, head
#define NUM_CONTROLLERS 2 // left, right (only Focus3 controllers)
#define IDX_LEFT_CON    0 // index of left controller pose in "tracker_pose" topic
#define IDX_RIGHT_CON   1 // index of right controller pose in "tracker_pose" topic
#define IDX_HEAD_CON    2 // index of HMD head pose in "tracker_pose" topic

#define NUM_BUTTONS        5 // number of buttons in Focu3 controller [trigger, grip, a, b, joy]
#define IDX_TRIGGER_BUTTON 0 // index of Trigger button in "l/rhand_joy" topic
#define IDX_GRIP_BUTTON    1 // index of Grip button in "l/rhand_joy" topic
#define IDX_A_BUTTON       2 // index of A button in "l/rhand_joy" topic
#define IDX_B_BUTTON       3 // index of B button in "l/rhand_joy" topic
#define IDX_JOY_BUTTON     4 // index of Joy button in "l/rhand_joy" topic

#define NUM_AXES           4 // number of axes in Focus3 controller [trigger, grip, joy_x, joy_y]
#define IDX_TRIGGER_AXES   0 // index of Trigger axes in "l/rhand_joy" topic (range: 0 ~ 1)
#define IDX_GRIP_AXES      1 // index of Grip axes in "l/rhand_joy" topic (range: 0 ~ 1)
#define IDX_JOY_X_AXES     2 // index of Joy x axes in "l/rhand_joy" topic (range: -1 ~ 1)
#define IDX_JOY_Y_AXES     3 // index of Joy y axes in "l/rhand_joy" topic (range: -1 ~ 1)

/*
Teleoperation modes:
    1) Base Mode       : joystick -> mobile base
    2) Arm Mode        : omega -> manipulator (EE in base frame)
    3) Whole-Body Mode : omega -> mobile base + manipulator (EE in world/task frame)
    4) Dual Mode       : omega + joystick -> simultaneous EE and base control
*/

namespace fr3_husky_controller::servers::fr3_husky
{

class ViveTracker final : public ActionServerBase<fr3_husky_msgs::action::ViveTracker>
{
public:
    using ActionT = fr3_husky_msgs::action::ViveTracker;
    using Base = ActionServerBase<ActionT>;
    using ComputeResult = typename Base::ComputeResult;
    using StopReason = typename Base::StopReason;
    using ResultPtr = typename Base::ResultPtr;

    ViveTracker(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater);
    ~ViveTracker() override = default;

    int priority() const override { return 7; }
    bool allowPreemption() const override { return true; }  // if true, a new incoming goal preempts (aborts) the current one

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
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr  pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr l_joy_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr r_joy_sub_;

    void subPoseCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg);
    void subLJoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);
    void subRJoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);

    // vive controller state data
    std::vector<Eigen::Affine3d> controller_poses_;      // left, right, head
    std::vector<Eigen::Affine3d> controller_poses_init_; // left, right, head
    std::vector<std::vector<bool>> button_states_;       // [left, right][trigger, grip, a, b, joy]
    std::vector<std::vector<bool>> prev_button_states_;  // previous button states for edge detection
    std::vector<std::vector<double>> axe_states_;        // [left, right][trigger, grip, joy_x, joy_y]

    // mutex
    std::mutex tracker_pose_mutex_;
    std::mutex joy_mutex_;

    // states for mode
    std::vector<bool> is_mouse_mode_on_{false, false};
    bool is_initialize_mode_on_{false};
    std::vector<bool> is_gripper_mode_on_{false, false};
    int mobile_mode_{0}; // 0: Stop mode, 1: left controller, 2: right controller

    // robot data
    std::vector<Eigen::Matrix3d> tracker_base2robot_base_;
    std::map<std::string, drc::TaskSpaceData> ee_data_; // world to EE
    Eigen::Affine3d T_world2mobi_base_;                 // world to mobile base
    Eigen::Affine3d T_world2mobi_base_init_;            // world to mobile base

    // action goal data
    int control_mode_;                     // 0: CLIK, 1: OSF, 2:QPIK, 3:QPID
    int teleop_mode_;                      // 0: Base, 1: Arm, 2: Wholebody, 3:Dual
    std::string left_controller_ee_name_;  // EE name for tracking left vive controller
    std::string right_controller_ee_name_; // EE name for tracking right vive controller
    bool move_ori_;
    double controller_pos_multiplier_;
    double controller_ori_multiplier_;
    double dual_mobile_null_torque_gain_{100.0};

    // initialize mode
    const Eigen::Vector<double, FR3_DOF> HomePose{0., -0.785, 0.0, -2.356, 0.0, 1.571, 0.785};
    ActionT::Goal saved_vive_goal_{};  // saved goal params for auto-resume after init
    using MoveToJointAction = fr3_husky_msgs::action::MoveToJoint;
    rclcpp_action::Client<MoveToJointAction>::SharedPtr move_to_joint_client_;
    rclcpp_action::Client<ActionT>::SharedPtr vt_self_client_;  // self-client for resume
    rclcpp::Subscription<action_msgs::msg::GoalStatusArray>::SharedPtr jtc_status_sub_;
    std::atomic<bool> waiting_for_jtc_{false};
};

}  // namespace fr3_husky_controller::servers::fr3_husky
