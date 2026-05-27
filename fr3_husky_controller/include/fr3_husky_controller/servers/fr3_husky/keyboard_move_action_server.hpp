#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <type_traits>
#include <deque>
#include <algorithm>
#include <cmath>


#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <mutex>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>  

#include <action_msgs/msg/goal_status.hpp>
#include <action_msgs/msg/goal_status_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <fr3_husky_msgs/action/move_to_joint.hpp>
#include <fr3_husky_msgs/action/keyboard_move.hpp>

#include <fr3_husky_controller/servers/action_server_base.hpp>
#include <fr3_husky_controller/model/fr3_husky_model_updater.hpp>
#include <fr3_husky_controller/utils/dyros_math.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#define IDX_LEFT_CON    0 // index of left
#define IDX_RIGHT_CON   1 // index of right

namespace fr3_husky_controller::servers::fr3_husky
{
    /*
    Left Translation:  up down left right forward backward.
    Left Rotation:  roll pitch yaw
    Left Gripper: toggle

    Right Translation:  up down left right forward backward.
    Right Rotation: roll pitch yaw
    Right Gripper: toggle
    */

class KeyboardMove final : public ActionServerBase<fr3_husky_msgs::action::KeyboardMove>
{
public:
    using ActionT = fr3_husky_msgs::action::KeyboardMove;
    using Base = ActionServerBase<ActionT>;
    using ComputeResult = typename Base::ComputeResult;
    using StopReason = typename Base::StopReason;
    using ResultPtr = typename Base::ResultPtr;


    KeyboardMove(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater);
    ~KeyboardMove() override = default;

    int priority() const override { return 7; }
    bool allowPreemption() const override { return true; }

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

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr keyboard_sub_;
    void clearKeyboardFlags();
    void onKeyboardCommand(const std_msgs::msg::String::SharedPtr msg);



    std::mutex lock_;
    
    

    // manipulator keypressed
    bool keypress_left_arm_left_{false};
    bool keypress_left_arm_right_{false};
    bool keypress_left_arm_forward_{false};
    bool keypress_left_arm_backward_{false};
    bool keypress_left_arm_upward_{false};
    bool keypress_left_arm_downward_{false};

    
    bool keypress_right_arm_left_{false};
    bool keypress_right_arm_right_{false};  
    bool keypress_right_arm_forward_{false};
    bool keypress_right_arm_backward_{false};
    bool keypress_right_arm_upward_{false};
    bool keypress_right_arm_downward_{false};
    
    // gripper
    bool prev_l_gtoggle_{false};
    bool prev_r_gtoggle_{false};
    bool keypress_left_arm_gripper_{false};
    bool keypress_right_arm_gripper_{false};

    // key active
    bool user_cmd_prev_{false};
    bool hold_latched_{false};
    double last_key_time_{0.0};
    double key_timeout_{0.25};
    static double now_sec_() {
        using namespace std::chrono;
        return duration<double>(steady_clock::now().time_since_epoch()).count();
    }

    // manipulator target
    Eigen::Affine3d x_target_l_;      // left EE pose in world
    Eigen::Affine3d x_target_r_;      // right EE pose in world
    Eigen::Affine3d x_goal_l_, x_goal_r_; // for smoothing 

    // states
    std::vector<bool> is_gripper_mode_on_{false, false};
    
    // robot data
    std::map<std::string, drc::TaskSpaceData> ee_data_;


    int dbg_cnt = 0;
    int total_elapsed_steps = 0;


    // action goal data
    int control_mode_;                     // 0: CLIK, 1: OSF, 2:QPIK, 3:QPID
    std::string left_controller_ee_name_;  // EE name for tracking left keyboard controller
    std::string right_controller_ee_name_; // EE name for tracking right keyboard controller
    double controller_pos_multiplier_;
    double controller_ori_multiplier_;



    // home pose mode: -> send goal to fr3_move_to_joint
    const Eigen::Vector<double, FR3_DOF> HomePose{0., -0.785, 0.0, -2.356, 0.0, 1.571, 0.785};
    ActionT::Goal saved_keyboard_goal_{};  // saved goal params for auto-resume after init

    // action clients
    using MoveToJointAction = fr3_husky_msgs::action::MoveToJoint;
    rclcpp_action::Client<MoveToJointAction>::SharedPtr move_to_joint_client_;
    rclcpp_action::Client<ActionT>::SharedPtr           keyboard_self_client_;  // self-client for resume

    // JTC completion monitoring: wait for JTC to finish executing before re-activating
    rclcpp::Subscription<action_msgs::msg::GoalStatusArray>::SharedPtr jtc_status_sub_;
    std::atomic<bool> waiting_for_jtc_{false};


};

}  // namespace fr3_husky_controller::servers::fr3
