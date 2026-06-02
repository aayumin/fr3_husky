#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <type_traits>
#include <deque>
#include <algorithm>
#include <cmath>

#include <action_msgs/msg/goal_status.hpp>
#include <action_msgs/msg/goal_status_array.hpp>
#include <fr3_husky_msgs/action/move_to_joint.hpp>
#include <fr3_husky_msgs/action/apple_vision_pro.hpp>

#include <fr3_husky_controller/servers/action_server_base.hpp>
#include <fr3_husky_controller/model/fr3_husky_model_updater.hpp>
#include <fr3_husky_controller/utils/dyros_math.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <std_msgs/msg/int32_multi_array.hpp>

#define NUM_TRACKERS    3 // left, right, head
#define NUM_CONTROLLERS 2 // left, right (only Focus3 controllers)
#define IDX_LEFT_CON    0 // index of left controller pose in "tracker_pose" topic
#define IDX_RIGHT_CON   1 // index of right controller pose in "tracker_pose" topic
#define IDX_HEAD_CON    2 // index of HMD head pose in "tracker_pose" topic

#define NUM_GESTURES                  6 // number of gesture in AVP controller 
#define IDX_PINCH_GESTURE             0 // index of pinch gesture in "l/r gesture" topic
#define IDX_PINCH_SNAP_LEFT_GESTURE   1 // index of pinch snap gesture in "l/r gesture" topic
#define IDX_PINCH_SNAP_RIGHT_GESTURE  2 // index of pinch snap gesture in "l/r gesture" topic
#define IDX_PINCH_SNAP_UP_GESTURE     3 // index of pinch snap gesture in "l/r gesture" topic
#define IDX_PINCH_SNAP_DOWN_GESTURE   4 // index of pinch snap gesture in "l/r gesture" topic
#define IDX_DOUBLE_TAP_GESTURE        5 // index of double tap gesture in "l/r gesture" topic

namespace fr3_husky_controller::servers::fr3_husky
{

class AppleVisionPro final : public ActionServerBase<fr3_husky_msgs::action::AppleVisionPro>
{
public:
    using ActionT = fr3_husky_msgs::action::AppleVisionPro;
    using Base = ActionServerBase<ActionT>;
    using ComputeResult = typename Base::ComputeResult;
    using StopReason = typename Base::StopReason;
    using ResultPtr = typename Base::ResultPtr;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    AppleVisionPro(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater);
    ~AppleVisionPro() override = default;

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
    // publishers & subscribers
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr  pose_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr l_gesture_state_sub_; // off: 0 | on: 1, 
    rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr r_gesture_state_sub_; // off: 0 | on: 1, 

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_raw_pose_l_pub_, target_raw_pose_r_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_smooth_pose_l_pub_, target_smooth_pose_r_pub_;

    void subPoseCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg);
    void subLGestureCallback(const std_msgs::msg::Int32MultiArray::SharedPtr msg);
    void subRGestureCallback(const std_msgs::msg::Int32MultiArray::SharedPtr msg);

    // AVP controller state data
    std::vector<Eigen::Affine3d> controller_poses_;      // left, right, head
    std::vector<Eigen::Affine3d> controller_poses_init_; // left, right, head
    std::vector<std::vector<bool>> gesture_states_;       // [left, right] 
    std::vector<std::vector<bool>> prev_gesture_states_;  // previous gesture states for edge detection


    // tracking state
    bool left_tracking_mode_on_;   // Action Goal 에서 OnOff
    bool right_tracking_mode_on_;  // Action Goal 에서 OnOff
    std::array<bool, NUM_TRACKERS> tracker_pose_valid_{{false, false, false}};  // 한번이라도 tracker 값을 받았는지
    std::vector<bool> is_realtime_tracking_started_{false, false};  // 필요조건 만족해서 실제 실시간 tracking 시작했는지
    bool auto_tracking_started_ = false;                   



    // states
    bool is_home_mode_on_{false};
    std::vector<bool> is_gripper_mode_on_{false, false};
    
    // robot data
    std::map<std::string, drc::TaskSpaceData> ee_data_;


    // smoothing
    Eigen::Affine3d prev_target_left_ = Eigen::Affine3d::Identity();
    Eigen::Affine3d prev_target_right_ = Eigen::Affine3d::Identity();
    bool is_first_target_left_ = true;
    bool is_first_target_right_ = true;
    Eigen::Affine3d smoothAndLimit(const Eigen::Affine3d& prev, const Eigen::Affine3d& target, double dt);
    double max_linear_vel_ = 0.2;   // m/s
    double max_angular_vel_ = 1.5;  // rad/s
    double smoothing_alpha_ = 0.1;  // low-pass gain (0~1)
    Eigen::Vector6d computeTargetVelocity(const Eigen::Affine3d& prev, const Eigen::Affine3d& cur, double dt);


    // initial capture
    bool is_initialized = false;
    int steps_until_capture_init_tracker = 25;
    int num_steps_for_capture = 0;
    std::deque<Eigen::Affine3d> left_pose_window_;
    std::deque<Eigen::Affine3d> right_pose_window_;
    std::deque<Eigen::Affine3d> head_pose_window_;
    double stable_window_sec_ = 0.1;
    int min_live_updates_in_window_ = 10;
    double min_live_p_diff_ = 1e-7;
    double min_live_r_diff_ = 1e-7;
    double max_stable_p_range_ = 0.05;
    double max_stable_r_range_ = 0.1;
    bool isPoseWindowLiveAndStable(const std::deque<Eigen::Affine3d>& poses);
    double rotationDiff(const Eigen::Matrix3d& R_a, const Eigen::Matrix3d& R_b);

    // check lost live
    int lost_live_steps_ = 0;
    int max_lost_live_steps_ = 30;
    bool isPoseWindowLive(const std::deque<Eigen::Affine3d>& poses);
    void resetRealtimeTracking();


    int dbg_cnt = 0;
    int total_elapsed_steps = 0;

    Eigen::Affine3d world_from_base_init_{Eigen::Affine3d::Identity()};
    Eigen::Affine3d world_from_base_cur_{Eigen::Affine3d::Identity()};

    // startup orientation alignment
    std::vector<bool> ori_startup_calib_done_;
    std::vector<int> ori_startup_calib_count_;
    std::vector<Eigen::Vector4d> ori_startup_calib_sum_;   // quaternion sum in [w x y z]
    int ori_startup_calib_samples_ = 10;                   // first n hand poses

    // action goal data
    int control_mode_;                     // 0: CLIK, 1: OSF, 2:QPIK, 3:QPID
    std::string left_controller_ee_name_;  // EE name for tracking left AVP controller
    std::string right_controller_ee_name_; // EE name for tracking right AVP controller
    bool move_ori_;
    bool constraint_yaw_only_;
    double controller_pos_multiplier_;
    double controller_ori_multiplier_;

    std::mutex tracker_pose_mutex_;
    std::mutex gesture_state_mutex_;

    // null space HomePose cubic
    Eigen::VectorXd q_init_for_home_; // joint config snapshot at manipulator mode start (for cubic null space)



    // home pose mode: -> send goal to fr3_move_to_joint
    const Eigen::Vector<double, FR3_DOF> HomePose{0., -0.785, 0.0, -2.356, 0.0, 1.571, 0.785};
    ActionT::Goal saved_avp_goal_{};  // saved goal params for auto-resume after init

    // action clients
    using MoveToJointAction = fr3_husky_msgs::action::MoveToJoint;
    rclcpp_action::Client<MoveToJointAction>::SharedPtr move_to_joint_client_;
    rclcpp_action::Client<ActionT>::SharedPtr           avp_self_client_;  // self-client for resume

    // JTC completion monitoring: wait for JTC to finish executing before re-activating
    rclcpp::Subscription<action_msgs::msg::GoalStatusArray>::SharedPtr jtc_status_sub_;
    std::atomic<bool> waiting_for_jtc_{false};


};

}  // namespace fr3_husky_controller::servers::fr3
