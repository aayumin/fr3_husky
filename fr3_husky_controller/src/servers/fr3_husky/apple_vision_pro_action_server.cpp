#include <fr3_husky_controller/servers/fr3_husky/apple_vision_pro_action_server.hpp>

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

// Extract robot name ("left" or "right") from an ee_name such as "left_fr3_hand_tcp".
// Returns an empty string if the prefix is not recognised.
std::string getRobotNameFromEEName(const std::string& ee_name)
{
    if (ee_name.rfind("left_", 0) == 0)  return "left";
    if (ee_name.rfind("right_", 0) == 0) return "right";
    return "";
}

}  // namespace

AppleVisionPro::AppleVisionPro(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_husky_model_updater_(getFR3HuskyModelUpdater(model_updater, name))
{
    pose_sub_         = node_->create_subscription<geometry_msgs::msg::PoseArray>("tracker_pose", 1, std::bind(&AppleVisionPro::subPoseCallback, this, std::placeholders::_1));
    l_gesture_state_sub_ = node_->create_subscription<std_msgs::msg::Int32MultiArray>("lhand_gesture", 1, std::bind(&AppleVisionPro::subLGestureCallback, this, std::placeholders::_1));
    r_gesture_state_sub_ = node_->create_subscription<std_msgs::msg::Int32MultiArray>("rhand_gesture", 1, std::bind(&AppleVisionPro::subRGestureCallback, this, std::placeholders::_1));

    controller_poses_.assign(NUM_TRACKERS, Eigen::Affine3d::Identity());
    controller_poses_init_.assign(NUM_TRACKERS, Eigen::Affine3d::Identity());
    gesture_states_.assign(NUM_CONTROLLERS, std::vector<bool>(NUM_GESTURES, false));
    prev_gesture_states_.assign(NUM_CONTROLLERS, std::vector<bool>(NUM_GESTURES, false));

    tracker_base2robot_base_.assign(NUM_CONTROLLERS, Eigen::Matrix3d::Identity());

    ee_data_.clear();

    // Action clients
    move_to_joint_client_ = rclcpp_action::create_client<MoveToJointAction>(node_, "fr3_move_to_joint");
    avp_self_client_       = rclcpp_action::create_client<ActionT>(node_, name_);

    // Subscribe to JTC action status to detect when trajectory execution completes
    auto jtc_qos = rclcpp::QoS(1).reliable().transient_local();
    jtc_status_sub_ = node_->create_subscription<action_msgs::msg::GoalStatusArray>(
        "fr3_joint_trajectory_controller/_action/status",
        jtc_qos,
        [this](const action_msgs::msg::GoalStatusArray::SharedPtr msg)
        {
            if (!waiting_for_jtc_.load(std::memory_order_relaxed)) return;

            bool jtc_busy = false;
            for (const auto& gs : msg->status_list)
            {
                if (gs.status == action_msgs::msg::GoalStatus::STATUS_EXECUTING ||
                    gs.status == action_msgs::msg::GoalStatus::STATUS_ACCEPTED)
                {
                    jtc_busy = true;
                    break;
                }
            }

            if (!jtc_busy)
            {
                waiting_for_jtc_.store(false, std::memory_order_relaxed);
                RCLCPP_INFO(node_->get_logger(),
                            "[%s] JTC finished — re-activating AppleVisionPro", name_.c_str());
                avp_self_client_->async_send_goal(saved_avp_goal_);
            }
        });

    // Initialize franka hand state
    for(const auto& robot_name : fr3_husky_model_updater_.robot_names_) fr3_husky_model_updater_.GripperHoming(robot_name); 

    RCLCPP_INFO(node_->get_logger(), "[%s] AppleVisionPro created", name_.c_str());
}

bool AppleVisionPro::acceptGoal(const ActionT::Goal& goal)
{


    if (!fr3_husky_model_updater_.HasEffortCommandInterface())
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: effort command interface is required",
                                         name_.c_str());
        return false;
    }

    if (goal.mode < 0 || goal.mode > 3)
    {
        RCLCPP_WARN(node_->get_logger(),
                                         "[%s] Reject action: mode must be 0 to 3 (0: CLIK, 1: OSF, 2: QPIK, 3: QPID). The mode from action goal is %d.",
                                         name_.c_str(),
                                         static_cast<int>(goal.mode));
        return false;
    }

    if(!goal.left_controller_ee_name.empty() && !fr3_husky_model_updater_.robot_data_->hasLinkFrame(goal.left_controller_ee_name))
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: left_controller_ee_name from the goal [%s] is not includede in URDF.",
                                         name_.c_str(), goal.left_controller_ee_name.c_str());
        return false;
    }

    if(!goal.right_controller_ee_name.empty() && !fr3_husky_model_updater_.robot_data_->hasLinkFrame(goal.right_controller_ee_name))
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: right_controller_ee_name from the goal [%s] is not includede in URDF.",
                                         name_.c_str(), goal.right_controller_ee_name.c_str());
        return false;
    }

    return true;
}

void AppleVisionPro::onGoalAccepted(const ActionT::Goal& goal)
{
    control_mode_ = goal.mode;
    left_controller_ee_name_ = goal.left_controller_ee_name;
    right_controller_ee_name_ = goal.right_controller_ee_name;
    move_ori_ = goal.move_orientation;
    controller_pos_multiplier_ = static_cast<double>(goal.controller_pos_multiplier);
    controller_ori_multiplier_ = static_cast<double>(goal.controller_ori_multiplier);
    saved_avp_goal_ = goal;

    requestActivate();
}


void AppleVisionPro::onStart()
{
    {
        std::lock_guard<std::mutex> lock(tracker_pose_mutex_);
        for(auto& tracker_pose : controller_poses_) tracker_pose.setIdentity();
    }
    for(auto& tracker_pose_init : controller_poses_init_) tracker_pose_init.setIdentity();
    {
        std::lock_guard<std::mutex> lock(gesture_state_mutex_);
        for(auto& gesture_state : gesture_states_) gesture_state = std::vector<bool>(NUM_GESTURES, false);
    }

    for(auto& prev_gesture_state : prev_gesture_states_) prev_gesture_state = std::vector<bool>(NUM_GESTURES, false);
    is_tracking_mode_on_.assign(NUM_CONTROLLERS, false);
    is_initialize_mode_on_ = false;
    is_gripper_mode_on_.assign(NUM_CONTROLLERS, false);
    ee_data_.clear();
    waiting_for_jtc_.store(false, std::memory_order_relaxed);
    control_start_time_ = -1.0; // sentinel: set on first compute() call
    q_init_for_home_ = fr3_husky_model_updater_.q_total_;

    if(!left_controller_ee_name_.empty())
    {
        ee_data_[left_controller_ee_name_] = drc::TaskSpaceData::Zero();
        ee_data_[left_controller_ee_name_].x = fr3_husky_model_updater_.robot_data_->getPose(left_controller_ee_name_);
        ee_data_[left_controller_ee_name_].xdot = fr3_husky_model_updater_.robot_data_->getVelocity(left_controller_ee_name_);
        ee_data_[left_controller_ee_name_].xddot.setZero();
        ee_data_[left_controller_ee_name_].setInit();
        ee_data_[left_controller_ee_name_].setDesired();
    }
    if(!right_controller_ee_name_.empty())
    {
        ee_data_[right_controller_ee_name_] = drc::TaskSpaceData::Zero();
        ee_data_[right_controller_ee_name_].x = fr3_husky_model_updater_.robot_data_->getPose(right_controller_ee_name_);
        ee_data_[right_controller_ee_name_].xdot = fr3_husky_model_updater_.robot_data_->getVelocity(right_controller_ee_name_);
        ee_data_[right_controller_ee_name_].xddot.setZero();
        ee_data_[right_controller_ee_name_].setInit();
        ee_data_[right_controller_ee_name_].setDesired();
    }

    RCLCPP_INFO(node_->get_logger(), "[%s] started", name_.c_str());
}

AppleVisionPro::ComputeResult AppleVisionPro::compute(const rclcpp::Time& time, const rclcpp::Duration& /*period*/)
{


    for(auto& [ee_name, ee_data] : ee_data_)
    {
        ee_data.x = fr3_husky_model_updater_.robot_data_->getPose(ee_name);
        ee_data.xdot = fr3_husky_model_updater_.robot_data_->getVelocity(ee_name);
        ee_data.xddot.setZero();
    }

    std::vector<Eigen::Affine3d> controller_poses_local;   // left, right, head
    std::vector<std::vector<bool>> gesture_states_local;    // [left, right]
    {
        std::lock_guard<std::mutex> lock(tracker_pose_mutex_);
        controller_poses_local = controller_poses_;
    }
    {
        std::lock_guard<std::mutex> lock(gesture_state_mutex_);
        gesture_states_local = gesture_states_;
    }


    if (control_start_time_ < 0) 
    {
        control_start_time_ = time.seconds();
        q_init_for_home_ = fr3_husky_model_updater_.q_total_;
    }

    // Initialize mode
    {
        if (!is_initialize_mode_on_)
        {
            // if "a" button on the AVP controller is pressed
            if ((!prev_gesture_states_[IDX_LEFT_CON][IDX_PINCH_SNAP_LEFT_GESTURE]  && gesture_states_local[IDX_LEFT_CON][IDX_PINCH_SNAP_LEFT_GESTURE]) ||
                (!prev_gesture_states_[IDX_RIGHT_CON][IDX_PINCH_SNAP_LEFT_GESTURE] && gesture_states_local[IDX_RIGHT_CON][IDX_PINCH_SNAP_LEFT_GESTURE]))
            {
                is_initialize_mode_on_ = true;

                MoveToJointAction::Goal mtj_goal;
                for (const auto& robot_name : fr3_husky_model_updater_.robot_names_)
                {
                    for (size_t j = 0; j < FR3_DOF; ++j)
                    {
                        mtj_goal.joint_names.push_back(robot_name + "_" + fr3_husky_model_updater_.arm_id_ + "_joint" + std::to_string(j+1));
                        mtj_goal.target_positions.push_back(HomePose(j));
                    }
                }
                mtj_goal.max_velocity_scaling_factor     = 0.1;
                mtj_goal.max_acceleration_scaling_factor = 0.1;

                // When MoveToJoint succeeds (trajectory sent to JTC), set waiting_for_jtc_ so
                // the JTC status subscriber re-activates AppleVisionPro after the robot finishes moving.
                auto send_opts = rclcpp_action::Client<MoveToJointAction>::SendGoalOptions();
                send_opts.result_callback =
                    [this](const rclcpp_action::ClientGoalHandle<MoveToJointAction>::WrappedResult&)
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] MoveToJoint done — waiting for JTC to finish", name_.c_str());
                        waiting_for_jtc_.store(true, std::memory_order_relaxed);
                    };

                move_to_joint_client_->async_send_goal(mtj_goal, send_opts);
                RCLCPP_INFO(node_->get_logger(),
                            "[%s] Initialize mode ON — goal sent to fr3_move_to_joint, yielding",
                            name_.c_str());

                // Yield: deactivate AppleVisionPro so MoveToJoint can become active_server_
                return ComputeResult::SUCCEEDED;
            }
        }
    }

    // Gripper control
    {
        // Left controller
        if (!left_controller_ee_name_.empty())
        {
            if (!prev_gesture_states_[IDX_LEFT_CON][IDX_DOUBLE_TAP_GESTURE] && gesture_states_local[IDX_LEFT_CON][IDX_DOUBLE_TAP_GESTURE])
            {
                const std::string robot_name = getRobotNameFromEEName(left_controller_ee_name_);
                if (!robot_name.empty())
                {
                    is_gripper_mode_on_[IDX_LEFT_CON] = !is_gripper_mode_on_[IDX_LEFT_CON];
                    if (is_gripper_mode_on_[IDX_LEFT_CON])
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] lhand trigger released → GripperGrasp('%s')", name_.c_str(), robot_name.c_str());
                        fr3_husky_model_updater_.GripperGrasp(robot_name);
                    }
                    else
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] lhand trigger released → GripperOpen('%s')", name_.c_str(), robot_name.c_str());
                        fr3_husky_model_updater_.GripperOpen(robot_name);
                    }
                }
            }
        }

        // Right controller
        if (!right_controller_ee_name_.empty())
        {
            if (!prev_gesture_states_[IDX_RIGHT_CON][IDX_DOUBLE_TAP_GESTURE] && gesture_states_local[IDX_RIGHT_CON][IDX_DOUBLE_TAP_GESTURE])
            {
                const std::string robot_name = getRobotNameFromEEName(right_controller_ee_name_);
                if (!robot_name.empty())
                {
                    is_gripper_mode_on_[IDX_RIGHT_CON] = !is_gripper_mode_on_[IDX_RIGHT_CON];
                    if (is_gripper_mode_on_[IDX_RIGHT_CON])
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] rhand trigger released → GripperGrasp('%s')", name_.c_str(), robot_name.c_str());
                        fr3_husky_model_updater_.GripperGrasp(robot_name);
                    }
                    else
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] rhand trigger released → GripperOpen('%s')", name_.c_str(), robot_name.c_str());
                        fr3_husky_model_updater_.GripperOpen(robot_name);
                    }
                }
            }
        }
    }

    // Manipulator control
    {   

        // Check real-time hand tracking mode
        for(size_t i = 0; i < NUM_CONTROLLERS; ++i)
        {
            // tracking mode change
            if(!is_tracking_mode_on_[i] && gesture_states_local[i][IDX_PINCH_SNAP_UP_GESTURE]) // activate tracking mode
            {
                RCLCPP_INFO(node_->get_logger(), "[%s] %s Tracking Mode activated!", name_.c_str(), (i==0)?"Left":"Right");
                is_tracking_mode_on_[i] = true;
                is_first_target_left_ = true;
                is_first_target_right_ = true;
    
                controller_poses_init_[i] = controller_poses_local[i];
                if(i == 0 && !left_controller_ee_name_.empty())       ee_data_[left_controller_ee_name_].setInit();
                else if(i == 1 && !right_controller_ee_name_.empty()) ee_data_[right_controller_ee_name_].setInit();

            }
            else if(is_tracking_mode_on_[i] && gesture_states_local[i][IDX_PINCH_SNAP_DOWN_GESTURE]) // deactivate tracking mode
            {
                RCLCPP_INFO(node_->get_logger(), "[%s] %s Tracking Mode deactivated!", name_.c_str(), (i==0)?"Left":"Right");
                is_tracking_mode_on_[i] = false;
    
                controller_poses_init_[i] = controller_poses_local[i];
                if(i == IDX_LEFT_CON && !left_controller_ee_name_.empty())        ee_data_[left_controller_ee_name_].setInit();
                else if(i == IDX_RIGHT_CON && !right_controller_ee_name_.empty()) ee_data_[right_controller_ee_name_].setInit();
            }
            
            // temporarily deactivate tracking when pinch motion
            if(!prev_gesture_states_[i][IDX_PINCH_GESTURE] && gesture_states_local[i][IDX_PINCH_GESTURE]) // deactivate tracking mode
            {
                RCLCPP_INFO(node_->get_logger(), "[%s] %s Tracking Mode deactivated!", name_.c_str(), (i==0)?"Left":"Right");
                is_tracking_mode_on_[i] = false;
    
                controller_poses_init_[i] = controller_poses_local[i];
                if(i == IDX_LEFT_CON && !left_controller_ee_name_.empty())        ee_data_[left_controller_ee_name_].setInit();
                else if(i == IDX_RIGHT_CON && !right_controller_ee_name_.empty()) ee_data_[right_controller_ee_name_].setInit();
            }
            else if(prev_gesture_states_[i][IDX_PINCH_GESTURE] && !gesture_states_local[i][IDX_PINCH_GESTURE]) // activate tracking mode
            {
                RCLCPP_INFO(node_->get_logger(), "[%s] %s Tracking Mode activated!", name_.c_str(), (i==0)?"Left":"Right");
                is_tracking_mode_on_[i] = true;
                is_first_target_left_ = true;
                is_first_target_right_ = true;
    
                controller_poses_init_[i] = controller_poses_local[i];
                if(i == 0 && !left_controller_ee_name_.empty())       ee_data_[left_controller_ee_name_].setInit();
                else if(i == 1 && !right_controller_ee_name_.empty()) ee_data_[right_controller_ee_name_].setInit();
            }
        }
        
        
        if(!left_controller_ee_name_.empty()) // left AVP controller
        {

            Eigen::Affine3d target_pose_diff; // EE init -> EE desired
            Eigen::Vector6d target_vel;
            target_pose_diff.setIdentity();
            target_vel.setZero();
            if(is_tracking_mode_on_[IDX_LEFT_CON])
            {
    
                const Eigen::Affine3d T_con_init2con_cur = controller_poses_init_[IDX_LEFT_CON].inverse() * controller_poses_local[IDX_LEFT_CON];
                const Eigen::Matrix3d R_ee_init2con_init = ee_data_[left_controller_ee_name_].x_init.linear().transpose() * tracker_base2robot_base_[IDX_LEFT_CON].transpose() * controller_poses_init_[IDX_LEFT_CON].linear();
    
                // Position
                target_pose_diff.translation() = controller_pos_multiplier_ * R_ee_init2con_init * T_con_init2con_cur.translation();



    
                // Orientation: using similarity transformation
                target_pose_diff.linear().setIdentity();
                if (move_ori_)
                {
                    const Eigen::AngleAxisd aa(T_con_init2con_cur.linear());
                    const Eigen::Matrix3d R_con_diff_scaled =
                        (std::abs(aa.angle()) > 1e-10)
                        ? Eigen::AngleAxisd(controller_ori_multiplier_ * aa.angle(), aa.axis()).toRotationMatrix()
                        : Eigen::Matrix3d::Identity();
    
                    target_pose_diff.linear() = R_ee_init2con_init * R_con_diff_scaled * R_ee_init2con_init.transpose();
                }
            }
    
            // ee_data_[left_controller_ee_name_].x_desired = ee_data_[left_controller_ee_name_].x_init * target_pose_diff;
            // ee_data_[left_controller_ee_name_].xdot_desired  = target_vel;

            // smoothed target pose
            Eigen::Affine3d raw_target = ee_data_[left_controller_ee_name_].x_init * target_pose_diff;
            double dt = fr3_husky_model_updater_.dt_;
            if (is_first_target_left_)
            {
                prev_target_left_ = raw_target;
                is_first_target_left_ = false;
            }
            Eigen::Affine3d smooth_target = smoothAndLimit(prev_target_left_, raw_target, dt);
            prev_target_left_ = smooth_target;
            ee_data_[left_controller_ee_name_].x_desired = smooth_target;
            ee_data_[left_controller_ee_name_].xdot_desired  = target_vel;

        }
    
        if(!right_controller_ee_name_.empty()) // right AVP controller
        {
            Eigen::Affine3d target_pose_diff; // EE init -> EE desired
            Eigen::Vector6d target_vel;
            target_pose_diff.setIdentity();
            target_vel.setZero();
            if(is_tracking_mode_on_[IDX_RIGHT_CON])
            {
                const Eigen::Affine3d T_con_init2con_cur = controller_poses_init_[IDX_RIGHT_CON].inverse() * controller_poses_local[IDX_RIGHT_CON];
                const Eigen::Matrix3d R_ee_init2con_init = ee_data_[right_controller_ee_name_].x_init.linear().transpose() * tracker_base2robot_base_[IDX_RIGHT_CON].transpose() * controller_poses_init_[IDX_RIGHT_CON].linear();
    
                // Position
                target_pose_diff.translation() = controller_pos_multiplier_ * R_ee_init2con_init * T_con_init2con_cur.translation();
    
                // Orientation: using similarity transformation
                target_pose_diff.linear().setIdentity();
                if (move_ori_)
                {
                    const Eigen::AngleAxisd aa(T_con_init2con_cur.linear());
                    const Eigen::Matrix3d R_con_diff_scaled =
                        (std::abs(aa.angle()) > 1e-10)
                        ? Eigen::AngleAxisd(controller_ori_multiplier_ * aa.angle(), aa.axis()).toRotationMatrix()
                        : Eigen::Matrix3d::Identity();
    
                    target_pose_diff.linear() = R_ee_init2con_init * R_con_diff_scaled * R_ee_init2con_init.transpose();
                }
            }
    
            // ee_data_[right_controller_ee_name_].x_desired = ee_data_[right_controller_ee_name_].x_init * target_pose_diff;
            // ee_data_[right_controller_ee_name_].xdot_desired  = target_vel;
            

            // smoothed target pose
            Eigen::Affine3d raw_target = ee_data_[right_controller_ee_name_].x_init * target_pose_diff;
            double dt = fr3_husky_model_updater_.dt_;
            if (is_first_target_right_)
            {
                prev_target_right_ = raw_target;
                is_first_target_right_ = false;
            }
            Eigen::Affine3d smooth_target = smoothAndLimit(prev_target_right_, raw_target, dt);
            prev_target_right_ = smooth_target;
            ee_data_[right_controller_ee_name_].x_desired = smooth_target;
            ee_data_[right_controller_ee_name_].xdot_desired  = target_vel;

        }
    
        bool is_qp_solved = true;
        std::string time_verbose = "";

        Eigen::VectorXd qdot_mobile = Eigen::VectorXd::Zero(fr3_husky_model_updater_.mobile_dof_);

        switch (control_mode_)
        {
            case 0: // CLIK
                {   
                    // --- Arm null_qdot: cubic toward HomePose ---
                    Eigen::VectorXd HomePose_total(fr3_husky_model_updater_.manipulator_dof_);
                    for(size_t i = 0; i < fr3_husky_model_updater_.num_robots_; ++i)
                        HomePose_total.segment(FR3_DOF*i, FR3_DOF) = HomePose;

                    static constexpr double null_space_duration = 5.0;
                    const Eigen::VectorXd zeros_mani = Eigen::VectorXd::Zero(fr3_husky_model_updater_.manipulator_dof_);
                    const Eigen::VectorXd null_qdot_mani =
                        fr3_husky_model_updater_.robot_controller_->moveManipulatorJointVelocityCubic(HomePose_total, zeros_mani, q_init_for_home_,  zeros_mani, time.seconds(), control_start_time_, null_space_duration);

                    // --- Mobile null_qdot: drive mobile toward EE target (compensates for arm homing) ---
                    // As arm is pulled toward home by null space, EE error grows → mobile drives to fill the gap
                    Eigen::Vector3d avg_ee_pos_error = Eigen::Vector3d::Zero();
                    int ee_count = 0;
                    for(const auto& [ee_name, ee_data] : ee_data_)
                    {
                        avg_ee_pos_error += ee_data.x_desired.translation() - ee_data.x.translation();
                        ++ee_count;
                    }
                    if(ee_count > 0) avg_ee_pos_error /= ee_count;


                    const Eigen::Affine3d world2base_cur = fr3_husky_model_updater_.robot_data_->getPose("base_link");
                    const Eigen::Vector3d ee_error_base = world2base_cur.linear().transpose() * avg_ee_pos_error;
                    static constexpr double mobile_null_gain = 10.0; // [wheel_vel/m]: tune as needed
                    Eigen::Vector3d base_vel_null;
                    base_vel_null << ee_error_base(0), 0.0, 0.0; // Husky cannot strafe (y=0)
                    const Eigen::VectorXd null_qdot_mobile =
                        fr3_husky_model_updater_.robot_controller_->MobileVelocityCommand(mobile_null_gain * base_vel_null);

                    // --- Assemble null_qdot via ActuatorIndex ---
                    Eigen::VectorXd null_qdot(fr3_husky_model_updater_.robot_data_->getActuatorDof());
                    const auto& act_idx = fr3_husky_model_updater_.robot_data_->getActuatorIndex();
                    null_qdot.segment(act_idx.mobi_start, fr3_husky_model_updater_.mobile_dof_) = null_qdot_mobile;
                    null_qdot.segment(act_idx.mani_start, fr3_husky_model_updater_.manipulator_dof_)      = null_qdot_mani;

                    fr3_husky_model_updater_.robot_controller_->CLIKStep(ee_data_, qdot_mobile, fr3_husky_model_updater_.qdot_desired_total_, null_qdot);
                    fr3_husky_model_updater_.q_desired_total_ = fr3_husky_model_updater_.q_total_ +
                                                                fr3_husky_model_updater_.dt_ * fr3_husky_model_updater_.qdot_desired_total_;
                    fr3_husky_model_updater_.torque_desired_total_ = fr3_husky_model_updater_.robot_controller_->moveManipulatorJointTorqueStep(fr3_husky_model_updater_.q_desired_total_, fr3_husky_model_updater_.qdot_desired_total_, false);


                    break;
                }
            case 1: // OSF
                {
                    // Build HomePose target for null space (per robot)
                    Eigen::VectorXd HomePose_total(fr3_husky_model_updater_.manipulator_dof_);
                    for(size_t i = 0; i < fr3_husky_model_updater_.num_robots_; ++i) HomePose_total.segment(FR3_DOF*i, FR3_DOF) = HomePose;
                    Eigen::VectorXd null_torque(fr3_husky_model_updater_.robot_data_->getActuatorDof());
                    null_torque.segment(fr3_husky_model_updater_.robot_data_->getActuatorIndex().mani_start, fr3_husky_model_updater_.manipulator_dof_) = 
                        fr3_husky_model_updater_.robot_controller_->moveManipulatorJointTorqueCubic(HomePose_total,
                                                                                            Eigen::VectorXd::Zero(fr3_husky_model_updater_.manipulator_dof_),
                                                                                            q_init_for_home_,
                                                                                            Eigen::VectorXd::Zero(fr3_husky_model_updater_.manipulator_dof_),
                                                                                            time.seconds(),
                                                                                            control_start_time_,
                                                                                            3.0);
                    // Null-space viscous damping: -kd * wheel_vel_ dissipates kinetic energy of the base
                    static constexpr double mobile_null_damping = 10.0; // [N·m·s/rad]: tune as needed
                    null_torque.segment(fr3_husky_model_updater_.robot_data_->getActuatorIndex().mobi_start, fr3_husky_model_updater_.mobile_dof_) =
                        -mobile_null_damping * fr3_husky_model_updater_.wheel_vel_;


                    Eigen::VectorXd wheel_acc_desired = Eigen::VectorXd::Zero(fr3_husky_model_updater_.mobile_dof_);
                    fr3_husky_model_updater_.robot_controller_->OSFStep(ee_data_,
                                                                        wheel_acc_desired,
                                                                        fr3_husky_model_updater_.torque_desired_total_,
                                                                        null_torque);
                    fr3_husky_model_updater_.wheel_vel_desired_ = fr3_husky_model_updater_.wheel_vel_ + wheel_acc_desired * fr3_husky_model_updater_.dt_;
                    break;
                }

            case 2: // QPIK
                {
                    is_qp_solved = fr3_husky_model_updater_.robot_controller_->QPIKStep(ee_data_, qdot_mobile, fr3_husky_model_updater_.qdot_desired_total_, time_verbose);
                    if(!is_qp_solved)
                    {
                        fr3_husky_model_updater_.qdot_desired_total_.setZero();
                        fr3_husky_model_updater_.wheel_vel_desired_.setZero();
                    }
                    fr3_husky_model_updater_.q_desired_total_ = fr3_husky_model_updater_.q_total_ + fr3_husky_model_updater_.dt_ * fr3_husky_model_updater_.qdot_desired_total_;
                    fr3_husky_model_updater_.torque_desired_total_ = fr3_husky_model_updater_.robot_controller_->moveManipulatorJointTorqueStep(fr3_husky_model_updater_.q_desired_total_, fr3_husky_model_updater_.qdot_desired_total_, false);
                    break;
                }


                
                
            case 3: // QPID
                {
                    Eigen::VectorXd wheel_acc_desired = Eigen::VectorXd::Zero(fr3_husky_model_updater_.mobile_dof_);
                    is_qp_solved = fr3_husky_model_updater_.robot_controller_->QPIDStep(ee_data_, wheel_acc_desired, fr3_husky_model_updater_.torque_desired_total_, time_verbose);
                    if(!is_qp_solved)
                    {
                        fr3_husky_model_updater_.torque_desired_total_ = fr3_husky_model_updater_.g_total_;
                        wheel_acc_desired.setZero();
                    }
                    fr3_husky_model_updater_.wheel_vel_desired_ = fr3_husky_model_updater_.wheel_vel_ + wheel_acc_desired * fr3_husky_model_updater_.dt_;
                    break;
                }


                
            default:
                fr3_husky_model_updater_.qdot_desired_total_.setZero();
                fr3_husky_model_updater_.torque_desired_total_.setZero();
                fr3_husky_model_updater_.wheel_vel_desired_.setZero();
                break;
        }

        fr3_husky_model_updater_.writeCommand(fr3_husky_model_updater_.torque_desired_total_ - fr3_husky_model_updater_.g_total_, Eigen::Vector2d::Zero());  // robot_controller automatically add gravity force
    
        auto fb = std::make_shared<ActionT::Feedback>();
        fb->is_qp_solved = is_qp_solved;
        fb->time_verbose = time_verbose;
        publishFeedback(fb);
    
        return ComputeResult::RUNNING;
    }
}

void AppleVisionPro::onStop(StopReason reason)
{
    fr3_husky_model_updater_.haltCommands();

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

AppleVisionPro::ResultPtr AppleVisionPro::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();
    result->is_completed = true;
    return result;
}

void AppleVisionPro::subPoseCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
{   

    if(msg->poses.size() != NUM_TRACKERS)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of PoseArray for tracker_pose (%ld) does not equal to 3.", name_.c_str(), msg->poses.size());
    }
    else
    {
        for(size_t i = 0; i < msg->poses.size(); ++i)
        {
            Eigen::Vector3d position(msg->poses[i].position.x, msg->poses[i].position.y, msg->poses[i].position.z);
            position = dyros_math::lowPassFilter(position, controller_poses_[i].translation(), 0.001, 0.002);
            Eigen::Quaterniond quaternion(msg->poses[i].orientation.w, msg->poses[i].orientation.x, msg->poses[i].orientation.y, msg->poses[i].orientation.z);
            quaternion.normalize();
            Eigen::Matrix3d orientation = quaternion.toRotationMatrix();
            {
                std::lock_guard<std::mutex> lock(tracker_pose_mutex_);
                controller_poses_[i].translation() = position;
                controller_poses_[i].linear() = orientation;
            }
        }
    }
}

void AppleVisionPro::subLGestureCallback(const std_msgs::msg::Int32MultiArray::SharedPtr msg)
{
    if(msg->data.size() != NUM_GESTURES)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of Int32MultiArray for lhand_gesture (%ld) does not equal to 4.", name_.c_str(), msg->data.size());
    }
    else
    {
        prev_gesture_states_[IDX_LEFT_CON] = gesture_states_[IDX_LEFT_CON];
        for(size_t i = 0; i < msg->data.size(); ++i)
        {
            std::lock_guard<std::mutex> lock(gesture_state_mutex_);
            gesture_states_[IDX_LEFT_CON][i] = (static_cast<int>(msg->data[i]) == 0) ? false : true;
        }

    }
}

void AppleVisionPro::subRGestureCallback(const std_msgs::msg::Int32MultiArray::SharedPtr msg)
{
    if(msg->data.size() != NUM_GESTURES)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of Int32MultiArray for rhand_gesture (%ld) does not equal to 4.", name_.c_str(), msg->data.size());
    }
    else
    {
        prev_gesture_states_[IDX_RIGHT_CON] = gesture_states_[IDX_RIGHT_CON];
        for(size_t i = 0; i < msg->data.size(); ++i)
        {
            std::lock_guard<std::mutex> lock(gesture_state_mutex_);
            gesture_states_[IDX_RIGHT_CON][i] = (static_cast<int>(msg->data[i]) == 0) ? false : true;
        }

    }
}


Eigen::Affine3d AppleVisionPro::smoothAndLimit(const Eigen::Affine3d& prev, const Eigen::Affine3d& target, double dt)
{
    Eigen::Affine3d result = prev;

    // --- 1. Low-pass filter (position)
    Eigen::Vector3d pos =
        (1.0 - smoothing_alpha_) * prev.translation() +
        smoothing_alpha_ * target.translation();

    // --- 2. Velocity limit (position)
    Eigen::Vector3d delta = pos - prev.translation();
    double max_step = max_linear_vel_ * dt;

    if (delta.norm() > max_step) delta = delta.normalized() * max_step;

    result.translation() = prev.translation() + delta;

    // --- 3. Orientation smoothing (slerp)
    Eigen::Quaterniond q_prev(prev.linear());
    Eigen::Quaterniond q_target(target.linear());

    Eigen::Quaterniond q_interp = q_prev.slerp(smoothing_alpha_, q_target);

    // --- 4. Angular velocity limit
    Eigen::AngleAxisd aa(q_prev.inverse() * q_interp);
    double max_angle = max_angular_vel_ * dt;

    if (std::abs(aa.angle()) > max_angle)
    {
        aa.angle() = max_angle;
        q_interp = q_prev * Eigen::Quaterniond(aa);
    }

    result.linear() = q_interp.toRotationMatrix();
    return result;
}


// Register this server into global registry (executed when this TU is linked)
// REGISTER_FR3_ACTION_SERVER(AppleVisionPro, "fr3_AVP_tracker")
REGISTER_FR3_HUSKY_ACTION_SERVER(AppleVisionPro, "fr3_AVP_tracker")


}  // namespace fr3_husky_controller::servers::fr3
/*
# send goal 
ros2 action send_goal /fr3_AVP_tracker fr3_husky_msgs/action/AppleVisionPro \
"{mode: 1, left_controller_ee_name: 'left_fr3_hand_tcp', right_controller_ee_name: 'right_fr3_hand_tcp', move_orientation: false, controller_pos_multiplier: 1.0, controller_ori_multiplier: 1.0}" \
--feedback
*/