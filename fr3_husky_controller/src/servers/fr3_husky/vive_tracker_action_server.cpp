#include <fr3_husky_controller/servers/fr3_husky/vive_tracker_action_server.hpp>

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


ViveTracker::ViveTracker(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_husky_model_updater_(getFR3HuskyModelUpdater(model_updater, name))
{
    // create subscriber
    pose_sub_  = node_->create_subscription<geometry_msgs::msg::PoseArray>("tracker_pose", 1, std::bind(&ViveTracker::subPoseCallback, this, std::placeholders::_1));
    l_joy_sub_ = node_->create_subscription<sensor_msgs::msg::Joy>("lhand_joy", 1, std::bind(&ViveTracker::subLJoyCallback, this, std::placeholders::_1));
    r_joy_sub_ = node_->create_subscription<sensor_msgs::msg::Joy>("rhand_joy", 1, std::bind(&ViveTracker::subRJoyCallback, this, std::placeholders::_1));

    // initialize vive controller states
    controller_poses_.assign(NUM_TRACKERS, Eigen::Affine3d::Identity());
    controller_poses_init_.assign(NUM_TRACKERS, Eigen::Affine3d::Identity());
    button_states_.assign(NUM_CONTROLLERS, std::vector<bool>(NUM_BUTTONS, false));
    prev_button_states_.assign(NUM_CONTROLLERS, std::vector<bool>(NUM_BUTTONS, false));
    axe_states_.assign(NUM_CONTROLLERS, std::vector<double>(NUM_AXES, 0.));

    // initialize robot data
    ee_data_.clear();
    tracker_base2robot_base_.assign(NUM_CONTROLLERS, Eigen::Matrix3d::Identity());

    // Action clients
    move_to_joint_client_ = rclcpp_action::create_client<MoveToJointAction>(node_, "fr3_husky_move_to_joint");
    vt_self_client_       = rclcpp_action::create_client<ActionT>(node_, name_);

    // Subscribe to JTC action status to detect when trajectory execution completes
    auto jtc_qos = rclcpp::QoS(1).reliable().transient_local();
    jtc_status_sub_ = node_->create_subscription<action_msgs::msg::GoalStatusArray>(
        "fr3_husky_joint_trajectory_controller/_action/status",
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
                            "[%s] JTC finished — re-activating ViveTracker", name_.c_str());
                vt_self_client_->async_send_goal(saved_vive_goal_);
            }
        });

    // Initialize franka hand state
    for(const auto& robot_name : model_updater_.robot_names_) fr3_husky_model_updater_.GripperHoming(robot_name);

    fr3_husky_model_updater_.robot_controller_->moma.setIDKvGain({{fr3_husky_model_updater_.robot_data_->getBaseLinkName(),
                                                                   Eigen::Vector6d::Constant(1000.0)}});

    RCLCPP_INFO(node_->get_logger(), "[%s] ViveTracker created", name_.c_str());
}

bool ViveTracker::acceptGoal(const ActionT::Goal& goal)
{
    if (goal.control_mode < 0 || goal.control_mode > 3)
    {
        RCLCPP_WARN(node_->get_logger(),
                                         "[%s] Reject action: mode must be 0 to 3 (0: CLIK, 1: OSF, 2: QPIK, 3: QPID). The mode from action goal is %d.",
                                         name_.c_str(),
                                         static_cast<int>(goal.control_mode));
        return false;
    }

    if (goal.teleop_mode < 0 || goal.teleop_mode > 3)
    {
        RCLCPP_WARN(node_->get_logger(),
                                         "[%s] Reject action: teleop_mode must be 0 to 3 (0: Base, 1: Arm, 2: Wholebody, 3:Dual). The mode from action goal is %d.",
                                         name_.c_str(),
                                         static_cast<int>(goal.teleop_mode));
        return false;
    }

    if(goal.teleop_mode != 0 && !model_updater_.HasEffortCommandInterface() && (goal.control_mode == 1 || goal.control_mode == 3))
    {
        RCLCPP_WARN(node_->get_logger(),
                    "[%s] Reject action: 1: OSF, 3: QPID can be activated only for Effort Command.",
                    name_.c_str());
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

void ViveTracker::onGoalAccepted(const ActionT::Goal& goal)
{
    control_mode_ = goal.control_mode;
    teleop_mode_ = goal.teleop_mode;
    left_controller_ee_name_ = goal.left_controller_ee_name;
    right_controller_ee_name_ = goal.right_controller_ee_name;
    move_ori_ = goal.move_orientation;
    controller_pos_multiplier_ = static_cast<double>(goal.controller_pos_multiplier);
    controller_ori_multiplier_ = static_cast<double>(goal.controller_ori_multiplier);
    saved_vive_goal_ = goal;
}

void ViveTracker::onStart()
{
    const double current_time = node_->now().seconds();

    {
        std::lock_guard<std::mutex> lock(tracker_pose_mutex_);
        for(auto& tracker_pose : controller_poses_) tracker_pose.setIdentity();
    }
    for(auto& controller_pose_init : controller_poses_init_) controller_pose_init.setIdentity();
    {
        std::lock_guard<std::mutex> lock(joy_mutex_);
        for(auto& button_state : button_states_) button_state = std::vector<bool>(NUM_BUTTONS, false);
        for(auto& axe_state : axe_states_) axe_state = std::vector<double>(NUM_AXES, 0.);
    }
    for(auto& prev_button_state : prev_button_states_) prev_button_state = std::vector<bool>(NUM_BUTTONS, false);
    is_mouse_mode_on_.assign(NUM_CONTROLLERS, false);
    is_initialize_mode_on_ = false;
    is_gripper_mode_on_.assign(NUM_CONTROLLERS, false);
    mobile_mode_ = 0;

    ee_data_.clear();
    if(!left_controller_ee_name_.empty())
    {
        ee_data_[left_controller_ee_name_] = drc::TaskSpaceData::Zero();
        ee_data_[left_controller_ee_name_].x = fr3_husky_model_updater_.robot_data_->getPose(left_controller_ee_name_);
        ee_data_[left_controller_ee_name_].xdot = fr3_husky_model_updater_.robot_data_->getVelocity(left_controller_ee_name_);
        ee_data_[left_controller_ee_name_].xddot.setZero();
        ee_data_[left_controller_ee_name_].current_time = current_time;
        ee_data_[left_controller_ee_name_].setInit();
        ee_data_[left_controller_ee_name_].setDesired();
    }
    if(!right_controller_ee_name_.empty())
    {
        ee_data_[right_controller_ee_name_] = drc::TaskSpaceData::Zero();
        ee_data_[right_controller_ee_name_].x = fr3_husky_model_updater_.robot_data_->getPose(right_controller_ee_name_);
        ee_data_[right_controller_ee_name_].xdot = fr3_husky_model_updater_.robot_data_->getVelocity(right_controller_ee_name_);
        ee_data_[right_controller_ee_name_].xddot.setZero();
        ee_data_[right_controller_ee_name_].current_time = current_time;
        ee_data_[right_controller_ee_name_].setInit();
        ee_data_[right_controller_ee_name_].setDesired();
    }

    T_world2mobi_base_ = fr3_husky_model_updater_.robot_data_->getPose(fr3_husky_model_updater_.robot_data_->getBaseLinkName());
    T_world2mobi_base_init_ = T_world2mobi_base_;

    fr3_husky_model_updater_.q_total_init_ = fr3_husky_model_updater_.q_total_;
    
    waiting_for_jtc_.store(false, std::memory_order_relaxed);

    RCLCPP_INFO(node_->get_logger(), "[%s] started", name_.c_str());
}

ViveTracker::ComputeResult ViveTracker::compute(const rclcpp::Time& time, const rclcpp::Duration& /*period*/)
{
    // update robot data
    const double current_time = time.seconds();

    for(auto& [ee_name, ee_data] : ee_data_)
    {
        ee_data.x = fr3_husky_model_updater_.robot_data_->getPose(ee_name);
        ee_data.xdot = fr3_husky_model_updater_.robot_data_->getVelocity(ee_name);
        ee_data.xddot.setZero();
        ee_data.current_time = current_time;
    }

    T_world2mobi_base_ = fr3_husky_model_updater_.robot_data_->getPose(fr3_husky_model_updater_.robot_data_->getBaseLinkName());

    // get controller local data 
    std::vector<Eigen::Affine3d> controller_poses_local;
    std::vector<std::vector<bool>> button_states_local;
    std::vector<std::vector<double>> axe_states_local;
    {
        std::lock_guard<std::mutex> lock(tracker_pose_mutex_);
        controller_poses_local = controller_poses_;
    }
    {
        std::lock_guard<std::mutex> lock(joy_mutex_);
        button_states_local = button_states_;
        axe_states_local = axe_states_;
    }

    // Initialize mode
    {
        if (!is_initialize_mode_on_)
        {
            // if "a" button on the vive controller is pressed
            if ((!prev_button_states_[IDX_LEFT_CON][IDX_A_BUTTON]  && button_states_local[IDX_LEFT_CON][IDX_A_BUTTON]) ||
                (!prev_button_states_[IDX_RIGHT_CON][IDX_A_BUTTON] && button_states_local[IDX_RIGHT_CON][IDX_A_BUTTON]))
            {
                is_initialize_mode_on_ = true;

                MoveToJointAction::Goal mtj_goal;
                for (const auto& robot_name : model_updater_.robot_names_)
                {
                    for (size_t j = 0; j < FR3_DOF; ++j)
                    {
                        mtj_goal.joint_names.push_back(robot_name + "_" + model_updater_.arm_id_ + "_joint" + std::to_string(j+1));
                        mtj_goal.target_positions.push_back(HomePose(j));
                        if(model_updater_.num_robots_ == 2 &&  j == 0)
                        {
                            if(robot_name.find("left")  != std::string::npos) mtj_goal.target_positions.back() -= M_PI / 6;
                            if(robot_name.find("right") != std::string::npos) mtj_goal.target_positions.back() += M_PI / 6;
                        }
                    }
                }
                mtj_goal.max_velocity_scaling_factor     = 0.1;
                mtj_goal.max_acceleration_scaling_factor = 0.1;

                // When MoveToJoint succeeds (trajectory sent to JTC), set waiting_for_jtc_ so
                // the JTC status subscriber re-activates ViveTracker after the robot finishes moving.
                auto send_opts = rclcpp_action::Client<MoveToJointAction>::SendGoalOptions();
                send_opts.result_callback =
                    [this](const rclcpp_action::ClientGoalHandle<MoveToJointAction>::WrappedResult& /*result*/)
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] MoveToJoint done — waiting for JTC to finish", name_.c_str());
                        waiting_for_jtc_.store(true, std::memory_order_relaxed);
                    };

                move_to_joint_client_->async_send_goal(mtj_goal, send_opts);
                RCLCPP_INFO(node_->get_logger(),
                            "[%s] Initialize mode ON — goal sent to fr3_move_to_joint, yielding",
                            name_.c_str());

                // Yield: deactivate ViveTracker so MoveToJoint can become active_server_
                prev_button_states_ = button_states_local;
                return ComputeResult::SUCCEEDED;
            }
        }
    }

    // Gripper control
    {
        // Left controller
        if (!left_controller_ee_name_.empty())
        {
            if (!prev_button_states_[IDX_LEFT_CON][IDX_TRIGGER_BUTTON] && button_states_local[IDX_LEFT_CON][IDX_TRIGGER_BUTTON])
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
            if (!prev_button_states_[IDX_RIGHT_CON][IDX_TRIGGER_BUTTON] && button_states_local[IDX_RIGHT_CON][IDX_TRIGGER_BUTTON])
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

    bool is_qp_solved = true;
    std::string time_verbose = "";

    // Check mouse mode for arm / whole-body / dual teleoperation.
    for(size_t i = 0; i < NUM_CONTROLLERS; ++i)
    {
        if(!is_mouse_mode_on_[i] && button_states_local[i][IDX_GRIP_BUTTON])
        {
            RCLCPP_INFO(node_->get_logger(), "[%s] %s Mouse Mode activated!", name_.c_str(), (i==0)?"Left":"Right");

            is_mouse_mode_on_[i] = true;
            controller_poses_init_[i] = controller_poses_local[i];
            if(i == IDX_LEFT_CON && !left_controller_ee_name_.empty())        ee_data_[left_controller_ee_name_].setInit();
            else if(i == IDX_RIGHT_CON && !right_controller_ee_name_.empty()) ee_data_[right_controller_ee_name_].setInit();
            T_world2mobi_base_init_ = T_world2mobi_base_;
        }
        else if(is_mouse_mode_on_[i] && !button_states_local[i][IDX_GRIP_BUTTON])
        {
            RCLCPP_INFO(node_->get_logger(), "[%s] %s Mouse Mode deactivated!", name_.c_str(), (i==0)?"Left":"Right");
            is_mouse_mode_on_[i] = false;

            controller_poses_init_[i] = controller_poses_local[i];
            if(i == IDX_LEFT_CON && !left_controller_ee_name_.empty())        ee_data_[left_controller_ee_name_].setInit();
            else if(i == IDX_RIGHT_CON && !right_controller_ee_name_.empty()) ee_data_[right_controller_ee_name_].setInit();
            T_world2mobi_base_init_ = T_world2mobi_base_;
        }
    }

    // Check mobile mode for base / dual teleoperation.
    if(mobile_mode_ == 0)
    {
        if(!prev_button_states_[IDX_LEFT_CON][IDX_JOY_BUTTON] && button_states_local[IDX_LEFT_CON][IDX_JOY_BUTTON])
        {
            RCLCPP_INFO(node_->get_logger(), "[%s] lhand joy released → Mobile control", name_.c_str());
            mobile_mode_ = 1;
        }
        else if(!prev_button_states_[IDX_RIGHT_CON][IDX_JOY_BUTTON] && button_states_local[IDX_RIGHT_CON][IDX_JOY_BUTTON])
        {
            RCLCPP_INFO(node_->get_logger(), "[%s] rhand joy released → Mobile control", name_.c_str());
            mobile_mode_ = 2;
        }
    }
    else if(mobile_mode_ == 1)
    {
        if(!prev_button_states_[IDX_LEFT_CON][IDX_JOY_BUTTON] && button_states_local[IDX_LEFT_CON][IDX_JOY_BUTTON])
        {
            RCLCPP_INFO(node_->get_logger(), "[%s] lhand joy released → Mobile stop", name_.c_str());
            mobile_mode_ = 0;
        }
        else if(!prev_button_states_[IDX_RIGHT_CON][IDX_JOY_BUTTON] && button_states_local[IDX_RIGHT_CON][IDX_JOY_BUTTON])
        {
            RCLCPP_INFO(node_->get_logger(), "[%s] rhand joy released → Mobile control", name_.c_str());
            mobile_mode_ = 2;
        }
    }
    else
    {
        if(!prev_button_states_[IDX_RIGHT_CON][IDX_JOY_BUTTON] && button_states_local[IDX_RIGHT_CON][IDX_JOY_BUTTON])
        {
            RCLCPP_INFO(node_->get_logger(), "[%s] rhand joy released → Mobile stop", name_.c_str());
            mobile_mode_ = 0;
        }
        else if(!prev_button_states_[IDX_LEFT_CON][IDX_JOY_BUTTON] && button_states_local[IDX_LEFT_CON][IDX_JOY_BUTTON])
        {
            RCLCPP_INFO(node_->get_logger(), "[%s] lhand joy released → Mobile control", name_.c_str());
            mobile_mode_ = 1;
        }
    }

    // Base Mode: arm holds, base driven by joystick
    if(teleop_mode_ == 0)
    {
        // Manipulator control: freeze
        fr3_husky_model_updater_.q_desired_total_ = fr3_husky_model_updater_.q_total_init_;
        fr3_husky_model_updater_.qdot_desired_total_.setZero();
        fr3_husky_model_updater_.torque_desired_total_
            = fr3_husky_model_updater_.robot_controller_->mani.moveJointTorqueStep(fr3_husky_model_updater_.q_desired_total_,
                                                                                   fr3_husky_model_updater_.qdot_desired_total_,
                                                                                   false);
        // Mobile: joy command                                                                           
        // TODO: scaling factor for joy to velocity
        if(mobile_mode_ == 1) // left joy control
        {
            fr3_husky_model_updater_.base_vel_b_desired_ << axe_states_[IDX_LEFT_CON][IDX_JOY_Y_AXES], 0.0, -axe_states_[IDX_LEFT_CON][IDX_JOY_X_AXES];
        }
        else if (mobile_mode_ == 2) // right joy control
        {
            fr3_husky_model_updater_.base_vel_b_desired_ << axe_states_[IDX_RIGHT_CON][IDX_JOY_Y_AXES], 0.0, -axe_states_[IDX_RIGHT_CON][IDX_JOY_X_AXES];
        }
        else // stop
        {
            fr3_husky_model_updater_.base_vel_b_desired_.setZero();
        }

        fr3_husky_model_updater_.wheel_vel_desired_ = fr3_husky_model_updater_.robot_controller_->MobileVelocityCommand(fr3_husky_model_updater_.base_vel_b_desired_);
    }

    // Arm Mode: base holds, arm driven by vive controller (EE in base frame)
    else if(teleop_mode_ == 1)
    {
        std::map<std::string, drc::TaskSpaceData> ee_data_mani = ee_data_; // mobile base to EE

        if(!left_controller_ee_name_.empty()) // left vive controller
        {
            // EE desired expressed in base_link frame
            const Eigen::Affine3d T_mobi_base2ee_init = T_world2mobi_base_init_.inverse() * ee_data_[left_controller_ee_name_].x_init;
            Eigen::Affine3d T_mobi_base2ee_des = T_mobi_base2ee_init;
            Eigen::Vector6d target_vel;
            target_vel.setZero();

            if(is_mouse_mode_on_[IDX_LEFT_CON])
            {
                const Eigen::Affine3d T_cont_init2cont = controller_poses_init_[IDX_LEFT_CON].inverse() * controller_poses_local[IDX_LEFT_CON];
                const Eigen::Matrix3d R_mobi_base2cont_init = tracker_base2robot_base_[IDX_LEFT_CON].transpose() * controller_poses_init_[IDX_LEFT_CON].linear();

                // Position: delta added in base_link frame
                T_mobi_base2ee_des.translation() += controller_pos_multiplier_ * R_mobi_base2cont_init * T_cont_init2cont.translation();

                // Orientation: global rotation in base_link frame (pre-multiply)
                if(move_ori_)
                {
                    const Eigen::AngleAxisd aa(T_cont_init2cont.linear());
                    const Eigen::Matrix3d R_cont_diff_scaled = (std::abs(aa.angle()) > 1e-10)
                        ? Eigen::AngleAxisd(controller_ori_multiplier_ * aa.angle(), aa.axis()).toRotationMatrix()
                        : Eigen::Matrix3d::Identity();
                    const Eigen::Matrix3d R_delta_in_base = R_mobi_base2cont_init * R_cont_diff_scaled * R_mobi_base2cont_init.transpose();
                    T_mobi_base2ee_des.linear() = R_delta_in_base * T_mobi_base2ee_init.linear();
                }
            }

            ee_data_[left_controller_ee_name_].x_desired    = T_world2mobi_base_ * T_mobi_base2ee_des;
            ee_data_[left_controller_ee_name_].xdot_desired = target_vel;
            ee_data_[left_controller_ee_name_].xddot_desired.setZero();
            
            // calculate target ee_data_ in mobile base frame 
            // fr3_husky_model_updater_.robot_controller_->mani.... needs EE in mobile base frame, not world frame 
            ee_data_mani[left_controller_ee_name_].x_desired = T_mobi_base2ee_des;
            ee_data_mani[left_controller_ee_name_].xdot_desired = target_vel;
            ee_data_mani[left_controller_ee_name_].xddot_desired.setZero();
        }

        if(!right_controller_ee_name_.empty()) // right vive controller
        {
            // EE desired expressed in base_link frame
            const Eigen::Affine3d T_mobi_base2ee_init = T_world2mobi_base_init_.inverse() * ee_data_[right_controller_ee_name_].x_init;
            Eigen::Affine3d T_mobi_base2ee_des = T_mobi_base2ee_init;
            Eigen::Vector6d target_vel;
            target_vel.setZero();

            if(is_mouse_mode_on_[IDX_RIGHT_CON])
            {
                const Eigen::Affine3d T_cont_init2cont = controller_poses_init_[IDX_RIGHT_CON].inverse() * controller_poses_local[IDX_RIGHT_CON];
                const Eigen::Matrix3d R_mobi_base2cont_init = tracker_base2robot_base_[IDX_RIGHT_CON].transpose() * controller_poses_init_[IDX_RIGHT_CON].linear();

                // Position: delta added in base_link frame
                T_mobi_base2ee_des.translation() += controller_pos_multiplier_ * R_mobi_base2cont_init * T_cont_init2cont.translation();

                // Orientation: global rotation in base_link frame (pre-multiply)
                if(move_ori_)
                {
                    const Eigen::AngleAxisd aa(T_cont_init2cont.linear());
                    const Eigen::Matrix3d R_cont_diff_scaled = (std::abs(aa.angle()) > 1e-10)
                        ? Eigen::AngleAxisd(controller_ori_multiplier_ * aa.angle(), aa.axis()).toRotationMatrix()
                        : Eigen::Matrix3d::Identity();
                    const Eigen::Matrix3d R_delta_in_base = R_mobi_base2cont_init * R_cont_diff_scaled * R_mobi_base2cont_init.transpose();
                    T_mobi_base2ee_des.linear() = R_delta_in_base * T_mobi_base2ee_init.linear();
                }
            }

            ee_data_[right_controller_ee_name_].x_desired    = T_world2mobi_base_ * T_mobi_base2ee_des;
            ee_data_[right_controller_ee_name_].xdot_desired = target_vel;
            ee_data_[right_controller_ee_name_].xddot_desired.setZero();

            // calculate target ee_data_ in mobile base frame 
            // fr3_husky_model_updater_.robot_controller_->mani.... needs EE in mobile base frame, not world frame 
            ee_data_mani[right_controller_ee_name_].x_desired = T_mobi_base2ee_des;
            ee_data_mani[right_controller_ee_name_].xdot_desired = target_vel;
            ee_data_mani[right_controller_ee_name_].xddot_desired.setZero();
        }

        bool is_qp_solved_arm = true;
        std::string time_verbose_arm = "";

        // Manipulator control: vive controller command
        switch(control_mode_)
        {
            case 0: // CLIK
            {
                fr3_husky_model_updater_.robot_controller_->mani.CLIKStep(ee_data_mani,
                                                                          fr3_husky_model_updater_.qdot_desired_total_);
                fr3_husky_model_updater_.q_desired_total_ =
                    fr3_husky_model_updater_.q_total_ +
                    fr3_husky_model_updater_.dt_ * fr3_husky_model_updater_.qdot_desired_total_;
                fr3_husky_model_updater_.torque_desired_total_ =
                    fr3_husky_model_updater_.robot_controller_->mani.moveJointTorqueStep(fr3_husky_model_updater_.q_desired_total_,
                                                                                         fr3_husky_model_updater_.qdot_desired_total_, 
                                                                                         false);
                break;
            }
            case 1: // OSF
            {
                fr3_husky_model_updater_.robot_controller_->mani.OSFStep(ee_data_mani,
                                                                         fr3_husky_model_updater_.torque_desired_total_);
                break;
            }
            case 2: // QPIK
            {
                is_qp_solved_arm = fr3_husky_model_updater_.robot_controller_->mani.QPIKStep(ee_data_mani,
                                                                                             fr3_husky_model_updater_.qdot_desired_total_,
                                                                                             time_verbose_arm);
                if(!is_qp_solved_arm) fr3_husky_model_updater_.qdot_desired_total_.setZero();
                fr3_husky_model_updater_.q_desired_total_ = fr3_husky_model_updater_.q_total_ +
                                                            fr3_husky_model_updater_.dt_ * fr3_husky_model_updater_.qdot_desired_total_;
                fr3_husky_model_updater_.torque_desired_total_ =
                    fr3_husky_model_updater_.robot_controller_->mani.moveJointTorqueStep(fr3_husky_model_updater_.q_desired_total_,
                                                                                         fr3_husky_model_updater_.qdot_desired_total_, 
                                                                                         false);
                break;
            }
            case 3: // QPID
            {
                is_qp_solved_arm = fr3_husky_model_updater_.robot_controller_->mani.QPIDStep(ee_data_mani,
                    fr3_husky_model_updater_.torque_desired_total_,
                    time_verbose_arm);
                if(!is_qp_solved_arm)
                    fr3_husky_model_updater_.torque_desired_total_ = fr3_husky_model_updater_.robot_data_->mani.getGravity();
                break;
            }
        }

        // Lock base — ignore any wheel output from the controller
        fr3_husky_model_updater_.wheel_vel_desired_.setZero();

        is_qp_solved = is_qp_solved_arm;
        time_verbose = time_verbose_arm;
    }

    // Whole-Body Mode: both base and arm driven by vive controller (EE in world frame)
    else if(teleop_mode_ == 2)
    {
        if(!left_controller_ee_name_.empty()) // left vive controller
        {
            Eigen::Affine3d T_ee_init2ee_des = Eigen::Affine3d::Identity();
            const Eigen::Vector6d target_vel = Eigen::Vector6d::Zero();
            if(is_mouse_mode_on_[IDX_LEFT_CON])
            {
                const Eigen::Affine3d T_cont_init2cont = controller_poses_init_[IDX_LEFT_CON].inverse() * controller_poses_local[IDX_LEFT_CON];
                const Eigen::Matrix3d R_ee_init2cont_init = ee_data_[left_controller_ee_name_].x_init.linear().transpose() * tracker_base2robot_base_[IDX_LEFT_CON].transpose() *  controller_poses_init_[IDX_LEFT_CON].linear();
                
                // Position
                T_ee_init2ee_des.translation() = controller_pos_multiplier_ * R_ee_init2cont_init * T_cont_init2cont.translation();

                // Orientation: using similarity transformation
                if(move_ori_)
                {
                    const Eigen::AngleAxisd aa(T_cont_init2cont.linear());
                    const Eigen::Matrix3d R_cont_diff_scaled =
                        (std::abs(aa.angle()) > 1e-10)
                        ? Eigen::AngleAxisd(controller_ori_multiplier_ * aa.angle(), aa.axis()).toRotationMatrix()
                        : Eigen::Matrix3d::Identity();
                    T_ee_init2ee_des.linear() = R_ee_init2cont_init * R_cont_diff_scaled * R_ee_init2cont_init.transpose();
                }
            }

            ee_data_[left_controller_ee_name_].x_desired    = ee_data_[left_controller_ee_name_].x_init * T_ee_init2ee_des;
            ee_data_[left_controller_ee_name_].xdot_desired = target_vel;
            ee_data_[left_controller_ee_name_].xddot_desired.setZero();
        }

        if(!right_controller_ee_name_.empty()) // right vive controller
        {
            Eigen::Affine3d T_ee_init2ee_des = Eigen::Affine3d::Identity();
            const Eigen::Vector6d target_vel = Eigen::Vector6d::Zero();
            if(is_mouse_mode_on_[IDX_RIGHT_CON])
            {
                const Eigen::Affine3d T_cont_init2cont = controller_poses_init_[IDX_RIGHT_CON].inverse() * controller_poses_local[IDX_RIGHT_CON];
                const Eigen::Matrix3d R_ee_init2cont_init = ee_data_[right_controller_ee_name_].x_init.linear().transpose() * tracker_base2robot_base_[IDX_RIGHT_CON].transpose() *  controller_poses_init_[IDX_RIGHT_CON].linear();
                
                // Position
                T_ee_init2ee_des.translation() = controller_pos_multiplier_ * R_ee_init2cont_init * T_cont_init2cont.translation();

                // Orientation: using similarity transformation
                if(move_ori_)
                {
                    const Eigen::AngleAxisd aa(T_cont_init2cont.linear());
                    const Eigen::Matrix3d R_cont_diff_scaled =
                        (std::abs(aa.angle()) > 1e-10)
                        ? Eigen::AngleAxisd(controller_ori_multiplier_ * aa.angle(), aa.axis()).toRotationMatrix()
                        : Eigen::Matrix3d::Identity();
                    T_ee_init2ee_des.linear() = R_ee_init2cont_init * R_cont_diff_scaled * R_ee_init2cont_init.transpose();
                }
            }

            ee_data_[right_controller_ee_name_].x_desired    = ee_data_[right_controller_ee_name_].x_init * T_ee_init2ee_des;
            ee_data_[right_controller_ee_name_].xdot_desired = target_vel;
            ee_data_[right_controller_ee_name_].xddot_desired.setZero();
        }

        bool is_qp_solved_wb = true;
        std::string time_verbose_wb = "";
        Eigen::VectorXd wheel_acc_desired;
        wheel_acc_desired.setZero(fr3_husky_model_updater_.mobile_dof_);

        switch(control_mode_)
        {
            case 0: // CLIK
            {
                fr3_husky_model_updater_.robot_controller_->moma.CLIKStep(ee_data_,
                                                                          fr3_husky_model_updater_.wheel_vel_desired_,
                                                                          fr3_husky_model_updater_.qdot_desired_total_);
                fr3_husky_model_updater_.q_desired_total_ = fr3_husky_model_updater_.q_total_ +
                                                            fr3_husky_model_updater_.dt_ * fr3_husky_model_updater_.qdot_desired_total_;
                fr3_husky_model_updater_.torque_desired_total_ =
                    fr3_husky_model_updater_.robot_controller_->mani.moveJointTorqueStep(fr3_husky_model_updater_.q_desired_total_,
                                                                                         fr3_husky_model_updater_.qdot_desired_total_, 
                                                                                         false);
                break;
            }
            case 1: // OSF
            {
                fr3_husky_model_updater_.robot_controller_->moma.OSFStep(ee_data_,
                                                                         wheel_acc_desired,
                                                                         fr3_husky_model_updater_.torque_desired_total_);
                fr3_husky_model_updater_.wheel_vel_desired_ = fr3_husky_model_updater_.wheel_vel_+ 
                                                              fr3_husky_model_updater_.dt_ * wheel_acc_desired;
                break;
            }
            case 2: // QPIK
            {
                is_qp_solved_wb = fr3_husky_model_updater_.robot_controller_->moma.QPIKStep(ee_data_,
                                                                                            fr3_husky_model_updater_.wheel_vel_desired_,
                                                                                            fr3_husky_model_updater_.qdot_desired_total_,
                                                                                            time_verbose_wb);
                if(!is_qp_solved_wb)
                {
                    fr3_husky_model_updater_.qdot_desired_total_.setZero();
                    fr3_husky_model_updater_.wheel_vel_desired_.setZero();
                }
                fr3_husky_model_updater_.q_desired_total_ +=
                    fr3_husky_model_updater_.dt_ * fr3_husky_model_updater_.qdot_desired_total_;
                fr3_husky_model_updater_.torque_desired_total_ =
                    fr3_husky_model_updater_.robot_controller_->mani.moveJointTorqueStep(fr3_husky_model_updater_.q_desired_total_,
                                                                                         fr3_husky_model_updater_.qdot_desired_total_, 
                                                                                         false);
                break;
            }
            case 3: // QPID
            {
                is_qp_solved_wb = fr3_husky_model_updater_.robot_controller_->moma.QPIDStep(ee_data_,
                                                                                            wheel_acc_desired,
                                                                                            fr3_husky_model_updater_.torque_desired_total_,
                                                                                            time_verbose_wb);
                if(!is_qp_solved_wb)
                {
                    fr3_husky_model_updater_.torque_desired_total_ = fr3_husky_model_updater_.robot_data_->mani.getGravity();
                    wheel_acc_desired.setZero();
                }
                fr3_husky_model_updater_.wheel_vel_desired_ = fr3_husky_model_updater_.wheel_vel_ +
                                                              fr3_husky_model_updater_.dt_ * wheel_acc_desired;
                break;
            }
        }
        is_qp_solved = is_qp_solved_wb;
        time_verbose = time_verbose_wb;
    }

    // Dual Mode: EE tracking (world frame, haptic) as primary task; joystick mobile velocity in null space
    else if(teleop_mode_ == 3)
    {
        if(!left_controller_ee_name_.empty()) // left vive controller
        {
            Eigen::Affine3d T_ee_init2ee_des = Eigen::Affine3d::Identity();
            const Eigen::Vector6d target_vel = Eigen::Vector6d::Zero();
            if(is_mouse_mode_on_[IDX_LEFT_CON])
            {
                const Eigen::Affine3d T_cont_init2cont = controller_poses_init_[IDX_LEFT_CON].inverse() * controller_poses_local[IDX_LEFT_CON];
                const Eigen::Matrix3d R_ee_init2cont_init = ee_data_[left_controller_ee_name_].x_init.linear().transpose() * tracker_base2robot_base_[IDX_LEFT_CON].transpose() *  controller_poses_init_[IDX_LEFT_CON].linear();
                
                // Position
                T_ee_init2ee_des.translation() = controller_pos_multiplier_ * R_ee_init2cont_init * T_cont_init2cont.translation();

                // Orientation: using similarity transformation
                if(move_ori_)
                {
                    const Eigen::AngleAxisd aa(T_cont_init2cont.linear());
                    const Eigen::Matrix3d R_cont_diff_scaled =
                        (std::abs(aa.angle()) > 1e-10)
                        ? Eigen::AngleAxisd(controller_ori_multiplier_ * aa.angle(), aa.axis()).toRotationMatrix()
                        : Eigen::Matrix3d::Identity();
                    T_ee_init2ee_des.linear() = R_ee_init2cont_init * R_cont_diff_scaled * R_ee_init2cont_init.transpose();
                }
            }

            ee_data_[left_controller_ee_name_].x_desired    = ee_data_[left_controller_ee_name_].x_init * T_ee_init2ee_des;
            ee_data_[left_controller_ee_name_].xdot_desired = target_vel;
            ee_data_[left_controller_ee_name_].xddot_desired.setZero();
        }

        if(!right_controller_ee_name_.empty()) // right vive controller
        {
            Eigen::Affine3d T_ee_init2ee_des = Eigen::Affine3d::Identity();
            const Eigen::Vector6d target_vel = Eigen::Vector6d::Zero();
            if(is_mouse_mode_on_[IDX_RIGHT_CON])
            {
                const Eigen::Affine3d T_cont_init2cont = controller_poses_init_[IDX_RIGHT_CON].inverse() * controller_poses_local[IDX_RIGHT_CON];
                const Eigen::Matrix3d R_ee_init2cont_init = ee_data_[right_controller_ee_name_].x_init.linear().transpose() * tracker_base2robot_base_[IDX_RIGHT_CON].transpose() *  controller_poses_init_[IDX_RIGHT_CON].linear();
                
                // Position
                T_ee_init2ee_des.translation() = controller_pos_multiplier_ * R_ee_init2cont_init * T_cont_init2cont.translation();

                // Orientation: using similarity transformation
                if(move_ori_)
                {
                    const Eigen::AngleAxisd aa(T_cont_init2cont.linear());
                    const Eigen::Matrix3d R_cont_diff_scaled =
                        (std::abs(aa.angle()) > 1e-10)
                        ? Eigen::AngleAxisd(controller_ori_multiplier_ * aa.angle(), aa.axis()).toRotationMatrix()
                        : Eigen::Matrix3d::Identity();
                    T_ee_init2ee_des.linear() = R_ee_init2cont_init * R_cont_diff_scaled * R_ee_init2cont_init.transpose();
                }
            }

            ee_data_[right_controller_ee_name_].x_desired    = ee_data_[right_controller_ee_name_].x_init * T_ee_init2ee_des;
            ee_data_[right_controller_ee_name_].xdot_desired = target_vel;
            ee_data_[right_controller_ee_name_].xddot_desired.setZero();
        }


        // Mobile: joy command                                                                           
        // TODO: scaling factor for joy to velocity
        Eigen::Vector3d base_vel_b_joy;
        if(mobile_mode_ == 1) // left joy control
        {
            base_vel_b_joy << axe_states_[IDX_LEFT_CON][IDX_JOY_Y_AXES], 0.0, -axe_states_[IDX_LEFT_CON][IDX_JOY_X_AXES];
        }
        else if (mobile_mode_ == 2) // right joy control
        {
            base_vel_b_joy << axe_states_[IDX_RIGHT_CON][IDX_JOY_Y_AXES], 0.0, -axe_states_[IDX_RIGHT_CON][IDX_JOY_X_AXES];
        }
        else // stop
        {
            base_vel_b_joy.setZero();
        }

        const Eigen::Vector2d joy_wheel_vel = fr3_husky_model_updater_.robot_controller_->mobi.VelocityCommand(base_vel_b_joy);

        bool is_qp_solved_dual = true;
        std::string time_verbose_dual = "";
        Eigen::VectorXd wheel_acc_desired;
        wheel_acc_desired.setZero(fr3_husky_model_updater_.mobile_dof_);

        switch(control_mode_)
        {
            case 0: // CLIK
            {
                fr3_husky_model_updater_.robot_controller_->moma.CLIKStep(ee_data_,
                                                                          fr3_husky_model_updater_.wheel_vel_desired_,
                                                                          fr3_husky_model_updater_.qdot_desired_total_,
                                                                          joy_wheel_vel);
                fr3_husky_model_updater_.q_desired_total_ = fr3_husky_model_updater_.q_total_ +
                                                            fr3_husky_model_updater_.dt_ * fr3_husky_model_updater_.qdot_desired_total_;
                fr3_husky_model_updater_.torque_desired_total_ =
                    fr3_husky_model_updater_.robot_controller_->mani.moveJointTorqueStep(fr3_husky_model_updater_.q_desired_total_,
                                                                                         fr3_husky_model_updater_.qdot_desired_total_, 
                                                                                         false);
                break;
            }
            case 1: // OSF
            {
                const Eigen::VectorXd null_torque_mobile = dual_mobile_null_torque_gain_ * (joy_wheel_vel - fr3_husky_model_updater_.wheel_vel_);
                fr3_husky_model_updater_.robot_controller_->moma.OSFStep(ee_data_,
                                                                         wheel_acc_desired,
                                                                         fr3_husky_model_updater_.torque_desired_total_,
                                                                         null_torque_mobile);
                fr3_husky_model_updater_.wheel_vel_desired_ =
                    fr3_husky_model_updater_.wheel_vel_ +
                    fr3_husky_model_updater_.dt_ * wheel_acc_desired;
                break;
            }
            case 2: // HQPIK
            {
                const std::string base_link_name = fr3_husky_model_updater_.robot_data_->getBaseLinkName();

                // Mobile base: velocity-only tracking (x_desired = current pose → zero position error)
                drc::TaskSpaceData base_task_data = drc::TaskSpaceData::Zero();
                base_task_data.x         = fr3_husky_model_updater_.robot_data_->getPose(base_link_name);
                base_task_data.xdot      = fr3_husky_model_updater_.robot_data_->getVelocity(base_link_name);
                base_task_data.x_desired = base_task_data.x;   // no position correction

                // Convert joy cmd_vel (body frame) → world-frame 6D twist for xdot_desired
                const Eigen::Matrix3d R_base = T_world2mobi_base_.linear();
                base_task_data.xdot_desired.head<3>() = R_base * Eigen::Vector3d(base_vel_b_joy(0), base_vel_b_joy(1), 0.0);
                base_task_data.xdot_desired.tail<3>() = R_base * Eigen::Vector3d(0.0, 0.0, base_vel_b_joy(2));

                // Priority 1: EE tracking  |  Priority 2: mobile base velocity
                const std::vector<std::map<std::string, drc::TaskSpaceData>> task_hierarchy = {
                    ee_data_,
                    {{base_link_name,   base_task_data}}
                };

                is_qp_solved_dual = fr3_husky_model_updater_.robot_controller_->moma.HQPIKStep(
                    task_hierarchy,
                    fr3_husky_model_updater_.wheel_vel_desired_,
                    fr3_husky_model_updater_.qdot_desired_total_,
                    time_verbose_dual);

                if(!is_qp_solved_dual)
                {
                    fr3_husky_model_updater_.qdot_desired_total_.setZero();
                    fr3_husky_model_updater_.wheel_vel_desired_.setZero();
                }
                fr3_husky_model_updater_.q_desired_total_ =
                    fr3_husky_model_updater_.q_total_ +
                    fr3_husky_model_updater_.dt_ * fr3_husky_model_updater_.qdot_desired_total_;
                fr3_husky_model_updater_.torque_desired_total_ =
                    fr3_husky_model_updater_.robot_controller_->mani.moveJointTorqueStep(
                        fr3_husky_model_updater_.q_desired_total_,
                        fr3_husky_model_updater_.qdot_desired_total_,
                        false);
                break;
            }
            case 3: // HQPID
            {
                const std::string base_link_name = fr3_husky_model_updater_.robot_data_->getBaseLinkName();

                drc::TaskSpaceData base_task_data = drc::TaskSpaceData::Zero();
                base_task_data.x         = fr3_husky_model_updater_.robot_data_->getPose(base_link_name);
                base_task_data.xdot      = fr3_husky_model_updater_.robot_data_->getVelocity(base_link_name);
                base_task_data.x_desired = base_task_data.x;  // no position correction
                // xddot_desired stays zero (no feed-forward)

                const Eigen::Matrix3d R_base = T_world2mobi_base_.linear();
                base_task_data.xdot_desired.head<3>() = R_base * Eigen::Vector3d(base_vel_b_joy(0), base_vel_b_joy(1), 0.0);
                base_task_data.xdot_desired.tail<3>() = R_base * Eigen::Vector3d(0.0, 0.0, base_vel_b_joy(2));

                const std::vector<std::map<std::string, drc::TaskSpaceData>> task_hierarchy = {
                    ee_data_,
                    {{base_link_name,   base_task_data}}
                };

                is_qp_solved_dual = fr3_husky_model_updater_.robot_controller_->moma.HQPIDStep(
                    task_hierarchy,
                    wheel_acc_desired,
                    fr3_husky_model_updater_.torque_desired_total_,
                    time_verbose_dual);

                if(!is_qp_solved_dual)
                {
                    const auto actuator_idx = fr3_husky_model_updater_.robot_data_->getActuatorIndex();
                    fr3_husky_model_updater_.torque_desired_total_ =
                        fr3_husky_model_updater_.robot_data_->getGravityActuated().segment(
                            actuator_idx.mani_start,
                            fr3_husky_model_updater_.manipulator_dof_);
                    wheel_acc_desired.setZero();
                }
                fr3_husky_model_updater_.wheel_vel_desired_ =
                    fr3_husky_model_updater_.wheel_vel_ +
                    fr3_husky_model_updater_.dt_ * wheel_acc_desired;
                break;
            }
        }

        is_qp_solved = is_qp_solved_dual;
        time_verbose = time_verbose_dual;
    }

    // Command input
    if(model_updater_.HasPositionCommandInterface())
    {
        fr3_husky_model_updater_.writeCommand(fr3_husky_model_updater_.q_desired_total_,
                                              fr3_husky_model_updater_.wheel_vel_desired_);
    }
    else if(model_updater_.HasVelocityCommandInterface())
    {
        fr3_husky_model_updater_.writeCommand(fr3_husky_model_updater_.qdot_desired_total_,
                                              fr3_husky_model_updater_.wheel_vel_desired_);
    }
    else
    {
        fr3_husky_model_updater_.writeCommand(fr3_husky_model_updater_.torque_desired_total_ - fr3_husky_model_updater_.g_total_,
                                              fr3_husky_model_updater_.wheel_vel_desired_);
    }

    prev_button_states_ = button_states_local;

    auto fb = std::make_shared<ActionT::Feedback>();
    fb->is_qp_solved = is_qp_solved;
    fb->time_verbose = time_verbose;
    publishFeedback(fb);
    return ComputeResult::RUNNING;
}

void ViveTracker::onStop(StopReason reason)
{
    model_updater_.haltCommands();

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

ViveTracker::ResultPtr ViveTracker::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();
    result->is_completed = true;
    return result;
}

void ViveTracker::subPoseCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
    const double cutoff_freq = 100.;
    if(msg->poses.size() != NUM_TRACKERS)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of PoseArray for tracker_pose (%ld) does not equal to 3.", name_.c_str(), msg->poses.size());
    }
    else
    {
        for(size_t i = 0; i < msg->poses.size(); ++i)
        {
            Eigen::Vector3d position(msg->poses[i].position.x, msg->poses[i].position.y, msg->poses[i].position.z);
            position = dyros_math::lowPassFilter(position, controller_poses_[i].translation(), fr3_husky_model_updater_.dt_, 1./cutoff_freq);
            Eigen::Quaterniond quaternion(msg->poses[i].orientation.w, msg->poses[i].orientation.x, msg->poses[i].orientation.y, msg->poses[i].orientation.z);
            quaternion.normalize();
            Eigen::Quaterniond prev_quat(controller_poses_[i].linear());
            const double alpha = fr3_husky_model_updater_.dt_ / (fr3_husky_model_updater_.dt_ + (1./cutoff_freq));
            Eigen::Quaterniond filtered_quat = prev_quat.slerp(alpha, quaternion);
            Eigen::Matrix3d orientation = filtered_quat.normalized().toRotationMatrix();
            {
                std::lock_guard<std::mutex> lock(tracker_pose_mutex_);
                controller_poses_[i].translation() = position;
                controller_poses_[i].linear() = orientation;
            }
        }
    }
}

void ViveTracker::subLJoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    if(msg->buttons.size() != NUM_BUTTONS)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of buttons for lhand_joy (%ld) does not equal to %d.", name_.c_str(), msg->buttons.size(), NUM_BUTTONS);
    }
    else if(msg->axes.size() != NUM_AXES)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of axes for lhand_joy (%ld) does not equal to %d.", name_.c_str(), msg->buttons.size(), NUM_AXES);
    }
    else
    {
        for(size_t i = 0; i < msg->buttons.size(); ++i)
        {
            std::lock_guard<std::mutex> lock(joy_mutex_);
            button_states_[IDX_LEFT_CON][i] = (static_cast<int>(msg->buttons[i]) == 0) ? false : true;
        }
        for(size_t i = 0; i < msg->axes.size(); ++i)
        {
            std::lock_guard<std::mutex> lock(joy_mutex_);
            axe_states_[IDX_LEFT_CON][i] = msg->axes[i];
        }

    }
}

void ViveTracker::subRJoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    if(msg->buttons.size() != NUM_BUTTONS)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of buttons for rhand_joy (%ld) does not equal to %d.", name_.c_str(), msg->buttons.size(), NUM_BUTTONS);
    }
    else if(msg->axes.size() != NUM_AXES)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of axes for rhand_joy (%ld) does not equal to %d.", name_.c_str(), msg->buttons.size(), NUM_AXES);
    }
    else
    {
        for(size_t i = 0; i < msg->buttons.size(); ++i)
        {
            std::lock_guard<std::mutex> lock(joy_mutex_);
            button_states_[IDX_RIGHT_CON][i] = (static_cast<int>(msg->buttons[i]) == 0) ? false : true;
        }
        for(size_t i = 0; i < msg->axes.size(); ++i)
        {
            std::lock_guard<std::mutex> lock(joy_mutex_);
            axe_states_[IDX_RIGHT_CON][i] = msg->axes[i];
        }

    }
}

// Register this server into global registry (executed when this TU is linked)
REGISTER_FR3_HUSKY_ACTION_SERVER(ViveTracker, "fr3_husky_vive_tracker")

}  // namespace fr3_husky_controller::servers::fr3_husky
/*
# send goal 
ros2 action send_goal /fr3_husky_vive_tracker fr3_husky_msgs/action/ViveTracker \
"{control_mode: 2, teleop_mode: 3, left_controller_ee_name: 'left_fr3_hand_tcp', right_controller_ee_name: 'right_fr3_hand_tcp', move_orientation: true, controller_pos_multiplier: 1.0, controller_ori_multiplier: 1.0}" \
--feedback
*/
