#include <fr3_husky_controller/servers/fr3/vive_tracker_action_server.hpp>

#include <stdexcept>

namespace fr3_husky_controller::servers::fr3
{

namespace
{
FR3ModelUpdater& getFR3ModelUpdater(ModelUpdaterBase& model_updater, const std::string& server_name)
{
    auto* fr3_model_updater = dynamic_cast<FR3ModelUpdater*>(&model_updater);
    if (!fr3_model_updater)
    {
        throw std::runtime_error("[" + server_name + "] requires FR3ModelUpdater");
    }
    return *fr3_model_updater;
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
  fr3_model_updater_(getFR3ModelUpdater(model_updater, name))
{
    pose_sub_         = node_->create_subscription<geometry_msgs::msg::PoseArray>("tracker_pose", 1, std::bind(&ViveTracker::subPoseCallback, this, std::placeholders::_1));
    l_joy_sub_ = node_->create_subscription<sensor_msgs::msg::Joy>("lhand_joy", 1, std::bind(&ViveTracker::subLJoyCallback, this, std::placeholders::_1));
    r_joy_sub_ = node_->create_subscription<sensor_msgs::msg::Joy>("rhand_joy", 1, std::bind(&ViveTracker::subRJoyCallback, this, std::placeholders::_1));

    controller_poses_.assign(NUM_TRACKERS, Eigen::Affine3d::Identity());
    controller_poses_init_.assign(NUM_TRACKERS, Eigen::Affine3d::Identity());
    button_states_.assign(NUM_CONTROLLERS, std::vector<bool>(NUM_BUTTONS, false));
    prev_button_states_.assign(NUM_CONTROLLERS, std::vector<bool>(NUM_BUTTONS, false));

    tracker_base2robot_base_.assign(NUM_CONTROLLERS, Eigen::Matrix3d::Identity());

    ee_data_.clear();

    for(const auto& frame : fr3_model_updater_.robot_data_->getLinkFrameVector())
    {
        if (frame.find("wheel") != std::string::npos) 
        {
            has_mobile_ = true;
            break;
        }
    }

    // Action clients
    move_to_joint_client_ = rclcpp_action::create_client<MoveToJointAction>(node_, "fr3_move_to_joint");
    vt_self_client_       = rclcpp_action::create_client<ActionT>(node_, name_);

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
                            "[%s] JTC finished — re-activating ViveTracker", name_.c_str());
                vt_self_client_->async_send_goal(saved_vive_goal_);
            }
        });

    // Initialize franka hand state
    for(const auto& robot_name : model_updater_.robot_names_) fr3_model_updater_.GripperHoming(robot_name); 

    RCLCPP_INFO(node_->get_logger(), "[%s] ViveTracker created", name_.c_str());
}

bool ViveTracker::acceptGoal(const ActionT::Goal& goal)
{
    if (!model_updater_.HasEffortCommandInterface())
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: effort command interface is required",
                                         name_.c_str());
        return false;
    }

    if (goal.control_mode < 0 || goal.control_mode > 3)
    {
        RCLCPP_WARN(node_->get_logger(),
                                         "[%s] Reject action: mode must be 0 to 3 (0: CLIK, 1: OSF, 2: QPIK, 3: QPID). The mode from action goal is %d.",
                                         name_.c_str(),
                                         static_cast<int>(goal.control_mode));
        return false;
    }

    if(!goal.left_controller_ee_name.empty() && !fr3_model_updater_.robot_data_->hasLinkFrame(goal.left_controller_ee_name))
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: left_controller_ee_name from the goal [%s] is not includede in URDF.",
                                         name_.c_str(), goal.left_controller_ee_name.c_str());
        return false;
    }

    if(!goal.right_controller_ee_name.empty() && !fr3_model_updater_.robot_data_->hasLinkFrame(goal.right_controller_ee_name))
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
    left_controller_ee_name_ = goal.left_controller_ee_name;
    right_controller_ee_name_ = goal.right_controller_ee_name;
    move_ori_ = goal.move_orientation;
    controller_pos_multiplier_ = static_cast<double>(goal.controller_pos_multiplier);
    controller_ori_multiplier_ = static_cast<double>(goal.controller_ori_multiplier);
    saved_vive_goal_ = goal;


    requestActivate();
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
        std::lock_guard<std::mutex> lock(button_state_mutex_);
        for(auto& button_state : button_states_) button_state = std::vector<bool>(NUM_BUTTONS, false);
    }

    for(auto& prev_button_state : prev_button_states_) prev_button_state = std::vector<bool>(NUM_BUTTONS, false);
    is_mouse_mode_on_.assign(NUM_CONTROLLERS, false);
    is_initialize_mode_on_ = false;
    is_gripper_mode_on_.assign(NUM_CONTROLLERS, false);
    ee_data_.clear();
    waiting_for_jtc_.store(false, std::memory_order_relaxed);

    if(!left_controller_ee_name_.empty())
    {
        ee_data_[left_controller_ee_name_] = drc::TaskSpaceData::Zero();
        ee_data_[left_controller_ee_name_].x = fr3_model_updater_.robot_data_->getPose(left_controller_ee_name_);
        ee_data_[left_controller_ee_name_].xdot = fr3_model_updater_.robot_data_->getVelocity(left_controller_ee_name_);
        ee_data_[left_controller_ee_name_].xddot.setZero();
        ee_data_[left_controller_ee_name_].current_time = current_time;
        ee_data_[left_controller_ee_name_].setInit();
        ee_data_[left_controller_ee_name_].setDesired();
    }
    if(!right_controller_ee_name_.empty())
    {
        ee_data_[right_controller_ee_name_] = drc::TaskSpaceData::Zero();
        ee_data_[right_controller_ee_name_].x = fr3_model_updater_.robot_data_->getPose(right_controller_ee_name_);
        ee_data_[right_controller_ee_name_].xdot = fr3_model_updater_.robot_data_->getVelocity(right_controller_ee_name_);
        ee_data_[right_controller_ee_name_].xddot.setZero();
        ee_data_[right_controller_ee_name_].current_time = current_time;
        ee_data_[right_controller_ee_name_].setInit();
        ee_data_[right_controller_ee_name_].setDesired();
    }

    RCLCPP_INFO(node_->get_logger(), "[%s] started", name_.c_str());
}

ViveTracker::ComputeResult ViveTracker::compute(const rclcpp::Time& time, const rclcpp::Duration& /*period*/)
{
    const double current_time = time.seconds();

    for(auto& [ee_name, ee_data] : ee_data_)
    {
        ee_data.x = fr3_model_updater_.robot_data_->getPose(ee_name);
        ee_data.xdot = fr3_model_updater_.robot_data_->getVelocity(ee_name);
        ee_data.xddot.setZero();
        ee_data.current_time = current_time;
    }

    std::vector<Eigen::Affine3d> controller_poses_local;   // left, right, head
    std::vector<std::vector<bool>> button_states_local;    // [left, right][trigger, grip, a, b]
    {
        std::lock_guard<std::mutex> lock(tracker_pose_mutex_);
        controller_poses_local = controller_poses_;
    }
    {
        std::lock_guard<std::mutex> lock(button_state_mutex_);
        button_states_local = button_states_;
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
                        if(model_updater_.num_robots_ == 2 && has_mobile_ &&  j == 0)
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
                    [this](const rclcpp_action::ClientGoalHandle<MoveToJointAction>::WrappedResult&)
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] MoveToJoint done — waiting for JTC to finish", name_.c_str());
                        waiting_for_jtc_.store(true, std::memory_order_relaxed);
                    };

                move_to_joint_client_->async_send_goal(mtj_goal, send_opts);
                RCLCPP_INFO(node_->get_logger(),
                            "[%s] Initialize mode ON — goal sent to fr3_move_to_joint, yielding",
                            name_.c_str());

                // Yield: deactivate ViveTracker so MoveToJoint can become active_server_
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
                        fr3_model_updater_.GripperGrasp(robot_name);
                    }
                    else
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] lhand trigger released → GripperOpen('%s')", name_.c_str(), robot_name.c_str());
                        fr3_model_updater_.GripperOpen(robot_name);
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
                        fr3_model_updater_.GripperGrasp(robot_name);
                    }
                    else
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] rhand trigger released → GripperOpen('%s')", name_.c_str(), robot_name.c_str());
                        fr3_model_updater_.GripperOpen(robot_name);
                    }
                }
            }
        }
    }

    // Manipulator control
    {
        // Check mouse mode
        for(size_t i = 0; i < NUM_CONTROLLERS; ++i)
        {
            if(!is_mouse_mode_on_[i] && button_states_local[i][IDX_GRIP_BUTTON]) // activate mouse mode when "grip" button pressed
            {
                RCLCPP_INFO(node_->get_logger(), "[%s] %s Mouse Mode activated!", name_.c_str(), (i==0)?"Left":"Right");
                is_mouse_mode_on_[i] = true;
    
                controller_poses_init_[i] = controller_poses_local[i];
                if(i == 0 && !left_controller_ee_name_.empty())       ee_data_[left_controller_ee_name_].setInit();
                else if(i == 1 && !right_controller_ee_name_.empty()) ee_data_[right_controller_ee_name_].setInit();
            }
            else if(is_mouse_mode_on_[i] && !button_states_local[i][IDX_GRIP_BUTTON]) // deactivate mouse mode when grip button released
            {
                RCLCPP_INFO(node_->get_logger(), "[%s] %s Mouse Mode deactivated!", name_.c_str(), (i==0)?"Left":"Right");
                is_mouse_mode_on_[i] = false;
    
                controller_poses_init_[i] = controller_poses_local[i];
                if(i == IDX_LEFT_CON && !left_controller_ee_name_.empty())        ee_data_[left_controller_ee_name_].setInit();
                else if(i == IDX_RIGHT_CON && !right_controller_ee_name_.empty()) ee_data_[right_controller_ee_name_].setInit();
            }
        }
    
        
        if(!left_controller_ee_name_.empty()) // left vive controller
        {
            Eigen::Affine3d target_pose_diff; // EE init -> EE desired
            Eigen::Vector6d target_vel;
            target_pose_diff.setIdentity();
            target_vel.setZero();
            if(is_mouse_mode_on_[IDX_LEFT_CON])
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
    
            ee_data_[left_controller_ee_name_].x_desired = ee_data_[left_controller_ee_name_].x_init * target_pose_diff;
            ee_data_[left_controller_ee_name_].xdot_desired  = target_vel;
        }
    
        if(!right_controller_ee_name_.empty()) // right vive controller
        {
            Eigen::Affine3d target_pose_diff; // EE init -> EE desired
            Eigen::Vector6d target_vel;
            target_pose_diff.setIdentity();
            target_vel.setZero();
            if(is_mouse_mode_on_[IDX_RIGHT_CON])
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
    
            ee_data_[right_controller_ee_name_].x_desired = ee_data_[right_controller_ee_name_].x_init * target_pose_diff;
            ee_data_[right_controller_ee_name_].xdot_desired  = target_vel;
        }
    
        bool is_qp_solved = true;
        std::string time_verbose = "";
        switch (control_mode_)
        {
            case 0: // CLIK
                fr3_model_updater_.robot_controller_->CLIKStep(ee_data_, fr3_model_updater_.qdot_desired_total_);
                fr3_model_updater_.q_desired_total_ = fr3_model_updater_.q_total_ +
                                                      fr3_model_updater_.dt_ * fr3_model_updater_.qdot_desired_total_;
                
                fr3_model_updater_.torque_desired_total_ = fr3_model_updater_.robot_controller_->moveJointTorqueStep(fr3_model_updater_.q_desired_total_,
                                                                                                                     fr3_model_updater_.qdot_desired_total_,
                                                                                                                     false);
                break;
            case 1: // OSF
                fr3_model_updater_.robot_controller_->OSFStep(ee_data_, fr3_model_updater_.torque_desired_total_);
                break;
            case 2: // QPIK
                is_qp_solved = fr3_model_updater_.robot_controller_->QPIKStep(ee_data_, fr3_model_updater_.qdot_desired_total_, time_verbose);
                if(!is_qp_solved) fr3_model_updater_.qdot_desired_total_.setZero();
                fr3_model_updater_.q_desired_total_ = fr3_model_updater_.q_total_ +
                                                      fr3_model_updater_.dt_ * fr3_model_updater_.qdot_desired_total_;
                fr3_model_updater_.torque_desired_total_ = fr3_model_updater_.robot_controller_->moveJointTorqueStep(fr3_model_updater_.q_desired_total_,
                                                                                                                     fr3_model_updater_.qdot_desired_total_,
                                                                                                                     false);
                break;
            case 3: // QPID
                is_qp_solved = fr3_model_updater_.robot_controller_->QPIDStep(ee_data_, fr3_model_updater_.torque_desired_total_, time_verbose);
                if(!is_qp_solved) fr3_model_updater_.torque_desired_total_ = fr3_model_updater_.robot_data_->getGravity();
                break;
            default:
                break;
        }
    
        fr3_model_updater_.writeCommand(fr3_model_updater_.torque_desired_total_ - fr3_model_updater_.g_total_); // robot_controller automatically add gravity force
    
        auto fb = std::make_shared<ActionT::Feedback>();
        fb->is_qp_solved = is_qp_solved;
        fb->time_verbose = time_verbose;
        publishFeedback(fb);
        
        prev_button_states_ = button_states_;
        return ComputeResult::RUNNING;
    }
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
            position = dyros_math::lowPassFilter(position, controller_poses_[i].translation(), fr3_model_updater_.dt_, 1./cutoff_freq);
            Eigen::Quaterniond quaternion(msg->poses[i].orientation.w, msg->poses[i].orientation.x, msg->poses[i].orientation.y, msg->poses[i].orientation.z);
            quaternion.normalize();
            Eigen::Quaterniond prev_quat(controller_poses_[i].linear());
            const double alpha = fr3_model_updater_.dt_ / (fr3_model_updater_.dt_ + (1./cutoff_freq));
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
    else
    {
        for(size_t i = 0; i < msg->buttons.size(); ++i)
        {
            std::lock_guard<std::mutex> lock(button_state_mutex_);
            button_states_[IDX_LEFT_CON][i] = (static_cast<int>(msg->buttons[i]) == 0) ? false : true;
        }

    }
}

void ViveTracker::subRJoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    if(msg->buttons.size() != NUM_BUTTONS)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of buttons for rhand_joy (%ld) does not equal to %d.", name_.c_str(), msg->buttons.size(), NUM_BUTTONS);
    }
    else
    {
        for(size_t i = 0; i < msg->buttons.size(); ++i)
        {
            std::lock_guard<std::mutex> lock(button_state_mutex_);
            button_states_[IDX_RIGHT_CON][i] = (static_cast<int>(msg->buttons[i]) == 0) ? false : true;
        }

    }
}


// Register this server into global registry (executed when this TU is linked)
REGISTER_FR3_ACTION_SERVER(ViveTracker, "fr3_vive_tracker")

}  // namespace fr3_husky_controller::servers::fr3
/*
# send goal 
ros2 action send_goal /fr3_vive_tracker fr3_husky_msgs/action/ViveTracker \
"{mode: 1, left_controller_ee_name: 'left_fr3_hand_tcp', right_controller_ee_name: 'right_fr3_hand_tcp', move_orientation: false, controller_pos_multiplier: 1.0, controller_ori_multiplier: 1.0}" \
--feedback
*/