#include <fr3_husky_controller/servers/fr3_husky/keyboard_move_action_server.hpp>
#include <mujoco/mujoco.h>

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <vector>





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


} // namespace


KeyboardMove::KeyboardMove(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_husky_model_updater_(getFR3HuskyModelUpdater(model_updater, name))
{

    ee_data_.clear();

    // Action clients
    move_to_joint_client_ = rclcpp_action::create_client<MoveToJointAction>(node_, "fr3_move_to_joint");
    keyboard_self_client_       = rclcpp_action::create_client<ActionT>(node_, name_);

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
                            "[%s] JTC finished — re-activating KeyboardMove", name_.c_str());
                keyboard_self_client_->async_send_goal(saved_keyboard_goal_);
            }
        });


    RCLCPP_INFO(node_->get_logger(), "[%s] KeyboardMove created", name_.c_str());
}

bool KeyboardMove::acceptGoal(const ActionT::Goal& goal)
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

void KeyboardMove::onGoalAccepted(const ActionT::Goal& goal)
{
    control_mode_ = goal.mode;
    left_controller_ee_name_ = goal.left_controller_ee_name;
    right_controller_ee_name_ = goal.right_controller_ee_name;
    controller_pos_multiplier_ = static_cast<double>(goal.controller_pos_multiplier);
    controller_ori_multiplier_ = static_cast<double>(goal.controller_ori_multiplier);
    saved_keyboard_goal_ = goal;


    requestActivate();
}


void KeyboardMove::onStart()
{   

    ee_data_.clear();
    waiting_for_jtc_.store(false, std::memory_order_relaxed);

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

    // init keyboard
    keypress_left_arm_left_ = keypress_left_arm_right_ = keypress_left_arm_forward_ = keypress_left_arm_backward_ = false;
    keypress_right_arm_left_ = keypress_right_arm_right_ = keypress_right_arm_forward_ = keypress_right_arm_backward_ = false;
    keypress_left_arm_upward_ = keypress_left_arm_downward_ = keypress_right_arm_upward_ = keypress_right_arm_downward_ = false;
    keypress_left_arm_gripper_ = keypress_right_arm_gripper_ = false;
    last_key_time_ = 0.0;  


    selected_keyboard_arm_ = KeyboardArm::RIGHT;
    selected_keyboard_direction_ = KeyboardDirection::NONE;
    keyboard_sign_ = 0;
    keyboard_gripper_toggle_ = false;
    prev_keyboard_gripper_toggle_ = false;

    if (!keyboard_sub_)
    {
        keyboard_sub_ = node_->create_subscription<std_msgs::msg::String>(
            "/keyboard_cmd",
            rclcpp::QoS(10),
            std::bind(&KeyboardMove::onKeyboardCommand, this, std::placeholders::_1));
    }

    x_target_l_ = fr3_husky_model_updater_.robot_data_->getPose(left_controller_ee_name_);
    x_target_r_ = fr3_husky_model_updater_.robot_data_->getPose(right_controller_ee_name_);
    x_goal_l_ = fr3_husky_model_updater_.robot_data_->getPose(left_controller_ee_name_);
    x_goal_r_ = fr3_husky_model_updater_.robot_data_->getPose(right_controller_ee_name_);

    RCLCPP_INFO(node_->get_logger(), "[%s] started", name_.c_str());
}

void KeyboardMove::clearKeyboardFlags()
{
    keyboard_sign_ = 0;
    keyboard_gripper_toggle_ = false;
}
void KeyboardMove::onKeyboardCommand(const std_msgs::msg::String::SharedPtr msg)
{
    if (!msg) return;

    std::lock_guard<std::mutex> lk(lock_);
    clearKeyboardFlags();

    const std::string& key = msg->data;

    if (key == "1") selected_keyboard_arm_ = KeyboardArm::LEFT;
    else if (key == "2") selected_keyboard_arm_ = KeyboardArm::RIGHT;
    else if (key == "a") selected_keyboard_direction_ = KeyboardDirection::X;
    else if (key == "s") selected_keyboard_direction_ = KeyboardDirection::Y;
    else if (key == "space") selected_keyboard_direction_ = KeyboardDirection::Z;
    else if (key == "i") selected_keyboard_direction_ = KeyboardDirection::ROLL;
    else if (key == "o") selected_keyboard_direction_ = KeyboardDirection::PITCH;
    else if (key == "p") selected_keyboard_direction_ = KeyboardDirection::YAW;
    else if (key == "up") keyboard_sign_ = 1;
    else if (key == "down") keyboard_sign_ = -1;
    else if (key == "g") keyboard_gripper_toggle_ = true;

    last_key_time_ = now_sec_();
}
KeyboardMove::ComputeResult KeyboardMove::compute(const rclcpp::Time& time, const rclcpp::Duration& /*period*/)
{

    total_elapsed_steps++;
    dbg_cnt++;

    for(auto& [ee_name, ee_data] : ee_data_)
    {
        ee_data.x = fr3_husky_model_updater_.robot_data_->getPose(ee_name);
        ee_data.xdot = fr3_husky_model_updater_.robot_data_->getVelocity(ee_name);
        ee_data.xddot.setZero();
    }




    // ============================================================
    // 0) Keyboard input
    // ============================================================
    KeyboardArm selected_arm;
    KeyboardDirection selected_direction;
    int sign = 0;
    bool gripper_toggle = false;

    {
        std::lock_guard<std::mutex> lk(lock_);
        selected_arm = selected_keyboard_arm_;
        selected_direction = selected_keyboard_direction_;
        sign = keyboard_sign_;
        gripper_toggle = keyboard_gripper_toggle_;
    }

    const double tnow = now_sec_();
    const bool key_active = (tnow - last_key_time_) < key_timeout_;
    const bool user_cmd = key_active && sign != 0 && selected_direction != KeyboardDirection::NONE;

    const double pos_step = 0.15 * fr3_husky_model_updater_.dt_;
    const double rot_step = 1.0 * fr3_husky_model_updater_.dt_;



    // ============================================================
    // 2) goal 갱신(입력 있는 경우에만) + target 생성(smoothing / hold latch)
    // ============================================================
    if (user_cmd)
    {
        Eigen::Affine3d& x_goal =
            selected_arm == KeyboardArm::LEFT ? x_goal_l_ : x_goal_r_;

        switch (selected_direction)
        {
            case KeyboardDirection::X:
                x_goal.translation().x() += sign * pos_step;
                break;
            case KeyboardDirection::Y:
                x_goal.translation().y() += sign * pos_step;
                break;
            case KeyboardDirection::Z:
                x_goal.translation().z() += sign * pos_step;
                break;
            case KeyboardDirection::ROLL:
                x_goal.linear() = x_goal.linear() * Eigen::AngleAxisd(sign * rot_step, Eigen::Vector3d::UnitX()).toRotationMatrix();
                break;
            case KeyboardDirection::PITCH:
                x_goal.linear() = x_goal.linear() * Eigen::AngleAxisd(sign * rot_step, Eigen::Vector3d::UnitY()).toRotationMatrix();
                break;
            case KeyboardDirection::YAW:
                x_goal.linear() = x_goal.linear() * Eigen::AngleAxisd(sign * rot_step, Eigen::Vector3d::UnitZ()).toRotationMatrix();
                break;
            default:
                break;
        }
    }


    const double alpha = 0.25;

    x_target_l_.translation() = (1.0 - alpha) * x_target_l_.translation() + alpha * x_goal_l_.translation();
    x_target_r_.translation() = (1.0 - alpha) * x_target_r_.translation() + alpha * x_goal_r_.translation();

    x_target_l_.linear() = x_goal_l_.linear();
    x_target_r_.linear() = x_goal_r_.linear();

    if (!left_controller_ee_name_.empty())
    {
        ee_data_[left_controller_ee_name_].x_desired = x_target_l_;
        ee_data_[left_controller_ee_name_].xdot_desired.setZero();
    }

    if (!right_controller_ee_name_.empty())
    {
        ee_data_[right_controller_ee_name_].x_desired = x_target_r_;
        ee_data_[right_controller_ee_name_].xdot_desired.setZero();
    }



    // ============================================================
    // 3) gripper control
    // ============================================================


    // Gripper control
    {
        const bool gripper_pressed = key_active && gripper_toggle && !prev_keyboard_gripper_toggle_;
        prev_keyboard_gripper_toggle_ = key_active && gripper_toggle;
        

        // Left controller
        if (!left_controller_ee_name_.empty())
        {   

            // TODO:   gripper toggle key pressed
            if (gripper_pressed && selected_arm == KeyboardArm::LEFT)
            {
                const std::string robot_name = getRobotNameFromEEName(left_controller_ee_name_);
                if (!robot_name.empty())
                {
                    is_gripper_mode_on_[IDX_LEFT_CON] = !is_gripper_mode_on_[IDX_LEFT_CON];
                    if (is_gripper_mode_on_[IDX_LEFT_CON])
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] lhand trigger released → GripperGrasp('%s')", name_.c_str(), robot_name.c_str());
                        fr3_husky_model_updater_.GripperGrasp(robot_name, 0.0, 0.1, 100.0);

                    }
                    else
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] lhand trigger released → GripperOpen('%s')", name_.c_str(), robot_name.c_str());


                        fr3_husky_model_updater_.GripperOpen(robot_name, 0.1);
                    }
                }
            }
        }

        // Right controller
        if (!right_controller_ee_name_.empty())
        {

            // TODO:   gripper toggle key pressed
            if (gripper_pressed && selected_arm == KeyboardArm::RIGHT)
            {
                const std::string robot_name = getRobotNameFromEEName(right_controller_ee_name_);
                if (!robot_name.empty())
                {
                    is_gripper_mode_on_[IDX_RIGHT_CON] = !is_gripper_mode_on_[IDX_RIGHT_CON];
                    if (is_gripper_mode_on_[IDX_RIGHT_CON])
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] rhand trigger released → GripperGrasp('%s')", name_.c_str(), robot_name.c_str());
                        fr3_husky_model_updater_.GripperGrasp(robot_name, 0.0, 0.1, 100.0);

                    }
                    else
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] rhand trigger released → GripperOpen('%s')", name_.c_str(), robot_name.c_str());


                        fr3_husky_model_updater_.GripperOpen(robot_name, 0.1);
                    }
                }
            }
        }
    }

    bool is_qp_solved = true;
    std::string time_verbose = "";

    Eigen::VectorXd qdot_mobile = Eigen::VectorXd::Zero(fr3_husky_model_updater_.mobile_dof_);

    switch (control_mode_)
    {
        case 0: // CLIK
            {   
                const Eigen::VectorXd null_qdot_mani =
                    Eigen::VectorXd::Zero(fr3_husky_model_updater_.manipulator_dof_);

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
                // const Eigen::VectorXd null_qdot_mobile =
                //     fr3_husky_model_updater_.robot_controller_->MobileVelocityCommand(mobile_null_gain * base_vel_null);

                const Eigen::VectorXd null_qdot_mobile =
                    Eigen::VectorXd::Zero(fr3_husky_model_updater_.mobile_dof_);

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
                Eigen::VectorXd null_torque(fr3_husky_model_updater_.robot_data_->getActuatorDof());
                null_torque.segment(
                    fr3_husky_model_updater_.robot_data_->getActuatorIndex().mani_start,
                    fr3_husky_model_updater_.manipulator_dof_) =
                    Eigen::VectorXd::Zero(fr3_husky_model_updater_.manipulator_dof_);
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

    fr3_husky_model_updater_.writeCommand(fr3_husky_model_updater_.torque_desired_total_ - fr3_husky_model_updater_.g_total_,
        fr3_husky_model_updater_.wheel_vel_desired_);  // robot_controller automatically add gravity force

    auto fb = std::make_shared<ActionT::Feedback>();
    fb->is_qp_solved = is_qp_solved;
    fb->time_verbose = time_verbose;
    publishFeedback(fb);


    return ComputeResult::RUNNING;
}


void KeyboardMove::onStop(StopReason reason)
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

KeyboardMove::ResultPtr KeyboardMove::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();
    result->is_completed = true;
    return result;
}







// Register this server into global registry (executed when this TU is linked)
REGISTER_FR3_HUSKY_ACTION_SERVER(KeyboardMove, "fr3_husky_keyboard_move")


}  // namespace fr3_husky_controller::servers::fr3
