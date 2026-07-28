#include <fr3_husky_controller/servers/fr3_husky/omega_haptic_action_server.hpp>

#include <cmath>
#include <map>
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

}  // namespace

OmegaHaptic::OmegaHaptic(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_husky_model_updater_(getFR3HuskyModelUpdater(model_updater, name))
{
    // create pub & subscriber
    wrench_pub_       = node_->create_publisher<geometry_msgs::msg::WrenchStamped>("/haptic/wrench_feedback", 1);
    force_pub_        = node_->create_publisher<geometry_msgs::msg::Vector3>("/haptic/force_feedback", 1);
    pose_sub_         = node_->create_subscription<geometry_msgs::msg::PoseStamped>("/haptic/pose", 1, std::bind(&OmegaHaptic::subPoseCallback, this, std::placeholders::_1));
    ori_encoder_sub_  = node_->create_subscription<std_msgs::msg::Float32MultiArray>("/haptic/encoder_orientation", 1, std::bind(&OmegaHaptic::subOriEncoderCallback, this, std::placeholders::_1));
    twist_sub_        = node_->create_subscription<geometry_msgs::msg::Twist>("/haptic/twist", 1, std::bind(&OmegaHaptic::subTwistCallback, this, std::placeholders::_1));
    button_state_sub_ = node_->create_subscription<std_msgs::msg::Int8MultiArray>("/haptic/button_state", 1, std::bind(&OmegaHaptic::subButtonCallback, this, std::placeholders::_1));
    base_vel_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 1, std::bind(&OmegaHaptic::subBaseVelCallback, this, std::placeholders::_1));
    joy_base_vel_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(
        "/joy_teleop/cmd_vel", 1, std::bind(&OmegaHaptic::subBaseVelCallback, this, std::placeholders::_1));
    fr3_husky_base_vel_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(
        "/fr3_husky/cmd_vel", 1, std::bind(&OmegaHaptic::subBaseVelCallback, this, std::placeholders::_1));

    // initialize haptic&joy controller states
    haptic_pose_.setIdentity();
    haptic_vel_.setZero();
    haptic_ori_encoder_.setZero();
    haptic_pose_init_.setIdentity();
    button0_state_ = false;
    haptic_base2robot_base_.setIdentity();
    haptic_base2robot_base_ = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    joy_cmd_vel_.setZero();

    // initialize robot data
    ee_data.clear();
    T_world2mobi_base_.setIdentity();
    T_world2mobi_base_init_.setIdentity();

    fr3_husky_model_updater_.robot_controller_->moma.setIDKvGain({{fr3_husky_model_updater_.robot_data_->getBaseLinkName(), 
                                                          Eigen::Vector6d::Constant(1000.0)}});

    RCLCPP_INFO(node_->get_logger(), "[%s] OmegaHaptic created", name_.c_str());
}

bool OmegaHaptic::acceptGoal(const ActionT::Goal& goal)
{
    if (goal.control_mode < 0 || goal.control_mode > 3)
    {
        RCLCPP_WARN(node_->get_logger(),
                    "[%s] Reject action: control_mode must be 0 to 3 (0: CLIK, 1: OSF, 2: QPIK, 3: QPID). The mode from action goal is %d.",
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

    if(!fr3_husky_model_updater_.robot_data_->hasLinkFrame(goal.ee_name))
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: ee_name from the goal [%s] is not includede in URDF.",
                                         name_.c_str(), goal.ee_name.c_str());
        return false;
    }

    return true;
}

void OmegaHaptic::onGoalAccepted(const ActionT::Goal& goal)
{
    control_mode_ = goal.control_mode;
    teleop_mode_ = goal.teleop_mode;
    control_ee_name_ = goal.ee_name;
    move_ori_ = goal.move_orientation;
    haptic_pos_multiplier_ = static_cast<double>(goal.haptic_pos_multiplier);
    haptic_ori_multiplier_ = static_cast<double>(goal.haptic_ori_multiplier);
    haptic_lin_vel_multiplier_ = static_cast<double>(goal.haptic_lin_vel_multiplier);
    haptic_ang_vel_multiplier_ = static_cast<double>(goal.haptic_ang_vel_multiplier);

    requestActivate();
}

void OmegaHaptic::onStart()
{
    const double current_time = node_->now().seconds();

    {
        std::lock_guard<std::mutex> lock(haptic_pose_mutex_);
        haptic_pose_.setIdentity();
    }
    {
        std::lock_guard<std::mutex> lock(haptic_vel_mutex_);
        haptic_vel_.setZero();
    }
    {
        std::lock_guard<std::mutex> lock(haptic_ori_encoder_mutex_);
        haptic_ori_encoder_.setZero();
    }
    haptic_pose_init_.setIdentity();
    {
        std::lock_guard<std::mutex> lock(button0_state_mutex_);
        button0_state_ = false;
    }
    {
        std::lock_guard<std::mutex> lock(joy_cmd_mutex_);
        joy_cmd_vel_.setZero();
    }
    
    is_mouse_mode_on_ = false;

    ee_data.clear();
    ee_data[control_ee_name_] = drc::TaskSpaceData::Zero();
    ee_data[control_ee_name_].x = fr3_husky_model_updater_.robot_data_->getPose(control_ee_name_);
    ee_data[control_ee_name_].xdot = fr3_husky_model_updater_.robot_data_->getVelocity(control_ee_name_);
    ee_data[control_ee_name_].xddot.setZero();
    ee_data[control_ee_name_].current_time = current_time;
    ee_data[control_ee_name_].setInit();
    ee_data[control_ee_name_].setDesired();

    T_world2mobi_base_ = fr3_husky_model_updater_.robot_data_->getPose(fr3_husky_model_updater_.robot_data_->getBaseLinkName());
    T_world2mobi_base_init_ = T_world2mobi_base_;

    fr3_husky_model_updater_.q_total_init_ = fr3_husky_model_updater_.q_total_;
    fr3_husky_model_updater_.q_desired_total_ = fr3_husky_model_updater_.q_total_;
    fr3_husky_model_updater_.qdot_desired_total_.setZero();
    fr3_husky_model_updater_.wheel_vel_desired_.setZero();

    RCLCPP_INFO(node_->get_logger(), "[%s] started", name_.c_str());
}

OmegaHaptic::ComputeResult OmegaHaptic::compute(const rclcpp::Time& time, const rclcpp::Duration& /*period*/)
{
    // update robot data
    const double current_time = time.seconds();

    ee_data[control_ee_name_].x    = fr3_husky_model_updater_.robot_data_->getPose(control_ee_name_);
    ee_data[control_ee_name_].xdot = fr3_husky_model_updater_.robot_data_->getVelocity(control_ee_name_);
    ee_data[control_ee_name_].xddot.setZero();
    ee_data[control_ee_name_].current_time = current_time;

    T_world2mobi_base_ = fr3_husky_model_updater_.robot_data_->getPose(fr3_husky_model_updater_.robot_data_->getBaseLinkName());

    // get haptic/joy local data 
    Eigen::Affine3d haptic_pose_local;
    Eigen::Vector6d haptic_vel_local;
    Eigen::Vector3d haptic_ori_encoder_local;
    bool button0_state_local = false;
    Eigen::Vector3d joy_cmd_vel_local;

    {
        std::lock_guard<std::mutex> lock(haptic_pose_mutex_);
        haptic_pose_local = haptic_pose_;
    }
    {
        std::lock_guard<std::mutex> lock(haptic_vel_mutex_);
        haptic_vel_local = haptic_vel_;
    }
    {
        std::lock_guard<std::mutex> lock(haptic_ori_encoder_mutex_);
        haptic_ori_encoder_local = haptic_ori_encoder_;
    }
    {
        std::lock_guard<std::mutex> lock(button0_state_mutex_);
        button0_state_local = button0_state_;
    }
    {
        std::lock_guard<std::mutex> lock(joy_cmd_mutex_);
        joy_cmd_vel_local = joy_cmd_vel_;
    }

    bool is_qp_solved = true;
    std::string time_verbose = "";

    // Check mouse mode
    if(!is_mouse_mode_on_ && button0_state_local) // activate mouse mode
    {
        RCLCPP_INFO(node_->get_logger(), "[%s] Mouse Mode activated!", name_.c_str());
        is_mouse_mode_on_ = true;

        haptic_pose_init_ = haptic_pose_local;
        ee_data[control_ee_name_].setInit();
        T_world2mobi_base_init_ = T_world2mobi_base_;
        fr3_husky_model_updater_.q_desired_total_ = fr3_husky_model_updater_.q_total_;
        fr3_husky_model_updater_.qdot_desired_total_.setZero();
    }
    else if(is_mouse_mode_on_ && !button0_state_local) // deactivate mouse mode
    {
        RCLCPP_INFO(node_->get_logger(), "[%s] Mouse Mode deactivated!", name_.c_str());
        is_mouse_mode_on_ = false;

        haptic_pose_init_ = haptic_pose_local;
        ee_data[control_ee_name_].setInit();
        T_world2mobi_base_init_ = T_world2mobi_base_;
        fr3_husky_model_updater_.q_desired_total_ = fr3_husky_model_updater_.q_total_;
        fr3_husky_model_updater_.qdot_desired_total_.setZero();
    }

    // Base Mode: arm holds, base driven by joystick
    if(teleop_mode_ == 0)
    {
        fr3_husky_model_updater_.q_desired_total_ = fr3_husky_model_updater_.q_total_init_;
        fr3_husky_model_updater_.qdot_desired_total_.setZero();
        fr3_husky_model_updater_.torque_desired_total_
            = fr3_husky_model_updater_.robot_controller_->mani.moveJointTorqueStep(fr3_husky_model_updater_.q_desired_total_,
                                                                                   fr3_husky_model_updater_.qdot_desired_total_,
                                                                                   false);
        fr3_husky_model_updater_.wheel_vel_desired_ = fr3_husky_model_updater_.robot_controller_->mobi.VelocityCommand(joy_cmd_vel_local);

        is_qp_solved = true;
        time_verbose = "";
    }

    // Arm Mode: base holds, arm driven by haptic (EE in base frame)
    else if(teleop_mode_ == 1)
    {
        // EE desired expressed in base_link frame
        const Eigen::Affine3d T_mobi_base2ee_init = T_world2mobi_base_init_.inverse() * ee_data[control_ee_name_].x_init;
        Eigen::Affine3d T_mobi_base2ee_des = T_mobi_base2ee_init;
        Eigen::Vector6d target_vel;
        target_vel.setZero();

        if(is_mouse_mode_on_)
        {
            const Eigen::Affine3d T_haptic_init2haptic = haptic_pose_init_.inverse() * haptic_pose_local;
            const Eigen::Matrix3d R_mobi_base2haptic_init = haptic_base2robot_base_.transpose() * haptic_pose_init_.linear();

            // Position: delta added in base_link frame
            T_mobi_base2ee_des.translation() += haptic_pos_multiplier_ * R_mobi_base2haptic_init * T_haptic_init2haptic.translation();

            // Orientation: global rotation in base_link frame (pre-multiply)
            if(move_ori_)
            {
                const Eigen::AngleAxisd aa(T_haptic_init2haptic.linear());
                const Eigen::Matrix3d R_haptic_diff_scaled = (std::abs(aa.angle()) > 1e-10)
                    ? Eigen::AngleAxisd(haptic_ori_multiplier_ * aa.angle(), aa.axis()).toRotationMatrix()
                    : Eigen::Matrix3d::Identity();
                const Eigen::Matrix3d R_delta_in_base = R_mobi_base2haptic_init * R_haptic_diff_scaled * R_mobi_base2haptic_init.transpose();
                T_mobi_base2ee_des.linear() = R_delta_in_base * T_mobi_base2ee_init.linear();
            }
        }

        ee_data[control_ee_name_].x_desired    = T_world2mobi_base_ * T_mobi_base2ee_des;
        ee_data[control_ee_name_].xdot_desired = target_vel;
        ee_data[control_ee_name_].xddot_desired.setZero();

        // calculate target ee_data in mobile base frame 
        // fr3_husky_model_updater_.robot_controller_->mani.... needs EE in mobile base frame, not world frame 
        auto ee_data_mani = ee_data; 
        ee_data_mani[control_ee_name_].x_desired = T_mobi_base2ee_des;
        ee_data_mani[control_ee_name_].xdot_desired = target_vel;
        ee_data_mani[control_ee_name_].xddot_desired.setZero();

        bool is_qp_solved_arm = true;
        std::string time_verbose_arm = "";

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

    // Whole-Body Mode: both base and arm driven by haptic (EE in world frame)
    else if(teleop_mode_ == 2)
    {
        Eigen::Affine3d T_ee_init2ee_des = Eigen::Affine3d::Identity();
        const Eigen::Vector6d target_vel = Eigen::Vector6d::Zero();
        if(is_mouse_mode_on_)
        {
            const Eigen::Affine3d T_haptic_init2haptic = haptic_pose_init_.inverse() * haptic_pose_local;
            const Eigen::Matrix3d R_ee_init2haptic_init = ee_data[control_ee_name_].x_init.linear().transpose() * haptic_base2robot_base_.transpose() * haptic_pose_init_.linear();

            // Position
            T_ee_init2ee_des.translation() = haptic_pos_multiplier_ * R_ee_init2haptic_init * T_haptic_init2haptic.translation();

            // Orientation: using similarity transformation
            if(move_ori_)
            {
                const Eigen::AngleAxisd aa(T_haptic_init2haptic.linear());
                const Eigen::Matrix3d R_haptic_diff_scaled =
                    (std::abs(aa.angle()) > 1e-10)
                    ? Eigen::AngleAxisd(haptic_ori_multiplier_ * aa.angle(), aa.axis()).toRotationMatrix()
                    : Eigen::Matrix3d::Identity();
                T_ee_init2ee_des.linear() = R_ee_init2haptic_init * R_haptic_diff_scaled * R_ee_init2haptic_init.transpose();
            }
        }

        ee_data[control_ee_name_].x_desired    = ee_data[control_ee_name_].x_init * T_ee_init2ee_des;
        ee_data[control_ee_name_].xdot_desired = target_vel;

        bool is_qp_solved_wb = true;
        std::string time_verbose_wb = "";
        Eigen::VectorXd wheel_acc_desired;
        wheel_acc_desired.setZero(fr3_husky_model_updater_.mobile_dof_);

        switch(control_mode_)
        {
            case 0: // CLIK
            {
                fr3_husky_model_updater_.robot_controller_->moma.CLIKStep(ee_data,
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
                fr3_husky_model_updater_.robot_controller_->moma.OSFStep(ee_data,
                                                                         wheel_acc_desired,
                                                                         fr3_husky_model_updater_.torque_desired_total_);
                fr3_husky_model_updater_.wheel_vel_desired_ = fr3_husky_model_updater_.wheel_vel_+ 
                                                              fr3_husky_model_updater_.dt_ * wheel_acc_desired;
                break;
            }
            case 2: // QPIK
            {
                is_qp_solved_wb = fr3_husky_model_updater_.robot_controller_->moma.QPIKStep(ee_data,
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
                is_qp_solved_wb = fr3_husky_model_updater_.robot_controller_->moma.QPIDStep(ee_data,
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
        Eigen::Affine3d T_ee_init2ee_des = Eigen::Affine3d::Identity();
        const Eigen::Vector6d target_vel = Eigen::Vector6d::Zero();
        if(is_mouse_mode_on_)
        {
            const Eigen::Affine3d T_haptic_init2haptic = haptic_pose_init_.inverse() * haptic_pose_local;
            const Eigen::Matrix3d R_ee_init2haptic_init = ee_data[control_ee_name_].x_init.linear().transpose() * haptic_base2robot_base_.transpose() * haptic_pose_init_.linear();

            T_ee_init2ee_des.translation() = haptic_pos_multiplier_ * R_ee_init2haptic_init * T_haptic_init2haptic.translation();

            if(move_ori_)
            {
                const Eigen::AngleAxisd aa(T_haptic_init2haptic.linear());
                const Eigen::Matrix3d R_haptic_diff_scaled =
                    (std::abs(aa.angle()) > 1e-10)
                    ? Eigen::AngleAxisd(haptic_ori_multiplier_ * aa.angle(), aa.axis()).toRotationMatrix()
                    : Eigen::Matrix3d::Identity();
                T_ee_init2ee_des.linear() = R_ee_init2haptic_init * R_haptic_diff_scaled * R_ee_init2haptic_init.transpose();
            }
        }

        ee_data[control_ee_name_].x_desired    = ee_data[control_ee_name_].x_init * T_ee_init2ee_des;
        ee_data[control_ee_name_].xdot_desired = target_vel;
        ee_data[control_ee_name_].xddot_desired.setZero();

        bool is_qp_solved_dual = true;
        std::string time_verbose_dual = "";
        Eigen::VectorXd wheel_acc_desired;
        wheel_acc_desired.setZero(fr3_husky_model_updater_.mobile_dof_);

        const Eigen::Vector2d joy_wheel_vel = fr3_husky_model_updater_.robot_controller_->mobi.VelocityCommand(joy_cmd_vel_local);
        const Eigen::VectorXd null_qdot_mobile = joy_wheel_vel;


        switch(control_mode_)
        {
            case 0: // CLIK
            {
                fr3_husky_model_updater_.robot_controller_->moma.CLIKStep(ee_data,
                                                                          fr3_husky_model_updater_.wheel_vel_desired_,
                                                                          fr3_husky_model_updater_.qdot_desired_total_,
                                                                          null_qdot_mobile);
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
                fr3_husky_model_updater_.robot_controller_->moma.OSFStep(ee_data,
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
                base_task_data.xdot_desired.head<3>() = R_base * Eigen::Vector3d(joy_cmd_vel_local(0), joy_cmd_vel_local(1), 0.0);
                base_task_data.xdot_desired.tail<3>() = R_base * Eigen::Vector3d(0.0, 0.0, joy_cmd_vel_local(2));

                // Priority 1: EE tracking  |  Priority 2: mobile base velocity
                const std::vector<std::map<std::string, drc::TaskSpaceData>> task_hierarchy = {
                    {{control_ee_name_, ee_data[control_ee_name_]}},
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
                base_task_data.xdot_desired.head<3>() = R_base * Eigen::Vector3d(joy_cmd_vel_local(0), joy_cmd_vel_local(1), 0.0);
                base_task_data.xdot_desired.tail<3>() = R_base * Eigen::Vector3d(0.0, 0.0, joy_cmd_vel_local(2));

                const std::vector<std::map<std::string, drc::TaskSpaceData>> task_hierarchy = {
                    {{control_ee_name_, ee_data[control_ee_name_]}},
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

    auto fb = std::make_shared<ActionT::Feedback>();
    fb->is_qp_solved = is_qp_solved;
    fb->time_verbose = time_verbose;
    publishFeedback(fb);
    return ComputeResult::RUNNING;
}

void OmegaHaptic::onStop(StopReason reason)
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

OmegaHaptic::ResultPtr OmegaHaptic::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();
    result->is_completed = true;
    return result;
}

void OmegaHaptic::subPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    const double cutoff_freq = 100.;

    Eigen::Vector3d position(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    position = dyros_math::lowPassFilter(position, haptic_pose_.translation(), fr3_husky_model_updater_.dt_, 1./cutoff_freq);
    Eigen::Quaterniond quaternion(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
    quaternion.normalize();
    Eigen::Quaterniond prev_quat(haptic_pose_.linear());
    const double alpha = fr3_husky_model_updater_.dt_ / (fr3_husky_model_updater_.dt_ + (1./cutoff_freq));
    Eigen::Quaterniond filtered_quat = prev_quat.slerp(alpha, quaternion);
    Eigen::Matrix3d orientation = filtered_quat.normalized().toRotationMatrix();
    {
        std::lock_guard<std::mutex> lock(haptic_pose_mutex_);
        haptic_pose_.translation() = position;
        haptic_pose_.linear() = orientation;
    }
}

void OmegaHaptic::subOriEncoderCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    Eigen::Vector3d ori_encoder(msg->data[0], msg->data[1], msg->data[2]);
    {
        std::lock_guard<std::mutex> lock(haptic_ori_encoder_mutex_);
        haptic_ori_encoder_ = ori_encoder;
    }
}

void OmegaHaptic::subTwistCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    Eigen::Vector6d vel(msg->linear.x, msg->linear.y, msg->linear.z,
                        msg->angular.x, msg->angular.y, msg->angular.z);
    vel = dyros_math::lowPassFilter(vel, haptic_vel_, 0.001, 0.002);
    
    {
        std::lock_guard<std::mutex> lock(haptic_vel_mutex_);
        haptic_vel_ = vel;
    }
}

void OmegaHaptic::subButtonCallback(const std_msgs::msg::Int8MultiArray::SharedPtr msg)
{
    // bool button0_state = static_cast<bool>(msg->data[0]);
    bool button0_state = (static_cast<int>(msg->data[0]) == 0) ? false : true;
    {
        std::lock_guard<std::mutex> lock(button0_state_mutex_);
        button0_state_ = button0_state;
    }
}

void OmegaHaptic::subBaseVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    Eigen::Vector3d vel(msg->linear.x, msg->linear.y, msg->angular.z);
    std::lock_guard<std::mutex> lock(joy_cmd_mutex_);
    joy_cmd_vel_ = dyros_math::lowPassFilter(vel, joy_cmd_vel_, 0.001, 0.002);
}

// Register this server into global registry (executed when this TU is linked)
REGISTER_FR3_HUSKY_ACTION_SERVER(OmegaHaptic, "omega_haptic")

}  // namespace fr3_husky_controller::servers::fr3_husky

/*
# 액션 이름 확인
ros2 action list -t | grep omega_haptic

# send goal 
ros2 action send_goal /omega_haptic fr3_husky_msgs/action/OmegaHaptic \
"{control_mode: 2, teleop_mode: 3, ee_name: 'left_fr3_hand_tcp', move_orientation: false, haptic_pos_multiplier: 1.0, haptic_ori_multiplier: 1.0, haptic_lin_vel_multiplier: 1.0, haptic_ang_vel_multiplier: 1.0}" \
--feedback
*/
