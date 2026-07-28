#pragma once

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

#include <Eigen/Eigen>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <hardware_interface/loaned_command_interface.hpp>
#include <hardware_interface/loaned_state_interface.hpp>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <realtime_tools/realtime_buffer.h>

#include "fr3_husky_controller/model/fr3_model_updater.hpp"

#include "dyros_robot_controller/mobile_manipulator/robot_data.h"
#include "dyros_robot_controller/mobile_manipulator/robot_controller.h"

namespace fr3_husky_controller
{

class FR3HuskyModelUpdater final : public ModelUpdaterBase
{
    private:
        void ensureFilteredOdometrySubscription();

        double wheel_encoder_multiplier_ = 1.0;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr filtered_odometry_sub_ = nullptr;
        realtime_tools::RealtimeBuffer<std::shared_ptr<nav_msgs::msg::Odometry>> filtered_odometry_buf_;

    public:
        FR3HuskyModelUpdater() = default;

        bool initialize(size_t num_robots,
                        size_t manipulator_dof,
                        double dt,
                        const std::vector<std::string>& robot_names,
                        const std::vector<std::string>& ee_names) override;
        void setDRCRobotData(const std::shared_ptr<drc::MobileManipulator::RobotData>&& robot_data);
        void setDRCRobotController(const std::shared_ptr<drc::MobileManipulator::RobotController>&& robot_controller) { robot_controller_ = std::move(robot_controller); }
        void setWheelEncoderMultiplier(double m) { wheel_encoder_multiplier_ = m; }
        void setSubtractGravityFromEffortCommand(bool enabled) { subtract_gravity_from_effort_command_ = enabled; }
        void updateJointStates() override;
        void updateRobotData() override;
        void haltCommands() override;
        bool getHandlesReady() const override { return !(robot_handle_.mani_joints.empty() || robot_handle_.left_wheels.empty() || robot_handle_.right_wheels.empty()); }
        void setInitFromCurrent();
        void writeCommand(const Eigen::VectorXd& command_mani,
                        const Eigen::Vector2d& command_mobi);
        void forceStopMobile();
        bool GripperMove(const std::string robot_name, double width, double speed);
        bool GripperOpen(const std::string robot_name, double speed = 0.1)  { return GripperMove(robot_name, 0.08, speed); }
        bool GripperClose(const std::string robot_name, double speed = 0.1) { return GripperMove(robot_name, 0.00, speed); }
        bool GripperHoming(const std::string robot_name);
        bool GripperGrasp(const std::string robot_name, double width = 0.0, double speed = 0.1, double force = 30.0, std::pair<double, double> epsilon = {0.08, 0.08});
        Eigen::VectorXd getJointTorque(const std::string robot_name) {return torque_[robot_name];}
    
    public:
        bool subtract_gravity_from_effort_command_{true};
        std::shared_ptr<drc::MobileManipulator::RobotData> robot_data_;
        std::shared_ptr<drc::MobileManipulator::RobotController> robot_controller_;

        std::map<std::string, Eigen::Affine3d>  T_mobibase2manibase_; // basefootprint -> fr3_link0
        std::map<std::string, int> mani_joint_q_idx_; // starting index of manipulator q joint
        std::map<std::string, int> mani_joint_v_idx_; // starting index of manipulator v joint
        // ========================================================================
        // ======================= Manipulator State Data =========================
        // ========================================================================
        // Initial
        std::map<std::string, Eigen::Vector<double, FR3_DOF>> q_init_;
        std::map<std::string, Eigen::Vector<double, FR3_DOF>> qdot_init_;
        std::map<std::string, Eigen::Vector<double, FR3_DOF>> qddot_init_;
        Eigen::VectorXd q_total_init_;
        Eigen::VectorXd qdot_total_init_;
        Eigen::VectorXd qddot_total_init_;

        // Current
        std::map<std::string, Eigen::Vector<double, FR3_DOF>> q_;
        std::map<std::string, Eigen::Vector<double, FR3_DOF>> qdot_;
        std::map<std::string, Eigen::Vector<double, FR3_DOF>> qddot_;
        std::map<std::string, Eigen::Vector<double, FR3_DOF>> torque_;
        Eigen::VectorXd q_total_;
        Eigen::VectorXd qdot_total_;
        Eigen::VectorXd qddot_total_;
        Eigen::VectorXd torque_total_;

        // Desired
        std::map<std::string, Eigen::Vector<double, FR3_DOF>> q_desired_;
        std::map<std::string, Eigen::Vector<double, FR3_DOF>> qdot_desired_;
        std::map<std::string, Eigen::Vector<double, FR3_DOF>> qddot_desired_;
        std::map<std::string, Eigen::Vector<double, FR3_DOF>> torque_desired_;
        Eigen::VectorXd q_desired_total_;
        Eigen::VectorXd qdot_desired_total_;
        Eigen::VectorXd qddot_desired_total_;
        Eigen::VectorXd torque_desired_total_;

        // Dynamics
        std::map<std::string, Eigen::Matrix<double, FR3_DOF, FR3_DOF>> M_;
        std::map<std::string, Eigen::Matrix<double, FR3_DOF, FR3_DOF>> M_inv_;
        std::map<std::string, Eigen::Vector<double, FR3_DOF>> g_;
        std::map<std::string, Eigen::Vector<double, FR3_DOF>> c_;
        Eigen::MatrixXd M_total_;
        Eigen::MatrixXd M_inv_total_;
        Eigen::VectorXd g_total_;
        Eigen::VectorXd c_total_;
        
        // ========================================================================
        // ============================= Task Space Data ==========================
        // ========================================================================
        // Initial
        Eigen::Affine2d  base_pose_w_init_;                         // mobile base pose wrt world frame
        Eigen::Vector3d  base_vel_w_init_;                          // mobile base velocity wrt world frame
        Eigen::Vector3d  base_vel_b_init_;                          // mobile base velocity wrt mobile base frame
        std::map<std::string, Eigen::Affine3d>  x_m_init_;          // EE pose wrt manipulator base frame
        std::map<std::string, Eigen::Vector6d>  xdot_m_init_;       // EE velocity wrt manipulator base frame
        std::map<std::string, Eigen::Affine3d>  x_w_init_;          // EE pose wrt world frame
        std::map<std::string, Eigen::Vector6d>  xdot_w_init_;       // EE velocity wrt world frame

        // Current
        Eigen::Affine2d  base_pose_w_;                               // mobile base pose wrt world frame (filtered if /odometry/filtered available, else wheel)
        Eigen::Vector3d  base_vel_w_;                                // mobile base velocity wrt world frame
        Eigen::Vector3d  base_vel_b_;                                // mobile base velocity wrt mobile base frame (filtered if available)
        Eigen::Affine2d  base_pose_w_wheel_;                         // mobile base pose from raw wheel odometry only (for /odom publishing)
        Eigen::Vector3d  base_vel_b_wheel_;                          // mobile base velocity from raw wheel odometry only (for /odom publishing)
        std::map<std::string, Eigen::Affine3d>  x_m_;                // EE pose wrt manipulator base frame       
        std::map<std::string, Eigen::Vector6d>  xdot_m_;             // EE velocity wrt manipulator base frame     
        std::map<std::string, Eigen::Affine3d>  x_w_;                // EE pose wrt world frame         
        std::map<std::string, Eigen::Vector6d>  xdot_w_;             // EE velocity wrt world frame         
        std::map<std::string, Eigen::Matrix<double, 6, FR3_DOF>> J_; // EE jacobian wrt manipulator base frame

        // Desired
        Eigen::Affine2d  base_pose_w_desired_;                       // mobile base pose wrt world frame 
        Eigen::Vector3d  base_vel_w_desired_;                        // mobile base velocity wrt world frame  
        Eigen::Vector3d  base_vel_b_desired_;                        // mobile base velocity wrt mobile base frame   
        std::map<std::string, Eigen::Affine3d>  x_m_desired_;        // EE pose wrt manipulator base frame   
        std::map<std::string, Eigen::Vector6d>  xdot_m_desired_;     // EE velocity wrt manipulator base frame   
        std::map<std::string, Eigen::Affine3d>  x_w_desired_;        // EE pose wrt world frame 
        std::map<std::string, Eigen::Vector6d>  xdot_w_desired_;     // EE velocity wrt world frame  
        
        // ========================================================================
        // ======================== Mobile Base State Data ========================
        // ========================================================================
        Eigen::Vector2d wheel_pos_;
        Eigen::Vector2d wheel_vel_;

        Eigen::Vector2d wheel_vel_desired_;

        

        // ========================================================================
        // ===============================  Debugging =============================
        // ========================================================================
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr command_mani_debug_pub_;
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr command_mobi_debug_pub_;
        rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr x_m_pub_;
        rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr xdot_m_l_pub_, xdot_m_r_pub_;
        rclcpp::TimerBase::SharedPtr debug_publish_timer_;
        void publishDebugState();

};

}  // namespace fr3_husky_controller
