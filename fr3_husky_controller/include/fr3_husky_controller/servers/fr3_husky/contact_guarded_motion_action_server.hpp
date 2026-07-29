#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <geometry_msgs/msg/pose.hpp>
#include <fr3_husky_msgs/action/contact_guarded_motion.hpp>

#include <fr3_husky_controller/servers/action_server_base.hpp>
#include <fr3_husky_controller/model/fr3_husky_model_updater.hpp>

namespace fr3_husky_controller::servers::fr3_husky
{

class ContactGuardedMotion final : public ActionServerBase<fr3_husky_msgs::action::ContactGuardedMotion>
{
public:
    using ActionT = fr3_husky_msgs::action::ContactGuardedMotion;
    using Base = ActionServerBase<ActionT>;
    using ComputeResult = typename Base::ComputeResult;
    using StopReason = typename Base::StopReason;
    using ResultPtr = typename Base::ResultPtr;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ContactGuardedMotion(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater);
    ~ContactGuardedMotion() override = default;

    int priority() const override { return 9; }
    bool allowPreemption() const override { return false; }

private:
    bool acceptGoal(const ActionT::Goal& goal) override;
    void onGoalAccepted(const ActionT::Goal& goal) override;
    void onStart() override;
    ComputeResult compute(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    void onStop(StopReason reason) override;
    ResultPtr makeResult(StopReason reason) override;

private:
    static Eigen::Affine3d poseMsgToAffine(const geometry_msgs::msg::Pose& msg);
    static double orientationError(const Eigen::Matrix3d& R_des, const Eigen::Matrix3d& R_cur);
    Eigen::Vector6d computeTargetVelocity(const Eigen::Affine3d& prev, const Eigen::Affine3d& cur, double dt);

private:
    FR3HuskyModelUpdater& fr3_husky_model_updater_;

    std::map<std::string, drc::TaskSpaceData> ee_data_;

    std::vector<std::string> ee_names_;
    std::vector<Eigen::Affine3d> start_poses_;
    std::vector<Eigen::Affine3d> target_poses_;

    rclcpp::Time start_time_;
    bool start_time_set_{false};

    double duration_{5.0};
    double pos_tolerance_{0.01};
    double ori_tolerance_{0.05};


    // for contact detection
    std::map<std::string, Eigen::VectorXd> start_joint_torque_;
    double last_torque_update_time_ = 0.0;
    bool contact_torque_bias_set_{false};
    bool contact_detected_{false};
    int contact_count_{0};
    int contact_debounce_count_{10};
    double contact_detection_start_time_{0.2};

    
    bool isContactDetected(const rclcpp::Time& time);

    int32_t result_error_code_{0};

};

}  // namespace fr3_husky_controller::servers::fr3_husky