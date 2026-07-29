#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <fr3_husky_msgs/action/screw_motion.hpp>

#include <fr3_husky_controller/model/fr3_model_updater.hpp>
#include <fr3_husky_controller/servers/action_server_base.hpp>

namespace fr3_husky_controller::servers::fr3
{

class ScrewMotionBase : public ActionServerBase<fr3_husky_msgs::action::ScrewMotion>
{
public:
    using ActionT = fr3_husky_msgs::action::ScrewMotion;
    using Base = ActionServerBase<ActionT>;
    using ComputeResult = typename Base::ComputeResult;
    using StopReason = typename Base::StopReason;
    using ResultPtr = typename Base::ResultPtr;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ScrewMotionBase(
        const std::string& name,
        const NodePtr& node,
        ModelUpdaterBase& model_updater,
        bool force_base_z_axis);
    ~ScrewMotionBase() override = default;

    int priority() const override { return 9; }
    bool allowPreemption() const override { return false; }

private:
    bool acceptGoal(const ActionT::Goal& goal) override;
    void onGoalAccepted(const ActionT::Goal& goal) override;
    void onStart() override;
    ComputeResult compute(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    void onStop(StopReason reason) override;
    ResultPtr makeResult(StopReason reason) override;

    static Eigen::Vector3d vectorMsgToEigen(const geometry_msgs::msg::Vector3& msg);
    static double orientationError(const Eigen::Matrix3d& R_des, const Eigen::Matrix3d& R_cur);
    static geometry_msgs::msg::Pose affineToPoseMsg(const Eigen::Affine3d& T);
    static Eigen::Matrix3d rotationFromAxisAngle(const Eigen::Vector3d& axis, double angle);
    std::vector<std::string> eeNamesFromArm(const std::string& arm) const;

    void writeTaskSpaceCommand();

private:
    FR3ModelUpdater& fr3_model_updater_;
    const bool force_base_z_axis_{false};

    std::map<std::string, drc::TaskSpaceData> ee_data_;

    std::string arm_;
    std::vector<std::string> ee_names_;
    std::map<std::string, Eigen::Affine3d> start_poses_;
    std::map<std::string, Eigen::Vector3d> centers_base_;
    Eigen::Vector3d axis_base_{Eigen::Vector3d::UnitZ()};
    Eigen::Vector3d center_direction_ee_{Eigen::Vector3d::UnitX()};

    rclcpp::Time start_time_;
    bool start_time_set_{false};

    double offset_{0.1};
    double angle_{0.0};
    double pitch_{0.0};
    std::string motion_mode_{"normal"};
    double press_depth_{0.001};
    double prepress_duration_{0.5};
    double duration_{10.0};
    double pos_tolerance_{0.01};
    double ori_tolerance_{0.05};

    double last_progress_{0.0};
    std::vector<double> last_position_errors_;
    std::vector<double> last_orientation_errors_;
    int32_t result_error_code_{0};
};

class ScrewMotion final : public ScrewMotionBase
{
public:
    ScrewMotion(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater)
    : ScrewMotionBase(name, node, model_updater, false) {}
};

class ScrewMotionZ final : public ScrewMotionBase
{
public:
    ScrewMotionZ(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater)
    : ScrewMotionBase(name, node, model_updater, true) {}
};

}  // namespace fr3_husky_controller::servers::fr3
