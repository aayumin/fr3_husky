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
#include <fr3_husky_msgs/action/task_space_delta_move.hpp>

#include <fr3_husky_controller/servers/action_server_base.hpp>
#include <fr3_husky_controller/model/fr3_model_updater.hpp>

namespace fr3_husky_controller::servers::fr3
{

class TaskSpaceDeltaMove final : public ActionServerBase<fr3_husky_msgs::action::TaskSpaceDeltaMove>
{
public:
    using ActionT = fr3_husky_msgs::action::TaskSpaceDeltaMove;
    using Base = ActionServerBase<ActionT>;
    using ComputeResult = typename Base::ComputeResult;
    using StopReason = typename Base::StopReason;
    using ResultPtr = typename Base::ResultPtr;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TaskSpaceDeltaMove(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater);
    ~TaskSpaceDeltaMove() override = default;

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
    FR3ModelUpdater& fr3_model_updater_;

    std::map<std::string, drc::TaskSpaceData> ee_data_;

    std::vector<std::string> ee_names_;
    std::vector<Eigen::Affine3d> start_poses_;
    std::vector<Eigen::Affine3d> target_delta_poses_;
    std::vector<Eigen::Affine3d> target_poses_;

    rclcpp::Time start_time_;
    bool start_time_set_{false};

    double duration_{5.0};
    double pos_tolerance_{0.01};
    double ori_tolerance_{0.05};

    int32_t result_error_code_{0};

};

}  // namespace fr3_husky_controller::servers::fr3