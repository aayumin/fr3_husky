#include <fr3_husky_controller/servers/fr3/move_to_joint_action_server.hpp>

#include <chrono>
#include <map>
#include <stdexcept>
#include <string>

#include <action_msgs/msg/goal_status.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <rclcpp/parameter_client.hpp>

namespace fr3_husky_controller::servers::fr3
{

// ============================================================
// Anonymous helpers
// ============================================================

namespace
{

FR3ModelUpdater& getFR3ModelUpdater(ModelUpdaterBase& model_updater,
                                    const std::string& server_name)
{
    auto* p = dynamic_cast<FR3ModelUpdater*>(&model_updater);
    if (!p)
    {
        throw std::runtime_error("[" + server_name + "] requires FR3ModelUpdater");
    }
    return *p;
}

}  // namespace

// ============================================================
// Constructor / Destructor
// ============================================================

MoveToJoint::MoveToJoint(const std::string& name, const NodePtr& node,
                         ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_model_updater_(getFR3ModelUpdater(model_updater, name))
{
    // --- Dedicated node for MoveGroupInterface ---
    // MoveGroupInterface needs its own rclcpp::Node (lifecycle nodes are not
    // accepted). We do NOT assign it to a persistent executor: MoveGroupInterface
    // calls rclcpp::spin_until_future_complete() internally, which creates a
    // temporary SingleThreadedExecutor. Pre-spinning the node in a second executor
    // causes std::terminate() inside the MoveGroupInterface constructor.
    rclcpp::NodeOptions mgi_opts;
    mgi_opts.automatically_declare_parameters_from_overrides(true);
    moveit_node_ = rclcpp::Node::make_shared(name_ + "_mgi", mgi_opts);
    mode_ = ServerMode::TASK;

    // --- Action client → fr3_joint_trajectory_controller ---
    jtc_client_ = rclcpp_action::create_client<FJT>(
        node_, "fr3_joint_trajectory_controller");


    // --- Derive planning group and valid joint prefixes from robot_names ---
    const auto& rnames = model_updater.robot_names_;
    const auto& arm_id = model_updater.arm_id_;

    if (rnames.size() >= 2)
    {
        planning_group_ = "dual_fr3_arm";
        for (const auto& rn : rnames)
            valid_joint_prefixes_.insert(rn + "_" + arm_id + "_");
    }
    else if (rnames.size() == 1)
    {
        planning_group_ = rnames[0] + "_" + arm_id + "_arm";
        valid_joint_prefixes_.insert(rnames[0] + "_" + arm_id + "_");
    }
    else
    {
        planning_group_ = arm_id + "_arm";
    }

    // --- Create persistent MoveGroupInterface ---
    {
        bool params_ok = false;
        try
        {
            auto pc = std::make_shared<rclcpp::SyncParametersClient>(moveit_node_, "move_group");
            if (pc->wait_for_service(std::chrono::seconds(5)))
            {
                const std::vector<std::string> param_names{
                    "robot_description", "robot_description_semantic"};
                for (const auto& pname : param_names)
                {
                    if (moveit_node_->has_parameter(pname)) continue;
                    auto vals = pc->get_parameters({pname});
                    if (!vals.empty() &&
                        vals[0].get_type() != rclcpp::ParameterType::PARAMETER_NOT_SET)
                        moveit_node_->declare_parameter(pname, vals[0].get_parameter_value());
                }
                params_ok = moveit_node_->has_parameter("robot_description_semantic");
            }
            else
            {
                RCLCPP_WARN(node_->get_logger(),
                            "[%s] move_group param service not available within 5 s — "
                            "MoveGroupInterface not created", name_.c_str());
            }
        }
        catch (const std::exception& e)
        {
            RCLCPP_WARN(node_->get_logger(),
                        "[%s] param forwarding failed: %s", name_.c_str(), e.what());
        }

        if (params_ok)
        {
            try
            {
                mgi_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                    moveit_node_, planning_group_,
                    std::shared_ptr<tf2_ros::Buffer>{},
                    rclcpp::Duration::from_seconds(10.0));
                RCLCPP_INFO(node_->get_logger(),
                            "[%s] MoveGroupInterface ready", name_.c_str());
            }
            catch (const std::exception& e)
            {
                RCLCPP_WARN(node_->get_logger(),
                            "[%s] MoveGroupInterface creation failed: %s",
                            name_.c_str(), e.what());
            }
        }
    }

    RCLCPP_INFO(node_->get_logger(), "[%s] MoveToJoint created — group=%s", name_.c_str(),
                planning_group_.c_str());
}

MoveToJoint::~MoveToJoint()
{
    // Signal planning thread to stop.
    // Note: if the thread is blocked inside MoveGroupInterface::plan() there
    // is no way to interrupt it; we must wait for the planning timeout to expire
    // (set via mgi.setPlanningTime() inside runPlanning()).
    cancel_flag_.store(true, std::memory_order_relaxed);
    if (planning_thread_.joinable())
    {
        planning_thread_.join();
    }
}

// ============================================================
// Helpers
// ============================================================

void MoveToJoint::writeHoldCommands()
{
    const size_t dof = model_updater_.manipulator_dof_;
    const Eigen::VectorXd qdot_zero = Eigen::VectorXd::Zero(dof);

    if (model_updater_.HasEffortCommandInterface())
    {
        if (!fr3_model_updater_.robot_controller_)
        {
            model_updater_.haltCommands();
            return;
        }
        // PD + gravity at latched positions: τ = Kp(q_hold-q) + Kv(0-qdot) + g
        const Eigen::VectorXd torque =
            fr3_model_updater_.robot_controller_->moveJointTorqueStep(
                q_hold_, qdot_zero, false);
        fr3_model_updater_.torque_desired_total_ = torque - fr3_model_updater_.g_total_;
        fr3_model_updater_.writeCommand(fr3_model_updater_.torque_desired_total_);
    }
    else if (model_updater_.HasVelocityCommandInterface())
    {
        fr3_model_updater_.qdot_desired_total_ = qdot_zero;
        fr3_model_updater_.writeCommand(fr3_model_updater_.qdot_desired_total_);
    }
    else if (model_updater_.HasPositionCommandInterface())
    {
        fr3_model_updater_.q_desired_total_ = q_hold_;
        fr3_model_updater_.writeCommand(fr3_model_updater_.q_desired_total_);
    }
}

// ============================================================
// Background planning thread
// ============================================================

void MoveToJoint::runPlanning()
{
    RCLCPP_INFO(node_->get_logger(), "[%s] planning group: %s", name_.c_str(),
                planning_group_.c_str());

    // ---- Phase 1: MoveIt2 planning ----------------------------------------
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool plan_ok = false;
    std::string err_msg;

    if (!mgi_)
    {
        RCLCPP_ERROR(node_->get_logger(),
                     "[%s] MoveGroupInterface not available. "
                     "Is move_group running? (launch fr3_moveit.launch.py, not "
                     "fr3_action_controller.launch.py alone)",
                     name_.c_str());
        std::lock_guard<std::mutex> lk(msg_mutex_);
        plan_error_msg_ = "MoveGroupInterface not available — start move_group first";
        plan_state_.store(PlanState::FAILED, std::memory_order_release);
        return;
    }

    if (!cancel_flag_.load(std::memory_order_relaxed))
    {
        try
        {
            mgi_->setPlanningTime(10.0);
            mgi_->setStartStateToCurrentState();
            mgi_->setMaxVelocityScalingFactor(goal_vel_scale_);
            mgi_->setMaxAccelerationScalingFactor(goal_acc_scale_);

            // Build target: fill ALL group joints from hardware q_total_,
            // then override with the goal's specified joints.
            // Using q_total_ directly avoids depending on move_group's /joint_states
            // subscription (which can fail in MuJoCo / fake-hardware mode).
            std::map<std::string, double> target;
            {
                const auto& rnames = model_updater_.robot_names_;
                const auto& arm_id = model_updater_.arm_id_;
                const Eigen::VectorXd& q_hw = fr3_model_updater_.q_total_;

                // q_total_ layout: [rnames[0]_joint1..7, rnames[1]_joint1..7, ...]
                size_t q_idx = 0;
                for (const auto& rn : rnames)
                {
                    for (int j = 1; j <= FR3_DOF; ++j, ++q_idx)
                    {
                        const std::string jname = rn + "_" + arm_id + "_joint" + std::to_string(j);
                        target[jname] = (q_idx < static_cast<size_t>(q_hw.size()))
                                        ? q_hw[q_idx] : 0.0;
                    }
                }

                // Override with goal joints
                for (size_t i = 0; i < goal_joint_names_.size(); ++i)
                    target[goal_joint_names_[i]] = goal_target_positions_[i];
            }

            if (!mgi_->setJointValueTarget(target))
            {
                err_msg = "setJointValueTarget failed "
                          "(joint names or values out of bounds?)";
            }
            else if (cancel_flag_.load(std::memory_order_relaxed))
            {
                err_msg = "Cancelled before planning";
            }
            else
            {
                const auto ec = mgi_->plan(plan);
                if (ec)
                {
                    plan_ok = true;
                    #if ROS_DISTRO == 22
                        RCLCPP_INFO(node_->get_logger(),
                                    "[%s] planning succeeded (%zu waypoints)",
                                    name_.c_str(),
                                    plan.trajectory_.joint_trajectory.points.size());
                    #else
                        RCLCPP_INFO(node_->get_logger(),
                                    "[%s] planning succeeded (%zu waypoints)",
                                    name_.c_str(),
                                    plan.trajectory.joint_trajectory.points.size());
                    #endif
                }
                else
                {
                    err_msg = "MoveIt2 plan failed (error_code=" +
                              std::to_string(ec.val) + ")";
                }
            }
        }
        catch (const std::exception& e)
        {
            err_msg = std::string("MoveGroupInterface exception: ") + e.what();
        }
        catch (...)
        {
            err_msg = "Unknown exception in MoveGroupInterface";
        }
    }
    else
    {
        err_msg = "Cancelled before planning";
    }

    if (!plan_ok)
    {
        RCLCPP_ERROR(node_->get_logger(), "[%s] %s", name_.c_str(), err_msg.c_str());
        {
            std::lock_guard<std::mutex> lk(msg_mutex_);
            plan_error_msg_ = err_msg;
        }
        plan_state_.store(PlanState::FAILED, std::memory_order_release);
        return;
    }

    const auto& jt = plan.trajectory_.joint_trajectory;
    if (jt.points.size() <= 1)
    {
        RCLCPP_INFO(node_->get_logger(),
                    "[%s] trajectory has %zu waypoint(s); treating as already reached",
                    name_.c_str(), jt.points.size());

        plan_state_.store(PlanState::DONE, std::memory_order_release);
        return;
    }

    if (cancel_flag_.load(std::memory_order_relaxed))
    {
        std::lock_guard<std::mutex> lk(msg_mutex_);
        plan_error_msg_ = "Cancelled after planning";
        plan_state_.store(PlanState::FAILED, std::memory_order_release);
        return;
    }

    // ---- Phase 2: Prepare handoff to fr3_joint_trajectory_controller ----
    if (!jtc_client_->wait_for_action_server(std::chrono::seconds(5)))
    {
        std::lock_guard<std::mutex> lk(msg_mutex_);
        plan_error_msg_ = "fr3_joint_trajectory_controller not available";
        RCLCPP_ERROR(node_->get_logger(), "[%s] %s", name_.c_str(),
                     plan_error_msg_.c_str());
        plan_state_.store(PlanState::FAILED, std::memory_order_release);
        return;
    }


    FJT::Goal jtc_goal;
    #if ROS_DISTRO == 22
        jtc_goal.trajectory = plan.trajectory_.joint_trajectory;
    #else
        jtc_goal.trajectory = plan.trajectory.joint_trajectory;
    #endif

    // Send trajectory to JTC and wait for the JTC action result.
    // MoveToJoint stays in EXECUTING until the JTC result callback marks
    // DONE or FAILED.
    auto send_opts = rclcpp_action::Client<FJT>::SendGoalOptions();
    send_opts.goal_response_callback =
        [this](std::shared_ptr<GoalHandleFJT> gh)
        {
            if (!gh)
            {
                RCLCPP_ERROR(node_->get_logger(),
                            "[%s] fr3_joint_trajectory_controller rejected goal",
                            name_.c_str());
                std::lock_guard<std::mutex> lk(msg_mutex_);
                plan_error_msg_ = "JTC rejected the trajectory goal";
                plan_state_.store(PlanState::FAILED, std::memory_order_release);
            }
            else
            {
                {
                    std::lock_guard<std::mutex> lk(jtc_goal_mutex_);
                    jtc_goal_handle_ = gh;
                }

                RCLCPP_INFO(node_->get_logger(),
                            "[%s] trajectory accepted by fr3_joint_trajectory_controller",
                            name_.c_str());
                plan_state_.store(PlanState::EXECUTING, std::memory_order_release);
            }
        };

    send_opts.result_callback =
        [this](const GoalHandleFJT::WrappedResult& result)
        {
            {
                std::lock_guard<std::mutex> lk(jtc_goal_mutex_);
                jtc_goal_handle_.reset();
            }

            if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
            {
                RCLCPP_INFO(node_->get_logger(),
                            "[%s] JTC execution completed successfully",
                            name_.c_str());
                plan_state_.store(PlanState::DONE, std::memory_order_release);
            }
            else
            {
                RCLCPP_ERROR(node_->get_logger(),
                            "[%s] JTC execution failed or canceled",
                            name_.c_str());
                std::lock_guard<std::mutex> lk(msg_mutex_);
                plan_error_msg_ = "JTC execution failed or canceled";
                plan_state_.store(PlanState::FAILED, std::memory_order_release);
            }
        };


    jtc_client_->async_send_goal(jtc_goal, send_opts);
}

// ============================================================
// Goal lifecycle
// ============================================================

bool MoveToJoint::acceptGoal(const ActionT::Goal& goal)
{
    if (goal.joint_names.empty())
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: joint_names is empty",
                    name_.c_str());
        return false;
    }
    if (goal.joint_names.size() != goal.target_positions.size())
    {
        RCLCPP_WARN(node_->get_logger(),
                    "[%s] Reject: joint_names.size()=%zu != target_positions.size()=%zu",
                    name_.c_str(),
                    goal.joint_names.size(), goal.target_positions.size());
        return false;
    }

    // Validate that every joint name belongs to the planning group.
    if (!valid_joint_prefixes_.empty())
    {
        for (const auto& jname : goal.joint_names)
        {
            bool ok = false;
            for (const auto& prefix : valid_joint_prefixes_)
            {
                if (jname.rfind(prefix, 0) == 0) { ok = true; break; }
            }
            if (!ok)
            {
                std::string valid;
                for (const auto& p : valid_joint_prefixes_) valid += p + "* ";
                RCLCPP_WARN(node_->get_logger(),
                            "[%s] Reject: joint '%s' not in group '%s' (valid: %s)",
                            name_.c_str(), jname.c_str(), planning_group_.c_str(),
                            valid.c_str());
                return false;
            }
        }
    }

    return true;
}

void MoveToJoint::onGoalAccepted(const ActionT::Goal& goal)
{
    // Cache goal
    goal_joint_names_     = goal.joint_names;
    goal_target_positions_ = goal.target_positions;
    goal_vel_scale_ = (goal.max_velocity_scaling_factor > 0.0)
                      ? goal.max_velocity_scaling_factor : 0.1;
    goal_acc_scale_ = (goal.max_acceleration_scaling_factor > 0.0)
                      ? goal.max_acceleration_scaling_factor : 0.1;

    // Reset per-goal state
    cancel_flag_.store(false, std::memory_order_relaxed);
    plan_state_.store(PlanState::PLANNING, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(msg_mutex_);
        plan_error_msg_.clear();
    }

    // Join previous planning thread if somehow still alive (shouldn't happen
    // in normal flow, but be safe)
    if (planning_thread_.joinable())
    {
        planning_thread_.join();
    }

    // Launch background planning (wrapped in catch-all to prevent std::terminate)
    planning_thread_ = std::thread([this] {
        try
        {
            runPlanning();
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(node_->get_logger(),
                         "[%s] planning thread threw: %s", name_.c_str(), e.what());
            std::lock_guard<std::mutex> lk(msg_mutex_);
            plan_error_msg_ = e.what();
            plan_state_.store(PlanState::FAILED, std::memory_order_release);
        }
        catch (...)
        {
            RCLCPP_ERROR(node_->get_logger(),
                         "[%s] planning thread threw unknown exception", name_.c_str());
            std::lock_guard<std::mutex> lk(msg_mutex_);
            plan_error_msg_ = "Unknown exception in planning thread";
            plan_state_.store(PlanState::FAILED, std::memory_order_release);
        }
    });

    RCLCPP_INFO(node_->get_logger(), "[%s] goal accepted — planning started",
                name_.c_str());


    requestActivate();
}

void MoveToJoint::onStart()
{
    // Latch current joint positions; writeHoldCommands() uses these every
    // control cycle while planning is in progress.
    fr3_model_updater_.setInitFromCurrent();
    q_hold_           = fr3_model_updater_.q_total_;
    result_error_code_ = ActionT::Result::ERROR_PLAN_FAILED;  // default until success
    RCLCPP_INFO(node_->get_logger(), "[%s] started, holding position during planning",
                name_.c_str());
}

// ============================================================
// Control loop
// ============================================================
MoveToJoint::ComputeResult MoveToJoint::compute(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/)
{
    const auto state = plan_state_.load(std::memory_order_acquire);

    if (state == PlanState::PLANNING)
    {
        writeHoldCommands();

        auto fb = std::make_shared<ActionT::Feedback>();
        fb->state          = 0;  // PLANNING
        fb->progress       = 0.0;
        fb->status_message = "Planning trajectory with MoveIt2 ...";
        publishFeedback(fb);

        return ComputeResult::RUNNING;
    }

    if (state == PlanState::EXECUTING)
    {
        auto fb = std::make_shared<ActionT::Feedback>();
        fb->state          = 1;  // EXECUTING
        fb->progress       = 0.5;
        fb->status_message = "Trajectory is being executed by fr3_husky_joint_trajectory_controller";
        publishFeedback(fb);

        return ComputeResult::RUNNING;
    }

    if (state == PlanState::FAILED)
    {
        std::string msg;
        {
            std::lock_guard<std::mutex> lk(msg_mutex_);
            msg = plan_error_msg_;
        }
        RCLCPP_ERROR(node_->get_logger(), "[%s] aborting: %s", name_.c_str(), msg.c_str());
        return ComputeResult::ABORTED;
    }

    auto fb = std::make_shared<ActionT::Feedback>();
    fb->state          = 2;  // DONE
    fb->progress       = 1.0;
    fb->status_message = "Trajectory execution completed";
    publishFeedback(fb);

    result_error_code_ = 0;
    return ComputeResult::SUCCEEDED;
}

// ============================================================
// Stop / Result
// ============================================================

void MoveToJoint::onStop(StopReason reason)
{
    if (reason == StopReason::CANCELED || reason == StopReason::ABORTED)
    {
        cancel_flag_.store(true, std::memory_order_relaxed);

        std::shared_ptr<GoalHandleFJT> gh;
        {
            std::lock_guard<std::mutex> lk(jtc_goal_mutex_);
            gh = jtc_goal_handle_;
        }

        if (gh)
        {
            RCLCPP_INFO(node_->get_logger(),
                        "[%s] canceling JTC goal...", name_.c_str());
            jtc_client_->async_cancel_goal(gh);
        }
    }

    if (reason != StopReason::SUCCEEDED)
    {
        model_updater_.haltCommands();
    }

    const char* rs = (reason == StopReason::CANCELED)  ? "canceled"  :
                     (reason == StopReason::SUCCEEDED) ? "succeeded" :
                     (reason == StopReason::ABORTED)   ? "aborted"   : "none";
    RCLCPP_INFO(node_->get_logger(), "[%s] stopped (%s)", name_.c_str(), rs);
}

MoveToJoint::ResultPtr MoveToJoint::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();
    if (reason == StopReason::SUCCEEDED)
    {
        result->success    = true;
        result->message    = "Planning succeeded; trajectory ready for "
                             "fr3_joint_trajectory_controller handoff";
        result->error_code = 0;
    }
    else if (reason == StopReason::CANCELED)
    {
        result->success    = false;
        result->message    = "Cancelled";
        result->error_code = 0;
    }
    else
    {
        result->success    = false;
        std::lock_guard<std::mutex> lk(msg_mutex_);
        result->message    = plan_error_msg_.empty() ? "Aborted" : plan_error_msg_;
        result->error_code = result_error_code_;
    }
    return result;
}

// Register this server into the global FR3 registry
REGISTER_FR3_ACTION_SERVER(MoveToJoint, "fr3_move_to_joint")

}  // namespace fr3_husky_controller::servers::fr3
/*
# send goal 
ros2 action send_goal /fr3_move_to_joint fr3_husky_msgs/action/MoveToJoint  \
    "{joint_names: [left_fr3_joint1, left_fr3_joint2, left_fr3_joint3, left_fr3_joint4, left_fr3_joint5, left_fr3_joint6, left_fr3_joint7],
    target_positions: [0., -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    max_velocity_scaling_factor: 0.1,
    max_acceleration_scaling_factor: 0.1}"

ros2 action send_goal /fr3_move_to_joint fr3_husky_msgs/action/MoveToJoint  \
    "{joint_names: [right_fr3_joint1, right_fr3_joint2, right_fr3_joint3, right_fr3_joint4, right_fr3_joint5, right_fr3_joint6, right_fr3_joint7],
    target_positions: [0., -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    max_velocity_scaling_factor: 0.1,
    max_acceleration_scaling_factor: 0.1}"

ros2 action send_goal /fr3_move_to_joint fr3_husky_msgs/action/MoveToJoint  \
    "{joint_names: [left_fr3_joint1, left_fr3_joint2, left_fr3_joint3, left_fr3_joint4, left_fr3_joint5, left_fr3_joint6, left_fr3_joint7, right_fr3_joint1, right_fr3_joint2, right_fr3_joint3, right_fr3_joint4, right_fr3_joint5, right_fr3_joint6, right_fr3_joint7],
    target_positions: [0., -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    max_velocity_scaling_factor: 0.1,
    max_acceleration_scaling_factor: 0.1}"
*/
