#include <fr3_husky_controller/servers/fr3_husky/gripper_command_action_server.hpp>

#include <mujoco/mujoco.h>

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace mujoco_ros_hardware
{
class MujocoWorldSingleton
{
public:
    static MujocoWorldSingleton& get();
    bool isSceneLoaded() const;
    mjModel* model() const;
    mjData* data() const;
    std::mutex& dataMutex();
};
}  // namespace mujoco_ros_hardware

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

bool setBlueCylinderRightTcpWeldActive(const rclcpp::Logger& logger, bool active)
{
    auto& world = mujoco_ros_hardware::MujocoWorldSingleton::get();
    if (!world.isSceneLoaded())
    {
        RCLCPP_WARN(logger, "[GripperCommand weld] MuJoCo scene is not loaded; cannot %s weld.",
                    active ? "attach" : "detach");
        return false;
    }

    std::lock_guard<std::mutex> lock(world.dataMutex());
    mjModel* model = world.model();
    mjData* data = world.data();
    if (!model || !data)
    {
        RCLCPP_WARN(logger, "[GripperCommand weld] MuJoCo model/data unavailable.");
        return false;
    }

    constexpr const char* kWeldName = "weld_blue_right_tcp";
    constexpr const char* kParentBodyName = "right_fr3_hand_tcp";
    constexpr const char* kChildBodyName = "obj";
    constexpr const char* kChildFreeJointName = "obj_joint";

    const int weld_id = mj_name2id(model, mjOBJ_EQUALITY, kWeldName);
    const int parent_body_id = mj_name2id(model, mjOBJ_BODY, kParentBodyName);
    const int child_body_id = mj_name2id(model, mjOBJ_BODY, kChildBodyName);

    if (weld_id < 0 || parent_body_id < 0 || child_body_id < 0)
    {
        RCLCPP_WARN(logger,
                    "[GripperCommand weld] Missing weld/body. weld=%d parent(%s)=%d child(%s)=%d",
                    weld_id, kParentBodyName, parent_body_id, kChildBodyName, child_body_id);
        return false;
    }

    if (active)
    {
        mjtNum rel_pos_world[3];
        mjtNum rel_pos_parent[3];
        mju_sub3(rel_pos_world, data->xpos + 3 * child_body_id, data->xpos + 3 * parent_body_id);
        mju_mulMatTVec(rel_pos_parent, data->xmat + 9 * parent_body_id, rel_pos_world, 3, 3);

        mjtNum parent_quat_inv[4];
        mjtNum rel_quat[4];
        mju_negQuat(parent_quat_inv, data->xquat + 4 * parent_body_id);
        mju_mulQuat(rel_quat, parent_quat_inv, data->xquat + 4 * child_body_id);
        const mjtNum rel_quat_norm = std::sqrt(rel_quat[0] * rel_quat[0] +
                                               rel_quat[1] * rel_quat[1] +
                                               rel_quat[2] * rel_quat[2] +
                                               rel_quat[3] * rel_quat[3]);
        if (rel_quat_norm > mjtNum(1e-12))
        {
            for (int i = 0; i < 4; ++i)
            {
                rel_quat[i] /= rel_quat_norm;
            }
        }

        mjtNum* eq_data = model->eq_data + weld_id * mjNEQDATA;
        eq_data[3] = rel_pos_parent[0];
        eq_data[4] = rel_pos_parent[1];
        eq_data[5] = rel_pos_parent[2];
        eq_data[6] = rel_quat[0];
        eq_data[7] = rel_quat[1];
        eq_data[8] = rel_quat[2];
        eq_data[9] = rel_quat[3];

        const int free_joint_id = mj_name2id(model, mjOBJ_JOINT, kChildFreeJointName);
        if (free_joint_id >= 0)
        {
            const int dof_adr = model->jnt_dofadr[free_joint_id];
            for (int i = 0; i < 6; ++i)
            {
                data->qvel[dof_adr + i] = 0.0;
            }
        }
    }

    data->eq_active[weld_id] = active ? 1 : 0;
    mj_forward(model, data);
    RCLCPP_INFO(logger, "[GripperCommand weld] %s %s: %s <-> %s",
                active ? "Attached" : "Detached", kWeldName, kParentBodyName, kChildBodyName);
    return true;
}
}  // namespace

GripperCommand::GripperCommand(
    const std::string& name,
    const NodePtr& node,
    ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_husky_model_updater_(getFR3HuskyModelUpdater(model_updater, name))
{
    mode_ = ServerMode::CONTROLLER;
}

bool GripperCommand::acceptGoal(const ActionT::Goal& goal)
{
    if (goal.arm_names != "left" && goal.arm_names != "right" && goal.arm_names != "both")
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: invalid arm_names '%s'", name_.c_str(), goal.arm_names.c_str());
        return false;
    }
    if (goal.command != "open" && goal.command != "grasp" && goal.command != "move")
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: invalid command '%s'", name_.c_str(), goal.command.c_str());
        return false;
    }
    if (goal.width < 0.0 || goal.width > 0.08)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: width %.4f out of range [0.0, 0.08]", name_.c_str(), goal.width);
        return false;
    }
    if (goal.speed <= 0.0 || goal.speed > 0.3)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: speed %.4f out of range (0.0, 0.3]", name_.c_str(), goal.speed);
        return false;
    }
    if (goal.command == "grasp" && (goal.force <= 0.0 || goal.force > 140.0))
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: force %.2f out of range (0.0, 140.0]", name_.c_str(), goal.force);
        return false;
    }
    if (goal.epsilon_inner < 0.0 || goal.epsilon_inner > 0.08 ||
        goal.epsilon_outer < 0.0 || goal.epsilon_outer > 0.08)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: epsilon out of range [0.0, 0.08]", name_.c_str());
        return false;
    }
    if (goal.use_weld && !goal.weld_name.empty() && goal.weld_name != "weld_blue_right_tcp")
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject: unsupported weld_name '%s'", name_.c_str(), goal.weld_name.c_str());
        return false;
    }
    return true;
}

void GripperCommand::onGoalAccepted(const ActionT::Goal& goal)
{
    goal_ = goal;
    has_goal_ = true;
    command_sent_ = false;
    command_success_ = false;
    result_message_.clear();
}

GripperCommand::ComputeResult GripperCommand::compute(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/)
{
    if (!has_goal_)
    {
        return ComputeResult::RUNNING;
    }

    if (command_sent_)
    {
        has_goal_ = false;
        return command_success_ ? ComputeResult::SUCCEEDED : ComputeResult::ABORTED;
    }

    command_sent_ = true;

    const std::vector<std::string> arms =
        (goal_.arm_names == "both") ? std::vector<std::string>{"left", "right"}
                                    : std::vector<std::string>{goal_.arm_names};

    for (const auto& arm : arms)
    {
        if (!runCommandForArm(arm))
        {
            command_success_ = false;
            has_goal_ = false;
            return ComputeResult::ABORTED;
        }
    }

    command_success_ = true;
    result_message_ = "Gripper command sent: command=" + goal_.command +
                      ", arm_names=" + goal_.arm_names +
                      ", width=" + std::to_string(goal_.width);
    has_goal_ = false;
    return ComputeResult::SUCCEEDED;
}

bool GripperCommand::runCommandForArm(const std::string& arm)
{
    bool ok = false;

    if (goal_.command == "open")
    {
        if (goal_.use_weld && arm == "right")
        {
            setWeldActive(false);
        }
        ok = fr3_husky_model_updater_.GripperOpen(arm, goal_.speed);
    }
    else if (goal_.command == "grasp")
    {
        ok = fr3_husky_model_updater_.GripperGrasp(
            arm,
            goal_.width,
            goal_.speed,
            goal_.force,
            {goal_.epsilon_inner, goal_.epsilon_outer});

        if (ok && goal_.use_weld && arm == "right")
        {
            setWeldActive(true);
        }
    }
    else
    {
        ok = fr3_husky_model_updater_.GripperMove(arm, goal_.width, goal_.speed);
    }

    if (!ok)
    {
        result_message_ = "Failed to send gripper command for arm=" + arm;
        return false;
    }
    return true;
}

bool GripperCommand::setWeldActive(bool active)
{
    if (!goal_.use_weld)
    {
        return true;
    }
    if (!goal_.weld_name.empty() && goal_.weld_name != "weld_blue_right_tcp")
    {
        return false;
    }
    return setBlueCylinderRightTcpWeldActive(node_->get_logger(), active);
}

GripperCommand::ResultPtr GripperCommand::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();
    result->success = (reason == StopReason::SUCCEEDED) && command_success_;
    result->message = result_message_;
    if (result->message.empty())
    {
        result->message = result->success ? "Gripper command sent." : "Gripper command failed.";
    }
    return result;
}

REGISTER_FR3_HUSKY_ACTION_SERVER(GripperCommand, "fr3_husky_gripper_command")

}  // namespace fr3_husky_controller::servers::fr3_husky
