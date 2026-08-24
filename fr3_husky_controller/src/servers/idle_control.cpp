#include <fr3_husky_controller/servers/idle_control.hpp>

namespace fr3_husky_controller::servers
{

IdleControl::IdleControl(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater)
: name_(name),
  node_(node),
  model_updater_(model_updater),
  fr3_husky_model_updater_(dynamic_cast<FR3HuskyModelUpdater*>(&model_updater))
{
    if (fr3_husky_model_updater_)
    {
        q_hold_.setZero(fr3_husky_model_updater_->manipulator_dof_);
    }
    RCLCPP_INFO(node_->get_logger(), "[%s] IdleControl created", name_.c_str());
}

bool IdleControl::compute(const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/)
{
    if (!was_idle_)
    {
        onActivated();
        was_idle_ = true;
    }

    if (!hold_initialized_)
    {
        if (fr3_husky_model_updater_ &&
            fr3_husky_model_updater_->q_total_init_.size() == static_cast<Eigen::Index>(fr3_husky_model_updater_->manipulator_dof_))
        {
            q_hold_ = fr3_husky_model_updater_->q_total_init_;
        }
        else if (fr3_husky_model_updater_ &&
                 fr3_husky_model_updater_->q_total_.size() == static_cast<Eigen::Index>(fr3_husky_model_updater_->manipulator_dof_))
        {
            q_hold_ = fr3_husky_model_updater_->q_total_;
        }
        else
        {
            model_updater_.haltCommands();
            return true;
        }
        hold_initialized_ = true;
    }

    if (fr3_husky_model_updater_)
    {
        fr3_husky_model_updater_->haltCommands();

    }
    else
    {
        model_updater_.haltCommands();
    }

    return true;
}

void IdleControl::onActivated()
{
    RCLCPP_INFO(node_->get_logger(), "[%s] entered idle", name_.c_str());
}

void IdleControl::onDeactivated()
{
    if (was_idle_)
    {
        RCLCPP_INFO(node_->get_logger(), "[%s] exit idle", name_.c_str());
        was_idle_ = false;
        hold_initialized_ = false;
    }
}

}  // namespace fr3_husky_controller::servers
