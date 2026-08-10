#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import rclpy
from action_msgs.msg import GoalInfo
from action_msgs.srv import CancelGoal
from rclpy.action import ActionClient
from rclpy.node import Node

from fr3_husky_msgs.action import RobomimicMove


class RobomimicMoveClient(Node):
    def __init__(self, disable=False, arm=None):
        super().__init__('robomimic_move_client')

        self._action_name = '/fr3_husky_robomimic_move'
        self._cancel_service_name = f'{self._action_name}/_action/cancel_goal'
        self._client = ActionClient(self, RobomimicMove, self._action_name)
        self._cancel_client = self.create_client(CancelGoal, self._cancel_service_name)
        self._disable = disable
        self._arm = arm

    def wait_for_action_server(self) -> bool:
        self.get_logger().info(f'Waiting for action server: {self._action_name}')
        available = self._client.wait_for_server(timeout_sec=5.0)
        if available:
            self.get_logger().info(f'Connected to action server: {self._action_name}')
        else:
            self.get_logger().error(f'Action server not available: {self._action_name}')
        return available

    def send_goal_and_wait_for_accept(self) -> bool:
        goal = RobomimicMove.Goal()
        goal.mode = 0
        goal.arm = self._arm
        goal.position_scale = 1.0
        goal.rotation_scale = 1.0
        goal.command_timeout = 0.5


        self.get_logger().info('Sending RobomimicMove goal')
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done():
            self.get_logger().error('Timed out waiting for goal response')
            return False

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False

        self.get_logger().info('Goal accepted; exiting client while action keeps running')
        return True

    def cancel_all_goals(self) -> bool:
        self.get_logger().info(f'Waiting for cancel service: {self._cancel_service_name}')
        if not self._cancel_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Cancel service not available: {self._cancel_service_name}')
            return False

        request = CancelGoal.Request()
        request.goal_info = GoalInfo()

        self.get_logger().info('Requesting cancel for all RobomimicMove goals')
        future = self._cancel_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done():
            self.get_logger().error('Timed out waiting for cancel response')
            return False

        response = future.result()
        if response is None:
            self.get_logger().error('Cancel request failed')
            return False

        self.get_logger().info(
            f'Cancel response code: {response.return_code}, canceled goals: {len(response.goals_canceling)}')
        return True

    def run(self) -> bool:
        if self._disable:
            return self.cancel_all_goals()

        return self.wait_for_action_server() and self.send_goal_and_wait_for_accept()


def parse_args():
    parser = argparse.ArgumentParser(description='Enable or disable RobomimicMove Inference')
    parser.add_argument(
        '--disable',
        action='store_true',
        help='Cancel the currently running RobomimicMove action instead of starting it')
    parser.add_argument(
        '--arm',
        required=True,
        type=str,
        help='left, right or dual')
    return parser.parse_args()


def run_robomimic_move(disable=False, arm=None):
    rclpy.init()
    node = RobomimicMoveClient(disable=disable, arm=arm)

    try:
        ok = node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    raise SystemExit(0 if ok else 1)


def main(args=None):
    del args
    cli_args = parse_args()
    run_robomimic_move(disable=cli_args.disable, arm = cli_args.arm)


if __name__ == '__main__':
    main()
