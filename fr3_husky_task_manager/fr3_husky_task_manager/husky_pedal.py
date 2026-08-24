#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from fr3_husky_msgs.action import HuskyPedal


class HuskyPedalClient(Node):
    def __init__(self, enable=True):
        super().__init__('husky_pedal_client')

        self._action_name = '/fr3_husky_pedal'
        self._client = ActionClient(self, HuskyPedal, self._action_name)

        self._goal_handle = None
        self._result_future = None

        self.declare_parameter('enable', enable)

        self.get_logger().info(f'Waiting for action server: {self._action_name}')
        self._client.wait_for_server()
        self.get_logger().info(f'Connected to action server: {self._action_name}')

    def send_goal_and_wait(self):
        enable = self.get_parameter('enable').get_parameter_value().bool_value

        goal = HuskyPedal.Goal()
        goal.enable = enable

        self.get_logger().info(f'Sending HuskyPedal goal: enable={enable}')

        send_goal_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if goal_handle is None:
            self.get_logger().error('Goal response is None')
            return None

        if not goal_handle.accepted:
            self.get_logger().warn('HuskyPedal goal rejected')
            return None

        self._goal_handle = goal_handle
        self.get_logger().info('HuskyPedal goal accepted')

        self._result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, self._result_future)

        wrapped_result = self._result_future.result()
        if wrapped_result is None:
            self.get_logger().error('Result is None')
            return None

        result = wrapped_result.result
        self.get_logger().info(f'Result - is_completed: {result.is_completed}')
        return result

    def cancel_goal(self):
        if self._goal_handle is None:
            return None
        return self._goal_handle.cancel_goal_async()


def run_husky_pedal(enable=True):
    rclpy.init()
    node = HuskyPedalClient(enable=enable)

    try:
        result = node.send_goal_and_wait()
        return result is not None and result.is_completed
    except KeyboardInterrupt:
        cancel_future = node.cancel_goal()
        if cancel_future is not None:
            rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=2.0)
        return False
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    del args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--disable',
        action='store_true',
        help='Send a disable goal instead of enable.',
    )

    parsed_args = parser.parse_args()
    run_husky_pedal(enable=not parsed_args.disable)


if __name__ == '__main__':
    main()
