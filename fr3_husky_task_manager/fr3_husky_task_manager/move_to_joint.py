#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from fr3_husky_msgs.action import MoveToJoint


class MoveToJointClient(Node):
    def __init__(self):
        super().__init__('move_to_joint_client')

        self._action_name = '/fr3_husky_move_to_joint'
        self._client = ActionClient(self, MoveToJoint, self._action_name)

        self._goal_handle = None
        self._result_future = None

        self.declare_parameter('arm', 'both')
        self.declare_parameter(
            'left_target_positions',
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        )
        self.declare_parameter(
            'right_target_positions',
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        )
        self.declare_parameter('max_velocity_scaling_factor', 0.1)
        self.declare_parameter('max_acceleration_scaling_factor', 0.1)

        self.get_logger().info(f'Waiting for action server: {self._action_name}')
        self._client.wait_for_server()
        self.get_logger().info(f'Connected to action server: {self._action_name}')

    def send_goal(self):
        arm = self.get_parameter('arm').get_parameter_value().string_value
        left_target_positions = list(
            self.get_parameter('left_target_positions').get_parameter_value().double_array_value
        )
        right_target_positions = list(
            self.get_parameter('right_target_positions').get_parameter_value().double_array_value
        )
        max_vel = self.get_parameter(
            'max_velocity_scaling_factor'
        ).get_parameter_value().double_value
        max_acc = self.get_parameter(
            'max_acceleration_scaling_factor'
        ).get_parameter_value().double_value

        left_joint_names = [
            'left_fr3_joint1',
            'left_fr3_joint2',
            'left_fr3_joint3',
            'left_fr3_joint4',
            'left_fr3_joint5',
            'left_fr3_joint6',
            'left_fr3_joint7',
        ]

        right_joint_names = [
            'right_fr3_joint1',
            'right_fr3_joint2',
            'right_fr3_joint3',
            'right_fr3_joint4',
            'right_fr3_joint5',
            'right_fr3_joint6',
            'right_fr3_joint7',
        ]

        goal = MoveToJoint.Goal()

        if arm == 'left':
            self._validate_positions(left_target_positions, 'left_target_positions')
            goal.joint_names = left_joint_names
            goal.target_positions = left_target_positions

        elif arm == 'right':
            self._validate_positions(right_target_positions, 'right_target_positions')
            goal.joint_names = right_joint_names
            goal.target_positions = right_target_positions

        elif arm == 'both':
            self._validate_positions(left_target_positions, 'left_target_positions')
            self._validate_positions(right_target_positions, 'right_target_positions')
            goal.joint_names = left_joint_names + right_joint_names
            goal.target_positions = left_target_positions + right_target_positions

        else:
            self.get_logger().error("Parameter 'arm' must be one of: left, right, both")
            rclpy.shutdown()
            return

        goal.max_velocity_scaling_factor = max_vel
        goal.max_acceleration_scaling_factor = max_acc

        self.get_logger().info(f'Sending MoveToJoint goal for arm={arm}')
        self.get_logger().info(f'joint_names: {goal.joint_names}')
        self.get_logger().info(f'target_positions: {goal.target_positions}')
        self.get_logger().info(f'max_velocity_scaling_factor: {goal.max_velocity_scaling_factor}')
        self.get_logger().info(f'max_acceleration_scaling_factor: {goal.max_acceleration_scaling_factor}')

        send_goal_future = self._client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def _validate_positions(self, positions, param_name):
        if len(positions) != 7:
            self.get_logger().error(f'{param_name} must have exactly 7 elements, got {len(positions)}')
            rclpy.shutdown()

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if goal_handle is None:
            self.get_logger().error('Goal response is None')
            rclpy.shutdown()
            return

        if not goal_handle.accepted:
            self.get_logger().warn('MoveToJoint goal rejected')
            rclpy.shutdown()
            return

        self._goal_handle = goal_handle
        self.get_logger().info('MoveToJoint goal accepted')

        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self.result_callback)


    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result - is_completed: {result.is_completed}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MoveToJointClient()
    node.send_goal()
    rclpy.spin(node)


if __name__ == '__main__':
    main()