#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from fr3_husky_msgs.action import AppleVisionPro


class AppleVisionProClient(Node):
    def __init__(self):
        super().__init__('apple_vision_pro_client')

        self._action_name = '/fr3_AVP_tracker'
        self._client = ActionClient(self, AppleVisionPro, self._action_name)

        self.get_logger().info(f'Waiting for action server: {self._action_name}')
        self._client.wait_for_server()
        self.get_logger().info(f'Connected to action server: {self._action_name}')

    def send_goal(self):
        goal = AppleVisionPro.Goal()
        goal.mode = 0
        # goal.left_controller_ee_name = 'left_fr3_hand_tcp'
        # goal.right_controller_ee_name = 'right_fr3_hand_tcp'
        goal.left_controller_ee_name = 'left_fr3_link8'
        goal.right_controller_ee_name = 'right_fr3_link8'
        goal.move_orientation = False
        goal.controller_pos_multiplier = 1.0
        goal.controller_ori_multiplier = 1.0

        self.get_logger().info('Sending AppleVisionPro goal')

        future = self._client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            rclpy.shutdown()
            return

        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)
        
    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(
            f'Result - is_completed: {result.is_completed}'
        )
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = AppleVisionProClient()
    node.send_goal()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
