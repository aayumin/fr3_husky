#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

# from franka_msgs.action import Grasp, Move
from fr3_husky_msgs.action import GripperCommand


class GripperMoveClient(Node):
    def __init__(self):
        super().__init__('gripper_move_client')

    def move(
        self,
        arm_names='both',
        command='open',
        width=None,
        speed=0.1,
        force=30.0,
        epsilon_inner=0.08,
        epsilon_outer=0.08,
    ):
        if arm_names not in ['left', 'right', 'both']:
            return f'Invalid arm_names value: {arm_names}'

        if command not in ['open', 'grasp', 'move']:
            return f'Invalid command value: {command}'

        if command == 'open':
            target_width = 0.08
        elif command == 'grasp':
            target_width = 0.0
        else:
            if width is None:
                return "Invalid width value: width is required when command is 'move'."
            target_width = width

        if target_width < 0.0 or target_width > 0.08:
            return f'Invalid width value: {target_width}. Use a value from 0.0 to 0.08 meters.'

        if speed <= 0.0:
            return f'Invalid speed value: {speed}. Use a positive speed.'

        if force <= 0.0 or force > 140.0:
            return f'Invalid force value: {force}. Use a value from 0.0 to 140.0 newtons.'

        if epsilon_inner < 0.0 or epsilon_inner > 0.08:
            return f'Invalid epsilon_inner value: {epsilon_inner}. Use a value from 0.0 to 0.08 meters.'

        if epsilon_outer < 0.0 or epsilon_outer > 0.08:
            return f'Invalid epsilon_outer value: {epsilon_outer}. Use a value from 0.0 to 0.08 meters.'

        arms = ['left', 'right'] if arm_names == 'both' else [arm_names]
        
        for arm in arms:
            # action_name = f'/fr3_husky_gripper_command'
            action_name = f'/fr3_gripper_command'
            client = ActionClient(self, GripperCommand, action_name)
            goal = GripperCommand.Goal()
            goal.arm_names = arm
            goal.command = command
            goal.width = float(target_width)
            goal.speed = float(speed)
            goal.force = float(force)
            goal.epsilon_inner = float(epsilon_inner)
            goal.epsilon_outer = float(epsilon_outer)
            goal.use_weld = True
            goal.weld_name = "weld_right_tcp"


            self.get_logger().info(f'Waiting for action server: {action_name}')
            client.wait_for_server()
            self.get_logger().info(f'Connected to action server: {action_name}')

            self.get_logger().info(
                f'Sending gripper {command} goal for arm={arm}, width={target_width}, speed={speed}'
            )
            send_goal_future = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_goal_future)

            goal_handle = send_goal_future.result()
            if goal_handle is None:
                return f'Gripper move failed: {arm} gripper goal response is None.'

            if not goal_handle.accepted:
                return f'Gripper move failed: {arm} gripper goal was rejected.'

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)

            wrapped_result = result_future.result()
            if wrapped_result is None:
                return f'Gripper move failed: {arm} gripper result is None.'

            result = wrapped_result.result
            self.get_logger().info(f'Result - success: {result.success}')

            if not result.success:
                return f'Gripper move failed: {arm} gripper did not move to width {target_width}.'

        action = 'moved to'
        if arm_names == 'both':
            return f'Gripper move completed: both grippers {action} width {target_width}.'
        return f'Gripper move completed: {arm_names} gripper {action} width {target_width}.'


def run_gripper_move(
    arm_names='both',
    command='open',
    width=None,
    speed=0.1,
    force=30.0,
    epsilon_inner=0.08,
    epsilon_outer=0.08,
):
    rclpy.init()
    node = GripperMoveClient()

    try:
        result = node.move(
            arm_names=arm_names,
            command=command,
            width=width,
            speed=speed,
            force=force,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer,
        )
        node.get_logger().info(result)
        return result
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    del args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arm',
        choices=['left', 'right', 'both'],
        default='both',
        help='Target gripper arm.',
    )
    parser.add_argument(
        '--command',
        choices=['open', 'grasp', 'move'],
        default='open',
        help='Gripper command. open uses width 0.08, grasp uses width 0.0, move requires --width.',
    )
    parser.add_argument(
        '--width',
        type=float,
        default=None,
        help='Target gripper width in meters. Required only when --command move.',
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=0.1,
        help='Gripper movement speed in meters per second.',
    )
    parser.add_argument(
        '--force',
        type=float,
        default=30.0,
        help='Grasp force in newtons. Used only when width is 0.0.',
    )
    parser.add_argument(
        '--epsilon-inner',
        type=float,
        default=0.08,
        help='Inner grasp tolerance in meters. Used only when width is 0.0.',
    )
    parser.add_argument(
        '--epsilon-outer',
        type=float,
        default=0.08,
        help='Outer grasp tolerance in meters. Used only when width is 0.0.',
    )

    parsed_args = parser.parse_args()
    print(
        run_gripper_move(
            parsed_args.arm,
            parsed_args.command,
            parsed_args.width,
            parsed_args.speed,
            parsed_args.force,
            parsed_args.epsilon_inner,
            parsed_args.epsilon_outer,
        )
    )


if __name__ == '__main__':
    main()
