#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose
from fr3_husky_task_manager.contact_guarded_motion import ContactGuardedMotionClient
from fr3_husky_task_manager.task_space_delta_move  import TaskSpaceDeltaMoveClient
from fr3_husky_task_manager.task_space_move  import TaskSpaceMoveClient


def run_nut_tightening(
    arm="right",
    nut_position=None,
    nut_yaw=0.0,
    rotation_angle = 0.0,
    dist_offset=0.1,
    degree=True,
    pos_tolerance=0.01,
    ori_tolerance=0.05,
):
    if not rclpy.ok(): rclpy.init()

    DEFAULT_LEFT_POSE = {
        "position": [0.55, 0.25, 0.75],
        "rpy": [3.141, 0.0, 0.52],
    }
    DEFAULT_RIGHT_POSE = {
        "position": [0.55, -0.25, 0.75],
        "rpy": [3.141, 0.0, -0.52],
    }
    if degree: 
        nut_yaw = nut_yaw * math.pi / 180
        rotation_angle = rotation_angle * math.pi / 180
    if abs(rotation_angle) < math.pi / 3.0: rotation_angle = math.pi / 3.0  # set default rotation angle

    MAX_REPEAT = 10

    for idx in range(MAX_REPEAT):

        ## prepare
        left_position = DEFAULT_LEFT_POSE["position"] 
        left_rpy = DEFAULT_LEFT_POSE["rpy"] 
        right_position = DEFAULT_RIGHT_POSE["position"] 
        right_rpy = DEFAULT_RIGHT_POSE["rpy"] 
        if arm == "left":
            left_rpy[2] += nut_yaw
            left_position = nut_position
            left_position[0] -=  dist_offset * math.cos(nut_yaw)
            left_position[1] -=  dist_offset * math.sin(nut_yaw)
        elif arm == "right":
            right_rpy[2] += nut_yaw
            right_position = nut_position
            right_position[0] -=  dist_offset * math.cos(nut_yaw)
            right_position[1] -=  dist_offset * math.sin(nut_yaw)
        else: raise(f"not implemented for arm={arm}")

        node = TaskSpaceDeltaMoveClient(arm, left_position, left_rpy, right_position, right_rpy, 3.0, pos_tolerance, ori_tolerance)
        is_success, result = send_goal_and_get_result(node, "Task-space delta move", arm)
        if not is_success: return result


        ## approach
        if arm == "left": left_position = nut_position
        elif arm == "right": right_position = nut_position
        else: raise(f"not implemented for arm={arm}")
        node = ContactGuardedMotionClient(arm, left_position, left_rpy, right_position, right_rpy, 1.0, pos_tolerance, ori_tolerance)
        is_success, result = send_goal_and_get_result(node, "Contact-guarded motion", arm)
        if not is_success: return result


        ## rotate and tightening the nut
        ## TODO
        if arm == "left":
            left_rpy = DEFAULT_LEFT_POSE["rpy"]
            left_rpy[2] += rotation_angle
            left_position = nut_position
        elif arm == "right":
            right_rpy = DEFAULT_RIGHT_POSE["rpy"]
            right_rpy[2] += nut_yaw
            right_position = nut_position
        else: raise(f"not implemented for arm={arm}")
        node = TaskSpaceDeltaMoveClient(arm, left_position, left_rpy, right_position, right_rpy, 1.0, pos_tolerance, ori_tolerance)
        is_success, result = send_goal_and_get_result(node, "Task-space delta move", arm)
        if not is_success: return result


        ## backward
        if arm == "left":
            left_rpy = DEFAULT_LEFT_POSE["rpy"]
            left_rpy[2] += rotation_angle
            left_position = nut_position
            left_position[0] -=  dist_offset * math.cos(rotation_angle)
            left_position[1] -=  dist_offset * math.sin(rotation_angle)
        elif arm == "right":
            right_rpy = DEFAULT_RIGHT_POSE["rpy"]
            right_rpy[2] += nut_yaw
            right_position = nut_position
            right_position[0] -=  dist_offset * math.cos(rotation_angle)
            right_position[1] -=  dist_offset * math.sin(rotation_angle)
        else: raise(f"not implemented for arm={arm}")
        
        node = TaskSpaceDeltaMoveClient(arm, left_position, left_rpy, right_position, right_rpy, 1.0, pos_tolerance, ori_tolerance)
        is_success, result = send_goal_and_get_result(node, "Task-space delta move", arm)
        if not is_success: return result
        nut_yaw = rotation_angle % math.pi / 3.0

    result = f"Successfully executed nut tightening motion [{MAX_REPEAT} times]"
    return result




def send_goal_and_get_result(node, action_server_name, arm=None):
    try:
        node.send_goal_and_wait()
        result = f"{action_server_name} completed successfully."
        if arm is not None: result += f" [arm:{arm}]"
        is_success = True
    except KeyboardInterrupt:
        cancel_future = node.cancel_goal()
        if cancel_future is not None: rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=2.0)
        if node._result_future is not None:
            try: rclpy.spin_until_future_complete(node, node._result_future, timeout_sec=5.0)
            except KeyboardInterrupt: pass
        result = f"{action_server_name} interrupted and cancelled."
        if arm is not None: result += f" [arm:{arm}]"
        is_success = False
    except Exception as e:
        result = f"{action_server_name} failed due to an error: {e}."
        if arm is not None: result += f" [arm:{arm}]"
        is_success = False
    return is_success, result


def main(args=None):
    del args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=["left", "right"],
        default="right",
        help="Target end-effector.",
    )
    parser.add_argument(
        "--nut-position",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Target nut position",
    )
    parser.add_argument(
        "--nut-yaw",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--rotation-angle",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--degree",
        action="store_true",
    )
    parser.add_argument(
        "--pos-tolerance",
        type=float,
        default=0.01,
        help="Position tolerance [m].",
    )
    parser.add_argument(
        "--ori-tolerance",
        type=float,
        default=0.05,
        help="Orientation tolerance [rad].",
    )

    parsed_args = parser.parse_args()

    run_nut_tightening(
        arm=parsed_args.arm,
        nut_position=parsed_args.nut_position,
        nut_yaw=parsed_args.nut_yaw,
        rotation_angle=parsed_args.rotation_angle,
        degree=parsed_args.degree,
        pos_tolerance=parsed_args.pos_tolerance,
        ori_tolerance=parsed_args.ori_tolerance,
    )


if __name__ == "__main__":
    main()