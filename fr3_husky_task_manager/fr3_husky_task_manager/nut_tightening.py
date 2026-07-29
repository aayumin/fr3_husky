#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3
from fr3_husky_task_manager.contact_guarded_motion import ContactGuardedMotionClient
from fr3_husky_task_manager.contact_guarded_delta_motion import ContactGuardedDeltaMotionClient
from fr3_husky_task_manager.task_space_delta_move  import TaskSpaceDeltaMoveClient
from fr3_husky_task_manager.task_space_move  import TaskSpaceMoveClient
from fr3_husky_task_manager.screw import ScrewMotionClient, run_screw_motion


def run_nut_tightening(
    arm="right",
    nut_position=None,
    nut_yaw=0.0,
    spanner_length = 0.1,
    rotation_angle = -60.0,
    dist_offset=0.15,
    radian=False,
    pos_tolerance=0.01,
    ori_tolerance=0.05,
    controller="fr3",
):
    if not rclpy.ok(): rclpy.init()

    DEFAULT_LEFT_POSE = {
        "position": [0.55, 0.25, 0.75],
        "rpy": [3.141, 0.0, 0.0],
    }
    DEFAULT_RIGHT_POSE = {
        "position": [0.55, -0.25, 0.75],
        "rpy": [3.141, 0.0, 0.0],
    }

    if radian: 
        nut_yaw_r = nut_yaw
        rotation_angle_r = rotation_angle
        rotation_angle_d = rotation_angle / math.pi * 180
    else:
        nut_yaw_r = nut_yaw * math.pi / 180
        rotation_angle_r = rotation_angle * math.pi / 180
        rotation_angle_d = rotation_angle

    assert abs(rotation_angle_d) >= 60.0
    MAX_REPEAT = 5


    ## prepare
    left_position = DEFAULT_LEFT_POSE["position"] 
    left_rpy = DEFAULT_LEFT_POSE["rpy"] 
    right_position = DEFAULT_RIGHT_POSE["position"] 
    right_rpy = DEFAULT_RIGHT_POSE["rpy"] 
    if arm == "left":
        left_rpy = [DEFAULT_LEFT_POSE["rpy"][0], DEFAULT_LEFT_POSE["rpy"][1], DEFAULT_LEFT_POSE["rpy"][2] + nut_yaw_r]
        left_position = [nut_position[0] - (dist_offset+spanner_length) * math.cos(nut_yaw_r), nut_position[1] - (dist_offset+spanner_length) * math.sin(nut_yaw_r), nut_position[2]]
    elif arm == "right":
        right_rpy = [DEFAULT_RIGHT_POSE["rpy"][0], DEFAULT_RIGHT_POSE["rpy"][1], DEFAULT_RIGHT_POSE["rpy"][2] + nut_yaw_r]
        right_position = [nut_position[0] - (dist_offset+spanner_length) * math.cos(nut_yaw_r), nut_position[1] - (dist_offset+spanner_length) * math.sin(nut_yaw_r), nut_position[2]]
    else: raise(f"not implemented for arm={arm}")
    node = TaskSpaceMoveClient(arm, left_position, left_rpy, right_position, right_rpy, 5.0, pos_tolerance, ori_tolerance)
    is_success, result = send_goal_and_get_result(node, "Task-space move", arm)
    if not is_success: return result


    ## approach
    if arm == "left": left_position = [nut_position[0] - spanner_length * math.cos(nut_yaw_r), nut_position[1]  - spanner_length * math.sin(nut_yaw_r), nut_position[2]]
    elif arm == "right": right_position = [nut_position[0] - spanner_length * math.cos(nut_yaw_r), nut_position[1] - spanner_length * math.sin(nut_yaw_r), nut_position[2]]
    else: raise(f"not implemented for arm={arm}")
    node = ContactGuardedMotionClient(arm, left_position, left_rpy, right_position, right_rpy, 3.0, pos_tolerance, ori_tolerance)
    is_success, result = send_goal_and_get_result(node, "Contact-guarded motion", arm)
    if not is_success: return result



    for idx in range(MAX_REPEAT):

        ## rotate and tightening the nut
        try:
            result = run_screw_motion(
                arm,
                offset=spanner_length,
                angle=rotation_angle_d,
                duration=abs(rotation_angle_d) / 10.0,
                controller=controller,
            )
        except:
            return "screw motion failed"
        finally:
            if not rclpy.ok(): rclpy.init()


        ## backward
        if arm == "left":
            left_position = [- dist_offset, 0.0, 0.0]  #  EE local frame
            left_rpy = [0.0, 0.0, 0.0]
        elif arm == "right":
            right_position = [- dist_offset, 0.0, 0.0]  #  EE local frame
            right_rpy = [0.0, 0.0, 0.0]
        else: raise(f"not implemented for arm={arm}")
        node = ContactGuardedDeltaMotionClient(arm, left_position, left_rpy, right_position, right_rpy, 3.0, pos_tolerance, ori_tolerance)
        is_success, result = send_goal_and_get_result(node, "Task-space move", arm)
        if not is_success: return result


        if idx == MAX_REPEAT -1: break


        ## prepare
        left_position = DEFAULT_LEFT_POSE["position"] 
        left_rpy = DEFAULT_LEFT_POSE["rpy"] 
        right_position = DEFAULT_RIGHT_POSE["position"] 
        right_rpy = DEFAULT_RIGHT_POSE["rpy"] 
        nut_yaw_r = (
            nut_yaw_r
            + rotation_angle_r
            - math.copysign(math.pi / 3.0, rotation_angle_r)
        )

        if arm == "left":
            left_rpy = [DEFAULT_LEFT_POSE["rpy"][0], DEFAULT_LEFT_POSE["rpy"][1], DEFAULT_LEFT_POSE["rpy"][2] + nut_yaw_r]
            left_position = [nut_position[0] - (dist_offset+spanner_length) * math.cos(nut_yaw_r), nut_position[1] - (dist_offset+spanner_length) * math.sin(nut_yaw_r), nut_position[2]]
        elif arm == "right":
            right_rpy =  [DEFAULT_RIGHT_POSE["rpy"][0], DEFAULT_RIGHT_POSE["rpy"][1], DEFAULT_RIGHT_POSE["rpy"][2] + nut_yaw_r]
            right_position = [nut_position[0] - (dist_offset+spanner_length) * math.cos(nut_yaw_r), nut_position[1] - (dist_offset+spanner_length) * math.sin(nut_yaw_r), nut_position[2]]
        else: raise(f"not implemented for arm={arm}")

        print("=========================")
        print(f"right_position: {right_position}")
        print(f"right_rpy: {right_rpy}")
        print(f"nut_position: {nut_position}")

        node = TaskSpaceMoveClient(arm, left_position, left_rpy, right_position, right_rpy, 5.0, pos_tolerance, ori_tolerance)
        is_success, result = send_goal_and_get_result(node, "Task-space move", arm)
        if not is_success: return result


        ## approach
        if arm == "left": left_position = [nut_position[0] - spanner_length * math.cos(nut_yaw_r), nut_position[1] - spanner_length * math.sin(nut_yaw_r), nut_position[2]]
        elif arm == "right": right_position = [nut_position[0] - spanner_length * math.cos(nut_yaw_r), nut_position[1] - spanner_length * math.sin(nut_yaw_r), nut_position[2]]
        else: raise(f"not implemented for arm={arm}")
        node = ContactGuardedMotionClient(arm, left_position, left_rpy, right_position, right_rpy, 3.0, pos_tolerance, ori_tolerance)
        is_success, result = send_goal_and_get_result(node, "Contact-guarded motion", arm)
        if not is_success: return result


        print("-------------------------")
        print(f"right_position: {right_position}")
        print(f"right_rpy: {right_rpy}")
        print(f"nut_position: {nut_position}")




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
        "--controller",
        choices=["fr3", "fr3_husky"],
        default="fr3",
        help="Controller action-server group used for screw motion.",
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
        "--spanner-length",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--rotation-angle",
        type=float,
        default=-60.0,
    )
    parser.add_argument(
        "--radian",
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
        spanner_length=parsed_args.spanner_length,
        rotation_angle=parsed_args.rotation_angle,
        radian=parsed_args.radian,
        pos_tolerance=parsed_args.pos_tolerance,
        ori_tolerance=parsed_args.ori_tolerance,
        controller=parsed_args.controller,
    )


if __name__ == "__main__":
    main()