#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose
from fr3_husky_msgs.action import TaskSpaceDeltaMove


class TaskSpaceDeltaMoveClient(Node):
    DEFAULT_LEFT_DELTA_POSE = {
        "position": [0.05, 0.0, 0.0],
        "rpy": [0.0, 0.0, 0.52],
    }

    

    DEFAULT_RIGHT_DELTA_POSE = {
        "position": [0.05, 0.0, 0.0],
        "rpy": [0.0, 0.0, -0.52],
    }

    def __init__(
        self,
        arm="both",
        left_position=None,
        left_rpy=None,
        right_position=None,
        right_rpy=None,
        duration=3.0,
        pos_tolerance=0.01,
        ori_tolerance=0.05,
    ):
        super().__init__("task_space_delta_move_client")

        self._action_name = "/fr3_husky_task_space_delta_move"
        self._client = ActionClient(self, TaskSpaceDeltaMove, self._action_name)

        self._goal_handle = None
        self._result_future = None

        self.declare_parameter("arm", arm)
        self.declare_parameter("duration", duration)
        self.declare_parameter("pos_tolerance", pos_tolerance)
        self.declare_parameter("ori_tolerance", ori_tolerance)

        self.declare_parameter(
            "left_position",
            left_position if left_position is not None else self.DEFAULT_LEFT_DELTA_POSE["position"],
        )
        self.declare_parameter(
            "left_rpy",
            left_rpy if left_rpy is not None else self.DEFAULT_LEFT_DELTA_POSE["rpy"],
        )
        self.declare_parameter(
            "right_position",
            right_position if right_position is not None else self.DEFAULT_RIGHT_DELTA_POSE["position"],
        )
        self.declare_parameter(
            "right_rpy",
            right_rpy if right_rpy is not None else self.DEFAULT_RIGHT_DELTA_POSE["rpy"],
        )

        self.get_logger().info(f"Waiting for action server: {self._action_name}")
        self._client.wait_for_server()
        self.get_logger().info(f"Connected to action server: {self._action_name}")

    @staticmethod
    def rpy_to_quaternion(roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
        return q

    @classmethod
    def make_pose(cls, position, rpy):
        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])

        qx, qy, qz, qw = cls.rpy_to_quaternion(
            float(rpy[0]),
            float(rpy[1]),
            float(rpy[2]),
        )
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose

    def send_goal_and_wait(self):
        arm = self.get_parameter("arm").get_parameter_value().string_value

        left_position = list(
            self.get_parameter("left_position").get_parameter_value().double_array_value
        )
        left_rpy = list(
            self.get_parameter("left_rpy").get_parameter_value().double_array_value
        )
        right_position = list(
            self.get_parameter("right_position").get_parameter_value().double_array_value
        )
        right_rpy = list(
            self.get_parameter("right_rpy").get_parameter_value().double_array_value
        )

        duration = self.get_parameter("duration").get_parameter_value().double_value
        pos_tolerance = self.get_parameter("pos_tolerance").get_parameter_value().double_value
        ori_tolerance = self.get_parameter("ori_tolerance").get_parameter_value().double_value

        self._validate_vector(left_position, "left_position", 3)
        self._validate_vector(left_rpy, "left_rpy", 3)
        self._validate_vector(right_position, "right_position", 3)
        self._validate_vector(right_rpy, "right_rpy", 3)

        goal = TaskSpaceDeltaMove.Goal()

        if arm == "left":
            goal.ee_names = ["left_fr3_hand_tcp"]
            goal.target_delta_poses = [self.make_pose(left_position, left_rpy)]
        elif arm == "right":
            goal.ee_names = ["right_fr3_hand_tcp"]
            goal.target_delta_poses = [self.make_pose(right_position, right_rpy)]
        elif arm == "both":
            goal.ee_names = ["left_fr3_hand_tcp", "right_fr3_hand_tcp"]
            goal.target_delta_poses = [
                self.make_pose(left_position, left_rpy),
                self.make_pose(right_position, right_rpy),
            ]
        else:
            self.get_logger().error("Parameter 'arm' must be one of: left, right, both")
            return

        goal.duration = duration
        goal.pos_tolerance = pos_tolerance
        goal.ori_tolerance = ori_tolerance

        self.get_logger().info(f"Sending TaskSpaceDeltaMove goal for arm={arm}")

        send_goal_future = self._client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback,
        )
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if goal_handle is None:
            self.get_logger().error("Goal response is None")
            return

        if not goal_handle.accepted:
            self.get_logger().warn("TaskSpaceDeltaMove goal rejected")
            return

        self._goal_handle = goal_handle
        self.get_logger().info("TaskSpaceDeltaMove goal accepted")

        self._result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, self._result_future)

        wrapped_result = self._result_future.result()
        if wrapped_result is None:
            self.get_logger().error("Result is None")
            return

        result = wrapped_result.result
        self.get_logger().info(f"Result - success: {result.success}")
        self.get_logger().info(f"Result - message: {result.message}")

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f"Feedback - progress: {feedback.progress:.2f}, "
            f"status: {feedback.status_message}"
        )

    def _validate_vector(self, values, param_name, size):
        if len(values) != size:
            self.get_logger().error(
                f"{param_name} must have exactly {size} elements, got {len(values)}"
            )
            rclpy.shutdown()

    def cancel_goal(self):
        if self._goal_handle is None:
            return None
        return self._goal_handle.cancel_goal_async()


def run_task_space_delta_move(
    arm="both",
    left_position=None,
    left_rpy=None,
    right_position=None,
    right_rpy=None,
    duration=3.0,
    pos_tolerance=0.01,
    ori_tolerance=0.05,
):
    if not rclpy.ok():
        rclpy.init()

    node = TaskSpaceDeltaMoveClient(
        arm=arm,
        left_position=left_position,
        left_rpy=left_rpy,
        right_position=right_position,
        right_rpy=right_rpy,
        duration=duration,
        pos_tolerance=pos_tolerance,
        ori_tolerance=ori_tolerance,
    )

    try:
        node.send_goal_and_wait()
        result = f"Task-space delta move completed successfully. [arm:{arm}]"

    except KeyboardInterrupt:
        cancel_future = node.cancel_goal()

        if cancel_future is not None:
            rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=2.0)

        if node._result_future is not None:
            try:
                rclpy.spin_until_future_complete(node, node._result_future, timeout_sec=5.0)
            except KeyboardInterrupt:
                pass

        result = f"Task-space delta move interrupted and cancelled. [arm:{arm}]"

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return result


def main(args=None):
    del args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=["left", "right", "both"],
        default="both",
        help="Target end-effector.",
    )
    parser.add_argument(
        "--left-position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Target delta position for the left end-effector.",
    )
    parser.add_argument(
        "--left-rpy",
        type=float,
        nargs=3,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Target delta RPY orientation for the left end-effector [rad].",
    )
    parser.add_argument(
        "--right-position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Target delta position for the right end-effector.",
    )
    parser.add_argument(
        "--right-rpy",
        type=float,
        nargs=3,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Target delta RPY orientation for the right end-effector [rad].",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Task-space motion duration [s].",
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

    run_task_space_delta_move(
        arm=parsed_args.arm,
        left_position=parsed_args.left_position,
        left_rpy=parsed_args.left_rpy,
        right_position=parsed_args.right_position,
        right_rpy=parsed_args.right_rpy,
        duration=parsed_args.duration,
        pos_tolerance=parsed_args.pos_tolerance,
        ori_tolerance=parsed_args.ori_tolerance,
    )


if __name__ == "__main__":
    main()