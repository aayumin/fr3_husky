#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Vector3
from fr3_husky_msgs.action import ScrewMotion


class ScrewMotionClient(Node):
    DEFAULT_AXIS_BASE = [0.0, 0.0, 1.0]
    DEFAULT_CENTER_DIRECTION_EE = [1.0, 0.0, 0.0]

    def __init__(
        self,
        arm="left",
        use_z_axis=True,
        axis_base=None,
        center_direction_ee=None,
        offset=0.1,
        angle=90.0,
        angle_in_degrees=True,
        duration=10.0,
        pos_tolerance=0.01,
        ori_tolerance=0.05,
    ):
        super().__init__("screw_motion_client")

        self._action_name = "/fr3_husky_screw_z" if use_z_axis else "/fr3_husky_screw"
        self._client = ActionClient(self, ScrewMotion, self._action_name)

        self._goal_handle = None
        self._result_future = None

        self.declare_parameter("arm", arm)
        self.declare_parameter("axis_base", axis_base if axis_base is not None else self.DEFAULT_AXIS_BASE)
        self.declare_parameter(
            "center_direction_ee",
            center_direction_ee if center_direction_ee is not None else self.DEFAULT_CENTER_DIRECTION_EE,
        )
        self.declare_parameter("offset", offset)
        self.declare_parameter("angle", angle)
        self.declare_parameter("angle_in_degrees", angle_in_degrees)
        self.declare_parameter("duration", duration)
        self.declare_parameter("pos_tolerance", pos_tolerance)
        self.declare_parameter("ori_tolerance", ori_tolerance)

        self.get_logger().info(f"Waiting for action server: {self._action_name}")
        self._client.wait_for_server()
        self.get_logger().info(f"Connected to action server: {self._action_name}")

    @staticmethod
    def make_vector(values):
        msg = Vector3()
        msg.x = float(values[0])
        msg.y = float(values[1])
        msg.z = float(values[2])
        return msg

    @staticmethod
    def _validate_vector(values, param_name, size=3):
        if len(values) != size:
            raise ValueError(f"{param_name} must have exactly {size} elements, got {len(values)}")

    def send_goal_and_wait(self):
        arm = self.get_parameter("arm").get_parameter_value().string_value
        axis_base = list(self.get_parameter("axis_base").get_parameter_value().double_array_value)
        center_direction_ee = list(
            self.get_parameter("center_direction_ee").get_parameter_value().double_array_value
        )
        offset = self.get_parameter("offset").get_parameter_value().double_value
        angle = self.get_parameter("angle").get_parameter_value().double_value
        angle_in_degrees = self.get_parameter("angle_in_degrees").get_parameter_value().bool_value
        duration = self.get_parameter("duration").get_parameter_value().double_value
        pos_tolerance = self.get_parameter("pos_tolerance").get_parameter_value().double_value
        ori_tolerance = self.get_parameter("ori_tolerance").get_parameter_value().double_value

        if arm not in ("left", "right", "both"):
            raise ValueError("arm must be one of: left, right, both")
        self._validate_vector(axis_base, "axis_base")
        self._validate_vector(center_direction_ee, "center_direction_ee")

        goal = ScrewMotion.Goal()
        goal.arm = arm
        goal.axis_base = self.make_vector(axis_base)
        goal.center_direction_ee = self.make_vector(center_direction_ee)
        goal.offset = offset
        goal.angle = angle
        goal.angle_in_degrees = angle_in_degrees
        goal.duration = duration
        goal.pos_tolerance = pos_tolerance
        goal.ori_tolerance = ori_tolerance

        unit = "deg" if angle_in_degrees else "rad"
        self.get_logger().info(
            f"Sending ScrewMotion goal to {self._action_name}: "
            f"arm={arm}, center_direction_ee={center_direction_ee}, "
            f"offset={offset:.4f}, angle={angle:.4f} {unit}, duration={duration:.3f}"
        )

        send_goal_future = self._client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback,
        )
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if goal_handle is None:
            raise RuntimeError("ScrewMotion goal response is None")
        if not goal_handle.accepted:
            raise RuntimeError("ScrewMotion goal rejected")

        self._goal_handle = goal_handle
        self.get_logger().info("ScrewMotion goal accepted")

        self._result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, self._result_future)

        wrapped_result = self._result_future.result()
        if wrapped_result is None:
            raise RuntimeError("ScrewMotion result is None")

        result = wrapped_result.result
        self.get_logger().info(f"Result - success: {result.success}")
        self.get_logger().info(f"Result - message: {result.message}")
        if not result.success:
            raise RuntimeError(result.message)
        return result

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        max_pos_error = max(feedback.position_errors) if feedback.position_errors else 0.0
        max_ori_error = max(feedback.orientation_errors) if feedback.orientation_errors else 0.0
        self.get_logger().info(
            f"Feedback - progress: {feedback.progress:.2f}, "
            f"pos_error: {max_pos_error:.4f}, ori_error: {max_ori_error:.4f}, "
            f"status: {feedback.status_message}"
        )

    def cancel_goal(self):
        if self._goal_handle is None:
            return None
        return self._goal_handle.cancel_goal_async()


def run_screw_motion(
    arm="left",
    use_z_axis=True,
    axis_base=None,
    center_direction_ee=None,
    offset=0.1,
    angle=90.0,
    angle_in_degrees=True,
    duration=10.0,
    pos_tolerance=0.01,
    ori_tolerance=0.05,
):
    if not rclpy.ok():
        rclpy.init()

    node = ScrewMotionClient(
        arm=arm,
        use_z_axis=use_z_axis,
        axis_base=axis_base,
        center_direction_ee=center_direction_ee,
        offset=offset,
        angle=angle,
        angle_in_degrees=angle_in_degrees,
        duration=duration,
        pos_tolerance=pos_tolerance,
        ori_tolerance=ori_tolerance,
    )

    try:
        node.send_goal_and_wait()
        result = f"Screw motion completed successfully. [arm:{arm}]"
    except KeyboardInterrupt:
        cancel_future = node.cancel_goal()
        if cancel_future is not None:
            rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=2.0)
        if node._result_future is not None:
            try:
                rclpy.spin_until_future_complete(node, node._result_future, timeout_sec=5.0)
            except KeyboardInterrupt:
                pass
        result = f"Screw motion interrupted and cancelled. [arm:{arm}]"
    except Exception as exc:
        result = f"Screw motion failed: {exc}. [arm:{arm}]"
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return result


def main(args=None):
    del args

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["left", "right", "both"], default="left")
    parser.add_argument(
        "--server",
        choices=["screw_z", "screw"],
        default="screw_z",
        help="Action server to use. screw_z fixes axis_base to base Z.",
    )
    parser.add_argument(
        "--axis-base",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Rotation axis in base frame. Ignored by screw_z.",
    )
    parser.add_argument(
        "--center-direction-ee",
        type=float,
        nargs=3,
        default=[1.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Direction from current EE origin to rotation center, expressed in current EE frame.",
    )
    parser.add_argument("--offset", type=float, default=0.1, help="Rotation radius [m].")
    parser.add_argument("--angle", type=float, default=90.0, help="Rotation angle. Degrees by default.")
    parser.add_argument("--degree", dest="angle_in_degrees", action="store_true", default=True)
    parser.add_argument("--radian", dest="angle_in_degrees", action="store_false")
    parser.add_argument("--duration", type=float, default=10.0, help="Motion duration [s].")
    parser.add_argument("--pos-tolerance", type=float, default=0.01, help="Position tolerance [m].")
    parser.add_argument("--ori-tolerance", type=float, default=0.05, help="Orientation tolerance [rad].")

    parsed_args = parser.parse_args()

    result = run_screw_motion(
        arm=parsed_args.arm,
        use_z_axis=(parsed_args.server == "screw_z"),
        axis_base=parsed_args.axis_base,
        center_direction_ee=parsed_args.center_direction_ee,
        offset=parsed_args.offset,
        angle=parsed_args.angle,
        angle_in_degrees=parsed_args.angle_in_degrees,
        duration=parsed_args.duration,
        pos_tolerance=parsed_args.pos_tolerance,
        ori_tolerance=parsed_args.ori_tolerance,
    )
    print(result)


if __name__ == "__main__":
    main()
