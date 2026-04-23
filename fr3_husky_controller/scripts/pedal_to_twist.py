#!/usr/bin/env python3

# Rudder pedal teleop:
# - forward : left pedal
# - backward : right pedal
# - turning left : misalignment, forward moving is allowed
# - turning right : misalignment + right pedal, forward moving is allowed

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class RudderPedalToTwist(Node):
    def __init__(self):
        super().__init__("husky_rudder_teleop")

        self.declare_parameter("axis_left", 0)
        self.declare_parameter("axis_right", 1)
        self.declare_parameter("axis_yaw_mag", 2)
        self.declare_parameter("scale_linear", 0.5)
        self.declare_parameter("scale_angular", 0.5)
        self.declare_parameter("deadzone_pedal", 0.05)
        self.declare_parameter("deadzone_yaw_mag", 0.05)
        self.declare_parameter("deadzone_yaw_sign", 0.05)
        self.declare_parameter("enable_button", -1)
        self.declare_parameter("cmd_vel_topic", "husky_controller/cmd_vel")
        self.declare_parameter("joy_topic", "joy")

        self._axis_left = int(self.get_parameter("axis_left").value)
        self._axis_right = int(self.get_parameter("axis_right").value)
        self._axis_yaw_mag = int(self.get_parameter("axis_yaw_mag").value)
        self._scale_linear = float(self.get_parameter("scale_linear").value)
        self._scale_angular = float(self.get_parameter("scale_angular").value)
        self._deadzone_pedal = float(self.get_parameter("deadzone_pedal").value)
        self._deadzone_yaw_mag = float(self.get_parameter("deadzone_yaw_mag").value)
        self._deadzone_yaw_sign = float(self.get_parameter("deadzone_yaw_sign").value)
        self._enable_button = int(self.get_parameter("enable_button").value)
        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._pedal_topic = str(self.get_parameter("joy_topic").value)

        self._warned_axes = False
        self._warned_buttons = False

        self._pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._sub = self.create_subscription(Joy, self._pedal_topic, self._pedal_cb, 10)

    def _pedal_cb(self, msg: Joy) -> None:
        max_axis = max(self._axis_left, self._axis_right, self._axis_yaw_mag)
        if len(msg.axes) <= max_axis:
            if not self._warned_axes:
                self.get_logger().error(
                    "Pedal axes too short: need index %d, got %d",
                    max_axis,
                    len(msg.axes),
                )
                self._warned_axes = True
            return

        if self._enable_button >= 0:
            if len(msg.buttons) <= self._enable_button:
                if not self._warned_buttons:
                    self.get_logger().error(
                        "Pedal buttons too short: need index %d, got %d",
                        self._enable_button,
                        len(msg.buttons),
                    )
                    self._warned_buttons = True
                return
            if msg.buttons[self._enable_button] == 0:
                self._publish_cmd(0.0, 0.0)
                return

        raw_left = float(msg.axes[self._axis_left])
        raw_right = float(msg.axes[self._axis_right])
        raw_yaw = float(msg.axes[self._axis_yaw_mag])

        left = (raw_left + 1.0) * 0.25
        right = (raw_right + 1.0) * 0.25
        yaw_mag = (raw_yaw + 1.0) * 0.25

        if left < self._deadzone_pedal:
            left = 0.0
        if right < self._deadzone_pedal:
            right = 0.0
        if yaw_mag < self._deadzone_yaw_mag:
            yaw_mag = 0.0

        turning = yaw_mag >= self._deadzone_yaw_mag
        turning_right = turning and right >= self._deadzone_pedal

        if turning:
            # During turning, allow forward only (left pedal).
            linear = self._scale_linear * -left
        else:
            # No turn: left pedal forward, right pedal backward.
            linear = self._scale_linear * (right - left)
        linear = _clamp(linear, -1.0, 1.0)

        if turning:
            yaw_sign = -1.0 if turning_right else 1.0
            yaw_mag = _clamp(yaw_mag, 0.0, 1.0)
        else:
            yaw_sign = 0.0
            yaw_mag = 0.0

        angular = self._scale_angular * yaw_mag * yaw_sign
        angular = _clamp(angular, -1.0, 1.0)

        self._publish_cmd(linear, angular)

    def _publish_cmd(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = RudderPedalToTwist()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()