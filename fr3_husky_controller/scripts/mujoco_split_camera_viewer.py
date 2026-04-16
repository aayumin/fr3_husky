#!/usr/bin/env python3

import math
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def _image_to_cv(msg):
    if msg.encoding in ("rgb8", "bgr8"):
        channels = 3
        dtype = np.uint8
    elif msg.encoding in ("mono8", "8UC1"):
        channels = 1
        dtype = np.uint8
    elif msg.encoding in ("32FC1",):
        channels = 1
        dtype = np.float32
    elif msg.encoding in ("16UC1",):
        channels = 1
        dtype = np.uint16
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    height = int(msg.height)
    width = int(msg.width)
    expected_step = width * channels * np.dtype(dtype).itemsize
    row_step = int(msg.step)

    data = np.frombuffer(msg.data, dtype=dtype)
    if row_step == expected_step:
        if channels == 1:
            image = data.reshape((height, width))
        else:
            image = data.reshape((height, width, channels))
    else:
        row_items = row_step // np.dtype(dtype).itemsize
        if channels == 1:
            image = data.reshape((height, row_items))[:, :width]
        else:
            image = data.reshape((height, row_items // channels, channels))[:, :width, :]

    if msg.encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if msg.encoding == "32FC1":
        return _depth_to_bgr(image)
    if msg.encoding == "16UC1":
        return _depth_to_bgr(image.astype(np.float32) * 0.001)
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _depth_to_bgr(depth):
    finite = np.isfinite(depth)
    positive = finite & (depth > 0.0)
    if not np.any(positive):
        normalized = np.zeros(depth.shape, dtype=np.uint8)
    else:
        valid = depth[positive]
        near = np.percentile(valid, 2.0)
        far = np.percentile(valid, 98.0)
        if math.isclose(float(near), float(far)):
            far = near + 1.0
        clipped = np.clip(depth, near, far)
        normalized = ((clipped - near) * 255.0 / (far - near)).astype(np.uint8)
        normalized[~positive] = 0
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


class MujocoSplitCameraViewer(Node):
    def __init__(self):
        super().__init__("mujoco_split_camera_viewer")
        self.declare_parameter(
            "left_topic",
            "/mujoco_ros_hardware/right_d435i/depth/image_rect_raw",
        )
        self.declare_parameter(
            "right_topic",
            "/mujoco_ros_hardware/top_azure/depth/image_rect_raw",
        )
        self.declare_parameter("window_name", "MuJoCo cameras")
        self.declare_parameter("view_width", 640)
        self.declare_parameter("view_height", 480)

        self.left_topic = self.get_parameter("left_topic").value
        self.right_topic = self.get_parameter("right_topic").value
        self.window_name = self.get_parameter("window_name").value
        self.view_width = int(self.get_parameter("view_width").value)
        self.view_height = int(self.get_parameter("view_height").value)

        self._lock = threading.Lock()
        self._left = None
        self._right = None

        self.create_subscription(Image, self.left_topic, self._left_callback, 10)
        self.create_subscription(Image, self.right_topic, self._right_callback, 10)
        self.create_timer(1.0 / 30.0, self._draw)

        self.get_logger().info(f"Left view: {self.left_topic}")
        self.get_logger().info(f"Right view: {self.right_topic}")

    def _left_callback(self, msg):
        self._store_image("left", msg)

    def _right_callback(self, msg):
        self._store_image("right", msg)

    def _store_image(self, side, msg):
        try:
            image = _image_to_cv(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=5.0)
            return
        with self._lock:
            if side == "left":
                self._left = image
            else:
                self._right = image

    def _draw(self):
        with self._lock:
            left = None if self._left is None else self._left.copy()
            right = None if self._right is None else self._right.copy()

        left = self._prepare_panel(left, "right_d435i_d")
        right = self._prepare_panel(right, "top_azure_de")
        cv2.imshow(self.window_name, np.hstack((left, right)))
        cv2.waitKey(1)

    def _prepare_panel(self, image, label):
        if image is None:
            image = np.zeros((self.view_height, self.view_width, 3), dtype=np.uint8)
            cv2.putText(
                image,
                "waiting for image",
                (24, self.view_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
        else:
            image = cv2.resize(image, (self.view_width, self.view_height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (self.view_width, 34), (0, 0, 0), -1)
        cv2.putText(
            image,
            label,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return image


def main():
    rclpy.init()
    node = MujocoSplitCameraViewer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
