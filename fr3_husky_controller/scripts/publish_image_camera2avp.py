#!/usr/bin/env python3
import socket
from dataclasses import dataclass
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


@dataclass
class StreamConfig:
    name: str
    topic: str
    last_sent_ns: int = 0
    frame_seq: int = 0
    latest_msg: Image | None = None


def image_to_color_payload(image: Image, max_width: int, jpeg_quality: int):
    if image.width == 0 or image.height == 0 or image.step == 0 or not image.data:
        return None, None, None, None

    encoding = image.encoding.lower()
    raw = np.frombuffer(image.data, dtype=np.uint8)

    if encoding == "bgr8":
        payload = raw.reshape((image.height, image.width, 3))
        payload_encoding = "png_bgr8"
    elif encoding == "rgb8":
        payload = raw.reshape((image.height, image.width, 3))
        payload = cv2.cvtColor(payload, cv2.COLOR_RGB2BGR)
        payload_encoding = "png_bgr8"
    elif encoding == "bgra8":
        payload = raw.reshape((image.height, image.width, 4))
        payload = cv2.cvtColor(payload, cv2.COLOR_BGRA2BGR)
        payload_encoding = "png_bgr8"
    elif encoding == "rgba8":
        payload = raw.reshape((image.height, image.width, 4))
        payload = cv2.cvtColor(payload, cv2.COLOR_RGBA2BGR)
        payload_encoding = "png_bgr8"
    elif encoding in ("8uc1", "mono8"):
        payload = raw.reshape((image.height, image.width))
        payload_encoding = "png_gray8"
    else:
        return None, None, None, None

    
    # Resize before network encoding. This is critical for UDP on Wi-Fi/visionOS.
    if max_width > 0 and payload.shape[1] > max_width:
        scale = max_width / float(payload.shape[1])
        new_size = (max_width, max(1, int(payload.shape[0] * scale)))
        payload = cv2.resize(payload, new_size, interpolation=cv2.INTER_AREA)

    # JPEG dramatically reduces chunk count compared with PNG for camera images.
    quality = int(max(1, min(100, jpeg_quality)))
    ok, encoded = cv2.imencode(".jpg", payload, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None, None, None, None

    resized_height, resized_width = payload.shape[:2]
    return encoded.tobytes(), "jpeg_bgr8", resized_width, resized_height


class CameraImageToAVPPublisher(Node):
    def __init__(self):
        super().__init__("camera_image_to_avp_publisher")

        self.declare_parameter("remote_ip", "127.0.0.1")
        self.declare_parameter("remote_port", 5010)
        self.declare_parameter("max_fps", 2.0)
        self.declare_parameter("max_packet_size", 1400)
        self.declare_parameter("jpeg_quality", 55)
        self.declare_parameter("max_width", 320)

        remote_ip = self.get_parameter("remote_ip").value
        remote_port = int(self.get_parameter("remote_port").value)
        max_fps = float(self.get_parameter("max_fps").value)
        self.max_period_ns = int(1e9 / max_fps) if max_fps > 0.0 else 0
        self.max_packet_size = int(self.get_parameter("max_packet_size").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.max_width = int(self.get_parameter("max_width").value)
        self._stream_lock = threading.Lock()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.remote_addr = (remote_ip, remote_port)

        self.streams = [
            StreamConfig("left_d435i_color", "/mujoco_ros_hardware/left_d435i/color/image_raw"),
            StreamConfig("top_azure_color", "/mujoco_ros_hardware/top_azure/color/image_raw"),
            StreamConfig("right_d435i_color", "/mujoco_ros_hardware/right_d435i/color/image_raw"),
        ]

        self.subscribers = []
        for stream in self.streams:
            sub = self.create_subscription(
                Image,
                stream.topic,
                lambda msg, stream=stream: self.on_image(msg, stream),
                qos_profile_sensor_data,
            )
            self.subscribers.append(sub)

        self.send_timer = self.create_timer(0.01, self.send_ready_frames)

        self.get_logger().info(f"Sending MuJoCo color images to UDP {remote_ip}:{remote_port}, max_packet_size={self.max_packet_size}, jpeg_quality={self.jpeg_quality}, max_width={self.max_width}")
        for stream in self.streams:
            self.get_logger().info(f"  stream={stream.name} topic={stream.topic}")

    def on_image(self, msg: Image, stream: StreamConfig):
        with self._stream_lock:
            stream.latest_msg = msg

    def send_ready_frames(self):
        now_ns = self.get_clock().now().nanoseconds
        with self._stream_lock:
            snapshots = list(self.streams)

        for stream in snapshots:
            if stream.latest_msg is None:
                continue
            if self.max_period_ns > 0 and stream.last_sent_ns != 0:
                if (now_ns - stream.last_sent_ns) < self.max_period_ns:
                    continue

            msg = stream.latest_msg
            payload_bytes, payload_encoding, send_width, send_height = image_to_color_payload(
                msg, self.max_width, self.jpeg_quality)
            if payload_bytes is None:
                self.get_logger().warn(
                    f"Unsupported color image encoding on {stream.topic}: {msg.encoding}",
                    throttle_duration_sec=2.0,
                )
                continue

            stream.frame_seq += 1
            self.send_chunked_frame(
                stream_name=stream.name,
                frame_seq=stream.frame_seq,
                width=send_width,
                height=send_height,
                payload_encoding=payload_encoding,
                stamp_sec=int(msg.header.stamp.sec),
                stamp_nsec=int(msg.header.stamp.nanosec),
                payload=payload_bytes,
            )
            stream.last_sent_ns = now_ns

    def send_chunked_frame(
        self,
        stream_name: str,
        frame_seq: int,
        width: int,
        height: int,
        payload_encoding: str,
        stamp_sec: int,
        stamp_nsec: int,
        payload: bytes,
    ):
        header_prefix = (
            f"FR3IMG,{stream_name},{frame_seq},{{chunk_idx}},{{chunk_count}},{width},{height},"
            f"{payload_encoding},{stamp_sec},{stamp_nsec}\n"
        )

        # Conservative chunk sizing accounting for formatted header length.
        max_payload_per_packet = max(1024, self.max_packet_size - len(header_prefix.format(chunk_idx=9999, chunk_count=9999)))
        chunk_count = (len(payload) + max_payload_per_packet - 1) // max_payload_per_packet

        # if frame_seq % 10 == 1:
        #     self.get_logger().info(
        #         f"send to={self.remote_addr[0]}:{self.remote_addr[1]} stream={stream_name} frame={frame_seq} "
        #         f"bytes={len(payload)} chunks={chunk_count} packet_payload={max_payload_per_packet}"
        #     )

        for chunk_idx in range(chunk_count):
            start = chunk_idx * max_payload_per_packet
            end = min(len(payload), start + max_payload_per_packet)
            header = header_prefix.format(chunk_idx=chunk_idx, chunk_count=chunk_count).encode("utf-8")
            packet = header + payload[start:end]
            # self.get_logger().info(
            #     f"------------ header={header[:100]}"
            # )
            try:
                self.sock.sendto(packet, self.remote_addr)
            except OSError as exc:
                self.get_logger().warn(
                    f"Failed to send {stream_name} frame {frame_seq} chunk {chunk_idx}/{chunk_count}: {exc}",
                    throttle_duration_sec=1.0,
                )
                return


def main(args=None):
    rclpy.init(args=args)
    node = CameraImageToAVPPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
