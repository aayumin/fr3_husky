#!/usr/bin/env python3

import asyncio
import threading
import time

import cv2
import numpy as np
import rclpy
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc import VideoStreamTrack
from av import VideoFrame
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class SharedImage:
    def __init__(self):
        self.lock = threading.Lock()
        self.image = None
        self.seq = 0

    def update(self, image):
        with self.lock:
            self.image = image
            self.seq += 1

    def get(self):
        with self.lock:
            if self.image is None:
                return None, self.seq
            return self.image.copy(), self.seq


class StreamConfig:
    def __init__(self, name, topic):
        self.name = name
        self.topic = topic
        self.shared = SharedImage()


def image_msg_to_bgr(msg: Image):
    enc = msg.encoding.lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "bgr8":
        return raw.reshape((msg.height, msg.step))[:, :msg.width * 3].reshape((msg.height, msg.width, 3)).copy()

    if enc == "rgb8":
        img = raw.reshape((msg.height, msg.step))[:, :msg.width * 3].reshape((msg.height, msg.width, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if enc == "bgra8":
        img = raw.reshape((msg.height, msg.step))[:, :msg.width * 4].reshape((msg.height, msg.width, 4))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if enc == "rgba8":
        img = raw.reshape((msg.height, msg.step))[:, :msg.width * 4].reshape((msg.height, msg.width, 4))
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    if enc in ("mono8", "8uc1"):
        img = raw.reshape((msg.height, msg.step))[:, :msg.width]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported encoding: {msg.encoding}")


class RosImageReceiver(Node):
    def __init__(self, streams, max_width):
        super().__init__("camera_webrtc_sender")
        self.streams = streams
        self.max_width = max_width
        self.subscribers = []

        for stream in self.streams:
            sub = self.create_subscription(
                Image,
                stream.topic,
                lambda msg, s=stream: self.on_image(msg, s),
                qos_profile_sensor_data,
            )
            self.subscribers.append(sub)
            self.get_logger().info(f"Subscribed: {stream.name} -> {stream.topic}")

    def on_image(self, msg, stream):
        try:
            img = image_msg_to_bgr(msg)

            if self.max_width > 0 and img.shape[1] > self.max_width:
                scale = self.max_width / float(img.shape[1])
                img = cv2.resize(img, (self.max_width, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)

            stream.shared.update(img)

        except Exception as e:
            self.get_logger().warn(f"[{stream.name}] image error: {e}", throttle_duration_sec=2.0)


class CameraTrack(VideoStreamTrack):
    def __init__(self, stream: StreamConfig, fps: float):
        super().__init__()
        self.stream = stream
        self.period = 1.0 / max(1.0, fps)
        self.last_seq = -1

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        image = None
        while image is None:
            image, seq = self.stream.shared.get()
            if image is None or seq == self.last_seq:
                await asyncio.sleep(0.005)
                image = None
                continue
            self.last_seq = seq

        frame = VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base

        await asyncio.sleep(self.period)
        return frame


class WebRTCServer:
    def __init__(self, streams, host, port, fps):
        self.streams = streams
        self.host = host
        self.port = port
        self.fps = fps
        self.pcs = set()

    async def index(self, request):
        names = [s.name for s in self.streams]
        return web.json_response({
            "status": "ok",
            "streams": names,
            "offer_endpoint": "/offer",
        })

    async def offer(self, request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"[WebRTC] state={pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                self.pcs.discard(pc)

        for stream in self.streams:
            pc.addTrack(CameraTrack(stream, self.fps))

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        local_description = pc.localDescription if pc.localDescription is not None else answer

        response = web.json_response({
            "sdp": local_description.sdp,
            "type": local_description.type,
        })
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    async def options(self, request):
        response = web.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    async def shutdown(self, app):
        await asyncio.gather(*[pc.close() for pc in self.pcs])
        self.pcs.clear()

    def make_app(self):
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_post("/offer", self.offer)
        app.router.add_options("/offer", self.options)
        app.on_shutdown.append(self.shutdown)
        return app

    async def run(self):
        app = self.make_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        print(f"[WebRTC] signaling server: http://{self.host}:{self.port}")
        print("[WebRTC] POST offer to: /offer")

        while True:
            await asyncio.sleep(3600)


def spin_ros(node):
    rclpy.spin(node)


async def main_async():
    rclpy.init()

    tmp_node = Node("camera_webrtc_param_loader")
    tmp_node.declare_parameter("host", "0.0.0.0")
    tmp_node.declare_parameter("port", 8080)
    tmp_node.declare_parameter("max_fps", 12.0)
    tmp_node.declare_parameter("max_width", 480)

    host = str(tmp_node.get_parameter("host").value)
    port = int(tmp_node.get_parameter("port").value)
    max_fps = float(tmp_node.get_parameter("max_fps").value)
    max_width = int(tmp_node.get_parameter("max_width").value)
    tmp_node.destroy_node()

    streams = [
        StreamConfig("left_d435i_color", "/mujoco_ros_hardware/left_d435i/color/image_raw"),
        StreamConfig("top_azure_color", "/mujoco_ros_hardware/top_azure/color/image_raw"),
        StreamConfig("right_d435i_color", "/mujoco_ros_hardware/right_d435i/color/image_raw"),
    ]

    ros_node = RosImageReceiver(streams, max_width)
    ros_thread = threading.Thread(target=spin_ros, args=(ros_node,), daemon=True)
    ros_thread.start()

    server = WebRTCServer(streams, host, port, max_fps)

    try:
        await server.run()
    finally:
        ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()