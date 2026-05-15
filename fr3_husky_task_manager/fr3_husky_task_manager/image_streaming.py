import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


HOST = "0.0.0.0"
PORT = 8080
JPEG_QUALITY = 80


TOPICS = {
    "top": "/mujoco_ros_hardware/top_azure_color/color/image_raw",
    "left": "/mujoco_ros_hardware/left_d435i_color/color/image_raw",
    "right": "/mujoco_ros_hardware/right_d435i_color/color/image_raw",
}


class SharedFrame:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.frame_count = 0

    def update(self, jpeg_bytes):
        with self.lock:
            self.jpeg = jpeg_bytes
            self.frame_count += 1

    def get(self):
        with self.lock:
            return self.jpeg, self.frame_count


shared_frames = {
    name: SharedFrame()
    for name in TOPICS.keys()
}


class MultiImageSubscriber(Node):
    def __init__(self):
        super().__init__("multi_image_mjpeg_server")

        self.bridge = CvBridge()
        self.image_subscriptions = []

        for stream_name, topic_name in TOPICS.items():

            subscription = self.create_subscription(
                Image,
                topic_name,
                lambda msg, s=stream_name: self.image_callback(msg, s),
                10
            )

            self.image_subscriptions.append(subscription)

            self.get_logger().info(
                f"Subscribed: {stream_name} -> {topic_name}"
            )

    def image_callback(self, msg, stream_name):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

            ok, encoded = cv2.imencode(
                ".jpg",
                cv_image,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )

            if not ok:
                return

            shared_frames[stream_name].update(
                encoded.tobytes()
            )

        except Exception as e:
            self.get_logger().error(
                f"[{stream_name}] Image conversion error: {e}"
            )


class MJPEGHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path == "/":
            self.send_index_page()
            return

        if not self.path.endswith(".mjpg"):
            self.send_error(404)
            return

        stream_name = self.path.replace("/", "").replace(".mjpg", "")

        if stream_name not in shared_frames:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()

        last_frame_count = -1

        try:
            while True:

                jpeg, frame_count = shared_frames[stream_name].get()

                if jpeg is None:
                    time.sleep(0.005)
                    continue

                if frame_count == last_frame_count:
                    time.sleep(0.001)
                    continue

                last_frame_count = frame_count

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(
                    b"Content-Type: image/jpeg\r\n"
                )

                self.wfile.write(
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )

                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")

        except BrokenPipeError:
            pass

        except ConnectionResetError:
            pass

    def send_index_page(self):

        html = """
<!doctype html>
<html>

<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>ROS2 Multi Camera Stream</title>

<style>

html, body {
    margin: 0;
    width: 100%;
    height: 100%;
    background: black;
    overflow: hidden;
}

.container {
    width: 100vw;
    height: 100vh;

    display: grid;

    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;

    gap: 8px;

    background: black;
    padding: 8px;

    box-sizing: border-box;
}

.camera {
    width: 100%;
    height: 100%;

    background: #111;

    display: flex;
    flex-direction: column;

    overflow: hidden;

    border-radius: 12px;
}

.label {
    color: white;
    padding: 8px;
    font-family: sans-serif;
    font-size: 18px;
}

img {
    flex: 1;
    width: 100%;
    height: 100%;

    object-fit: contain;

    background: black;
}

</style>
</head>

<body>

<div class="container">

    <div class="camera">
        <div class="label">Azure Kinect</div>
        <img src="/top.mjpg">
    </div>

    <div class="camera">
        <div class="label">Left D435i</div>
        <img src="/left.mjpg">
    </div>

    <div class="camera">
        <div class="label">Right D435i</div>
        <img src="/right.mjpg">
    </div>

</div>

</body>
</html>
"""

        body = html.encode("utf-8")

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.end_headers()

        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def start_http_server():

    server = ThreadingHTTPServer(
        (HOST, PORT),
        MJPEGHandler
    )

    print(f"MJPEG server running:")
    print(f"http://{HOST}:{PORT}")

    for name in TOPICS.keys():
        print(
            f"http://localhost:{PORT}/{name}.mjpg"
        )

    server.serve_forever()


def main():

    rclpy.init()

    node = MultiImageSubscriber()

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True
    )

    http_thread.start()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()