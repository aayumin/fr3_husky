import asyncio
import argparse
import numpy as np
import socket
import sys
import threading
import io

# ROS2 시스템 패키지 상속 연동
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from PIL import Image, ImageDraw

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

# ==========================================
# 1. ROS2 토픽 데이터를 WebRTC로 브릿지하는 노드
# ==========================================
class RosImageBridgeNode(Node):
    def __init__(self):
        super().__init__('avp_webrtc_ros_bridge')
        self.rs_latest_bytes = None
        self.image_topic_name = '/camera/camera/color/image_raw/compressed'
        
        self.sub_img = self.create_subscription(
            CompressedImage, 
            self.image_topic_name, 
            self.image_callback, 
            10
        )
        self.get_logger().info(f"[ROS2 Bridge] Subscribing to {self.image_topic_name}")

    def image_callback(self, msg):
        self.rs_latest_bytes = bytes(msg.data)


# ==========================================
# 2. 카메라 매니저 (글로벌 상태 관리)
# ==========================================
class CameraManager:
    def __init__(self, width=640, fps=30):  
        self.width = width
        self.fps = fps
        self.period = 1.0 / max(1, fps)
        
        self.ros_node = None
        self.ros_thread = None
        self.is_running = False

    def start_devices(self):
        self.is_running = True
        try:
            if not rclpy.ok():
                rclpy.init()
            self.ros_node = RosImageBridgeNode()
            
            self.ros_thread = threading.Thread(target=rclpy.spin, args=(self.ros_node,), daemon=True)
            self.ros_thread.start()
            print("[ROS2 RealSense Bridge] Thread started successfully.", flush=True)
        except Exception as e:
            print(f"[ROS2 Bridge Failure] {e}", flush=True)

    def stop_devices(self):
        self.is_running = False
        if self.ros_node:
            try:
                self.ros_node.destroy_node()
                rclpy.shutdown()
            except: pass
        print("[CameraManager] Terminated.", flush=True)


camera_manager = CameraManager()


# ==========================================
# 3. WebRTC 트랙 클래스 (💡 visionOS 호환 bgr24 정렬 매핑)
# ==========================================
class AzureKinectTrack(VideoStreamTrack):
    """Kinect 대용으로 visionOS 가 선호하는 선명한 BGR24 그리드 더미 이미지 송출"""
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        
        # 그린 스크린에 안내 텍스트 레이아웃 배치
        img = Image.new('RGB', (640, 480), color=(30, 80, 30))
        d = ImageDraw.Draw(img)
        d.text((40, 40), "Kinect Dummy Mode", fill=(255, 255, 255))
        
        # 💡 PIL(RGB)에서 OpenCV 표준 규격인 BGR 넘파이 배열로 정밀 변환
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) if 'cv2' in sys.modules else np.array(img)[..., ::-1]
        self.dummy_np = np.ascontiguousarray(img_bgr)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame.from_ndarray(self.dummy_np, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        await asyncio.sleep(self.manager.period)
        return frame

class RealSenseToAzureTrack(VideoStreamTrack):
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        img = Image.new('RGB', (640, 480), color=(20, 30, 80))
        d = ImageDraw.Draw(img)
        d.text((40, 40), "CONNECTING TO REAL_SENSE TOPIC...", fill=(255, 255, 0))
        self.waiting_np = np.ascontiguousarray(np.array(img)[..., ::-1])

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        if self.manager.ros_node is None or self.manager.ros_node.rs_latest_bytes is None:
            frame = VideoFrame.from_ndarray(self.waiting_np, format="bgr24")
            frame.pts = pts
            frame.time_base = time_base
            await asyncio.sleep(self.manager.period)
            return frame
        try:
            img_bytes = self.manager.ros_node.rs_latest_bytes
            img_pil = Image.open(io.BytesIO(img_bytes))
            # 💡 비전프로 top_azure 화각에 맞춘 회전 연산
            # img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
            # img_pil = img_pil.transpose(Image.ROTATE_270)
            frame = VideoFrame.from_ndarray(np.ascontiguousarray(np.array(img_pil)[..., ::-1]), format="bgr24")
        except Exception:
            frame = VideoFrame.from_ndarray(self.waiting_np, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        await asyncio.sleep(self.manager.period)
        return frame



class RealSenseTrack(VideoStreamTrack):
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        
        # ROS2 토픽이 공급되기 전까지 보여줄 붉은색 경고 박스 배경 생성
        img = Image.new('RGB', (640, 480), color=(120, 30, 30))
        d = ImageDraw.Draw(img)
        d.text((40, 40), "CONNECTING ROS2 TOPIC...", fill=(255, 255, 0))
        
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) if 'cv2' in sys.modules else np.array(img)[..., ::-1]
        self.waiting_np = np.ascontiguousarray(img_bgr)
        print("[RealSenseTrack] Optimized WebRTC Track initialized.", flush=True)

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        # ROS2 이미지 유실 상태라면 에러를 내지 않고 안내 대기판 전송
        if self.manager.ros_node is None or self.manager.ros_node.rs_latest_bytes is None:
            frame = VideoFrame.from_ndarray(self.waiting_np, format="bgr24")
            frame.pts = pts
            frame.time_base = time_base
            await asyncio.sleep(self.manager.period)
            return frame

        try:
            img_bytes = self.manager.ros_node.rs_latest_bytes
            img_buf = io.BytesIO(img_bytes)
            img_pil = Image.open(img_buf)
            
            # visionOS 공간 좌표 정렬을 위한 렌즈 반전 및 회전 기하 연산
            # img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
            # img_pil = img_pil.transpose(Image.ROTATE_270)
            
            # 💡 PIL RGB 데이터를 aiortc 고정 포맷인 bgr24로 완벽 캘리브레이션
            rgb_np = np.array(img_pil)
            bgr_np = rgb_np[..., ::-1] # RGB -> BGR 변환 채널 슬라이싱
            image_contiguous = np.ascontiguousarray(bgr_np) # 메모리 파편화 방지 연속화
            
            frame = VideoFrame.from_ndarray(image_contiguous, format="bgr24")
        except Exception as e:
            frame = VideoFrame.from_ndarray(self.waiting_np, format="bgr24")

        frame.pts = pts
        frame.time_base = time_base
        await asyncio.sleep(self.manager.period)
        return frame
# ==========================================
# 4. WebRTC Server 클래스
# ==========================================
class WebRTCServer:
    def __init__(self, host: str, port: int = 8080, width: int = 640):
        self.host = host
        self.port = port
        self.width = width
        self.pcs = set()

    async def index(self, request):
        html = """<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
html, body { margin: 0; padding: 0; width: 100%; height: 100%; background: black; overflow: hidden; }
video { width: 100%; height: 100%; object-fit: cover; background: black; }
pre { color: white; padding: 12px; }
</style>
</head>
<body>
<video id="video" autoplay playsinline muted></video>
<script>
const pc = new RTCPeerConnection();
const video = document.getElementById("video");

pc.ontrack = (event) => {
    if (event.streams && event.streams.length > 0) {
        video.srcObject = event.streams[0];
    } else {
        const stream = new MediaStream();
        stream.addTrack(event.track);
        video.srcObject = stream;
    }
    video.play().catch(err => console.log("video play error:", err));
};

async function start() {
    const params = new URLSearchParams(window.location.search);
    const streamName = params.get("stream") || "top_azure_color";

    pc.addTransceiver("video", { direction: "recvonly" });

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const response = await fetch("/offer?stream=" + encodeURIComponent(streamName), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(pc.localDescription)
    });

    if (!response.ok) throw new Error("offer failed: " + response.status);

    const answer = await response.json();
    await pc.setRemoteDescription(answer);
}

start().catch(err => {
    console.error(err);
    document.body.innerHTML = "<pre>" + err + "</pre>";
});
</script>
</body>
</html>"""
        return web.Response(text=html, content_type="text/html")

    async def offer(self, request):
        stream_name = request.query.get("stream", "top_azure_color")
        print(f"[WebRTC] OFFER RECEIVED stream={stream_name}", flush=True)

        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"[WebRTC] state={pc.connectionState} stream={stream_name}", flush=True)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                self.pcs.discard(pc)

        if stream_name == "top_azure_color":
            print("[WebRTC] ADD TRACK top_azure_color (Dummy Grid Mode)", flush=True)
            # pc.addTrack(AzureKinectTrack(manager=camera_manager))
            pc.addTrack(RealSenseToAzureTrack(manager=camera_manager))
            
        elif stream_name == "right_d435i_color":
            print("[WebRTC] ADD TRACK right_d435i_color", flush=True)
            pc.addTrack(RealSenseTrack(manager=camera_manager))
        else:
            return web.json_response({"error": f"unknown stream: {stream_name}"}, status=400)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        local_description = pc.localDescription if pc.localDescription is not None else answer

        return web.json_response({
            "sdp": local_description.sdp,
            "type": local_description.type,
        })

    async def run(self):
        camera_manager.width = self.width
        camera_manager.start_devices()

        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_post("/offer", self.offer)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        print(f"[WebRTC] bind address: {self.host}:{self.port}", flush=True)

        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            coros = [pc.close() for pc in self.pcs]
            await asyncio.gather(*coros)
            self.pcs.clear()
            
            camera_manager.stop_devices()
            await runner.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream ROS2 RealSense images through WebRTC to visionOS."
    )
    parser.add_argument(
        "--host",
        required=True,
        help="Server bind address. Enter 0.0.0.0 to listen on all interfaces.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP/WebRTC signaling port. Default: 8080.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Maximum output width. Default: 960.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    host = args.host
    if host != "0.0.0.0":
        try:
            socket.inet_aton(host)
        except socket.error:
            raise ValueError(f"잘못된 호스트 IP 주소 형태입니다: '{host}'.")

    if not (1024 <= args.port <= 65535):
        raise ValueError(f"포트 번호 범위 오류: {args.port}. 1024에서 65535 사이여야 합니다.")


if __name__ == "__main__":
    try:
        args = parse_args()
        validate_args(args)

        server = WebRTCServer(
            host=args.host,
            port=args.port,
            width=args.width,
        )

        asyncio.run(server.run())

    except ValueError as error:
        print(f"[Configuration Error] {error}", flush=True)
        sys.exit(2)

    except KeyboardInterrupt:
        print("\n[Server] Terminated cleanly by user (Ctrl+C).", flush=True)
        camera_manager.stop_devices()
        sys.exit(0)
