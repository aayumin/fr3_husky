#!/usr/bin/env python3

import argparse
import asyncio
import os
import sys
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription


class TextFileStreamer:
    def __init__(self, file_path: str, poll_interval: float):
        self.file_path = file_path
        self.poll_interval = poll_interval
        self.channels = set()
        self.last_text = ""

    def read_text(self) -> str:
        if not os.path.exists(self.file_path):
            return ""
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    async def run(self):
        while True:
            text = self.read_text()
            if text and text != self.last_text:
                # print(f"[WebRTC Text] file updated: {text}")
                # print(f"[WebRTC Text] channels: {len(self.channels)}")
                self.last_text = text
                dead = []
                for ch in self.channels:
                    try:
                        if ch.readyState == "open":
                            # print("[WebRTC Text] sending")
                            ch.send(text)
                    except Exception:
                        dead.append(ch)
                for ch in dead:
                    self.channels.discard(ch)
            await asyncio.sleep(self.poll_interval)

    def add_channel(self, channel):
        self.channels.add(channel)

        @channel.on("open")
        def on_open():
            if self.last_text:
                channel.send(self.last_text)

        @channel.on("close")
        def on_close():
            self.channels.discard(channel)


class WebRTCTextServer:
    def __init__(self, host: str, port: int, streamer: TextFileStreamer):
        self.host = host
        self.port = port
        self.streamer = streamer
        self.pcs = set()

    async def index(self, request):
        return web.json_response({
            "status": "ok",
            "type": "webrtc_text_datachannel",
            "offer_endpoint": "/offer",
            "file": self.streamer.file_path,
        })

    async def offer(self, request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("datachannel")
        def on_datachannel(channel):
            print(f"[WebRTC Text] datachannel opened: {channel.label}")
            self.streamer.add_channel(channel)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            # print(f"[WebRTC Text] state={pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                self.pcs.discard(pc)

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

    async def run(self):
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_post("/offer", self.offer)
        app.router.add_options("/offer", self.options)
        app.on_shutdown.append(self.shutdown)

        asyncio.create_task(self.streamer.run())

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        print(f"[WebRTC Text] signaling server: http://{self.host}:{self.port}")
        print(f"[WebRTC Text] watching file: {self.streamer.file_path}")

        while True:
            await asyncio.sleep(3600)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--file", default="/root/ssds_HL/recognized_speech.txt")
    parser.add_argument("--poll-hz", type=float, default=20.0)
    args = parser.parse_args()

    print("speech_webrtc started", flush=True)

    poll_interval = 1.0 / max(args.poll_hz, 1.0)
    streamer = TextFileStreamer(args.file, poll_interval)
    server = WebRTCTextServer(args.host, args.port, streamer)

    asyncio.run(server.run())


if __name__ == "__main__":
    main()
