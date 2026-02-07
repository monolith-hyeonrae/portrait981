"""WebRTC video streaming server.

Provides real-time video streaming to web browsers via WebRTC.

Example:
    >>> # Start WebRTC server
    >>> visualbase webrtc 0 --port 8080
    >>> # Open http://localhost:8080 in browser

Requires: aiortc, aiohttp (install with `uv sync --extra webrtc`)
"""

import asyncio
import logging
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

# Check for WebRTC dependencies
HAS_WEBRTC = False
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from aiortc.contrib.media import MediaRelay
    from av import VideoFrame
    from aiohttp import web
    HAS_WEBRTC = True
except ImportError:
    pass


def _check_webrtc():
    """Check if WebRTC dependencies are available."""
    if not HAS_WEBRTC:
        raise ImportError(
            "aiortc and aiohttp are required for WebRTC streaming. "
            "Install with: uv sync --extra webrtc"
        )


# Embedded HTML viewer
VIEWER_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>VisualBase WebRTC Viewer</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        h1 { color: #0f0; margin-bottom: 20px; }
        #video-container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        video {
            max-width: 100%;
            background: #000;
            border: 2px solid #333;
            border-radius: 8px;
        }
        #status {
            margin: 10px 0;
            padding: 10px 20px;
            background: #333;
            border-radius: 4px;
        }
        .connected { color: #0f0; }
        .connecting { color: #ff0; }
        .disconnected { color: #f00; }
        #stats {
            font-family: monospace;
            font-size: 12px;
            color: #888;
            margin-top: 10px;
        }
        button {
            background: #0f0;
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 5px;
        }
        button:hover { background: #0c0; }
        button:disabled { background: #555; color: #888; cursor: not-allowed; }
    </style>
</head>
<body>
    <h1>VisualBase WebRTC Viewer</h1>
    <div id="video-container">
        <video id="video" autoplay playsinline muted></video>
        <div id="status" class="disconnected">Disconnected</div>
        <div>
            <button id="connect" onclick="connect()">Connect</button>
            <button id="disconnect" onclick="disconnect()" disabled>Disconnect</button>
        </div>
        <div id="stats"></div>
    </div>
    <script>
        let pc = null;
        let statsInterval = null;

        function setStatus(text, className) {
            const status = document.getElementById('status');
            status.textContent = text;
            status.className = className;
        }

        async function connect() {
            setStatus('Connecting...', 'connecting');
            document.getElementById('connect').disabled = true;

            pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });

            pc.ontrack = (event) => {
                document.getElementById('video').srcObject = event.streams[0];
            };

            pc.oniceconnectionstatechange = () => {
                if (pc.iceConnectionState === 'connected') {
                    setStatus('Connected', 'connected');
                    document.getElementById('disconnect').disabled = false;
                    startStats();
                } else if (pc.iceConnectionState === 'disconnected' ||
                           pc.iceConnectionState === 'failed') {
                    setStatus('Disconnected', 'disconnected');
                    cleanup();
                }
            };

            pc.addTransceiver('video', { direction: 'recvonly' });

            try {
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);

                const response = await fetch('/offer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: pc.localDescription.sdp,
                        type: pc.localDescription.type
                    })
                });

                const answer = await response.json();
                await pc.setRemoteDescription(new RTCSessionDescription(answer));
            } catch (e) {
                console.error('Connection error:', e);
                setStatus('Connection failed: ' + e.message, 'disconnected');
                cleanup();
            }
        }

        function disconnect() {
            if (pc) pc.close();
            cleanup();
            setStatus('Disconnected', 'disconnected');
        }

        function cleanup() {
            document.getElementById('connect').disabled = false;
            document.getElementById('disconnect').disabled = true;
            document.getElementById('video').srcObject = null;
            if (statsInterval) {
                clearInterval(statsInterval);
                statsInterval = null;
            }
            document.getElementById('stats').textContent = '';
            pc = null;
        }

        function startStats() {
            statsInterval = setInterval(async () => {
                if (!pc) return;
                const stats = await pc.getStats();
                let info = [];
                stats.forEach(report => {
                    if (report.type === 'inbound-rtp' && report.kind === 'video') {
                        info.push('Frames: ' + (report.framesReceived || 0));
                        info.push('FPS: ' + (report.framesPerSecond || 0));
                        info.push('Resolution: ' + (report.frameWidth || '?') + 'x' + (report.frameHeight || '?'));
                    }
                });
                document.getElementById('stats').textContent = info.join(' | ');
            }, 1000);
        }

        window.onload = () => setTimeout(connect, 500);
    </script>
</body>
</html>
"""


# Stub classes when dependencies not available
class _StubClass:
    def __init__(self, *args, **kwargs):
        _check_webrtc()


if HAS_WEBRTC:
    class FrameVideoTrack(VideoStreamTrack):
        """WebRTC video track that serves frames from a callback."""

        kind = "video"

        def __init__(self, frame_callback: Callable[[], Optional[np.ndarray]], fps: int = 30):
            super().__init__()
            self._frame_callback = frame_callback
            self._fps = fps

        async def recv(self):
            pts, time_base = await self.next_timestamp()
            bgr_frame = self._frame_callback()
            if bgr_frame is None:
                bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb_frame = bgr_frame[:, :, ::-1].copy()
            frame = VideoFrame.from_ndarray(rgb_frame, format="rgb24")
            frame.pts = pts
            frame.time_base = time_base
            return frame

    class WebRTCServer:
        """WebRTC streaming server."""

        def __init__(
            self,
            frame_callback: Callable[[], Optional[np.ndarray]],
            host: str = "0.0.0.0",
            port: int = 8080,
            fps: int = 30,
        ):
            self._frame_callback = frame_callback
            self._host = host
            self._port = port
            self._fps = fps
            self._pcs: set = set()
            self._relay = MediaRelay()
            self._video_track = None

        async def _handle_index(self, request):
            return web.Response(text=VIEWER_HTML, content_type="text/html")

        async def _handle_offer(self, request):
            params = await request.json()
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

            pc = RTCPeerConnection()
            self._pcs.add(pc)

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"Connection state: {pc.connectionState}")
                if pc.connectionState in ("failed", "closed"):
                    await pc.close()
                    self._pcs.discard(pc)

            if self._video_track is None:
                self._video_track = FrameVideoTrack(self._frame_callback, self._fps)

            pc.addTrack(self._relay.subscribe(self._video_track))

            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return web.json_response({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            })

        async def _on_shutdown(self, app):
            coros = [pc.close() for pc in self._pcs]
            await asyncio.gather(*coros)
            self._pcs.clear()

        def run(self):
            """Run the WebRTC server (blocking)."""
            app = web.Application()
            app.router.add_get("/", self._handle_index)
            app.router.add_post("/offer", self._handle_offer)
            app.on_shutdown.append(self._on_shutdown)

            logger.info(f"WebRTC server starting on http://{self._host}:{self._port}")
            print(f"Open http://localhost:{self._port} in your browser")

            web.run_app(app, host=self._host, port=self._port, print=None)

else:
    # Stub classes when WebRTC not available
    FrameVideoTrack = _StubClass
    WebRTCServer = _StubClass
