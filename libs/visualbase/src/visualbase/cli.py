"""Command-line interface for visualbase."""

import os
import sys

# Suppress Qt/OpenCV warnings before any imports
def _suppress_qt_warnings():
    """Suppress Qt and OpenCV warnings for cleaner CLI output."""
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


class _StderrFilter:
    """Filter stderr to suppress Qt/OpenCV warnings."""

    SUPPRESS_PATTERNS = (
        "QFontDatabase:",
        "Note that Qt no longer ships fonts",
        "XDG_SESSION_TYPE=wayland",
        "qt.qpa.",
    )

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        if not any(p in text for p in self.SUPPRESS_PATTERNS):
            self._stream.write(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


_suppress_qt_warnings()
sys.stderr = _StderrFilter(sys.stderr)

import argparse


def _detect_source_type(source: str) -> str:
    """Detect source type from path/URL.

    Returns:
        "rtsp", "camera", or "file"
    """
    if source.startswith("rtsp://") or source.startswith("rtsps://"):
        return "rtsp"
    if source.startswith("/dev/video") or source.isdigit():
        return "camera"
    return "file"


def main():
    parser = argparse.ArgumentParser(
        prog="visualbase",
        description="VisualBase - Video streaming and clip extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  visualbase play video.mp4              # Play video file
  visualbase play /dev/video0            # Preview USB camera
  visualbase play 0                      # Preview camera (shorthand)
  visualbase play rtsp://host/stream     # Preview RTSP stream
  visualbase info video.mp4              # Show video info
  visualbase clip video.mp4 -t 30        # Extract clip at 30s
""",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # play command - unified for file/camera/rtsp
    play_parser = subparsers.add_parser(
        "play",
        help="Play video from file, camera, or RTSP stream",
        description="Auto-detects source type: rtsp:// → RTSP, /dev/video* or number → camera, else → file",
    )
    play_parser.add_argument(
        "source",
        help="Video source: file path, camera device (/dev/video0 or 0), or RTSP URL",
    )
    play_parser.add_argument(
        "--fps", type=float, default=0,
        help="Target FPS (0 = original/default)",
    )
    play_parser.add_argument(
        "--resolution", "-r", type=str, default=None,
        help="Resolution as WIDTHxHEIGHT (e.g., 640x480)",
    )
    play_parser.add_argument(
        "--buffer", "-b", type=float, default=60.0,
        help="Ring buffer retention in seconds (default: 60)",
    )
    play_parser.add_argument(
        "--output-dir", "-o", type=str, default="./clips",
        help="Output directory for saved clips (default: ./clips)",
    )
    play_parser.add_argument(
        "--no-overlay", action="store_true",
        help="Hide info overlay on video",
    )
    play_parser.add_argument(
        "--no-window", action="store_true",
        help="Headless mode (no GUI window)",
    )
    play_parser.add_argument(
        "--no-reconnect", action="store_true",
        help="Disable auto-reconnection for RTSP streams",
    )
    play_parser.add_argument(
        "--decoder", "-d", type=str, default="auto",
        choices=["auto", "nvdec", "vaapi", "cpu"],
        help="Video decoder (default: auto). nvdec=NVIDIA, vaapi=Intel/AMD, cpu=software",
    )

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show video file information",
    )
    info_parser.add_argument("path", help="Path to video file")

    # clip command
    clip_parser = subparsers.add_parser(
        "clip",
        help="Extract a clip from video file",
    )
    clip_parser.add_argument("path", help="Path to video file")
    clip_parser.add_argument(
        "--time", "-t", type=float, required=True,
        help="Event time in seconds",
    )
    clip_parser.add_argument(
        "--pre", type=float, default=3.0,
        help="Seconds before event (default: 3.0)",
    )
    clip_parser.add_argument(
        "--post", type=float, default=2.0,
        help="Seconds after event (default: 2.0)",
    )
    clip_parser.add_argument(
        "--output-dir", "-o", type=str, default="./clips",
        help="Output directory (default: ./clips)",
    )
    clip_parser.add_argument(
        "--label", "-l", type=str, default="clip",
        help="Label for the clip (default: clip)",
    )

    # daemon command (ZMQ streaming)
    daemon_parser = subparsers.add_parser(
        "daemon",
        help="Run as ZMQ streaming daemon (allows dynamic subscriber attach/detach)",
        description="Reads from source and publishes frames via ZMQ PUB/SUB",
    )
    daemon_parser.add_argument(
        "source",
        help="Video source: camera device (0 or /dev/video0) or RTSP URL",
    )
    daemon_parser.add_argument(
        "--pub", "-p", type=str, default="tcp://*:5555",
        help="ZMQ PUB address (default: tcp://*:5555)",
    )
    daemon_parser.add_argument(
        "--fps", type=float, default=0,
        help="Target FPS (0 = source FPS)",
    )
    daemon_parser.add_argument(
        "--hwm", type=int, default=2,
        help="ZMQ high water mark (default: 2)",
    )

    # subscribe command (ZMQ debug/verify)
    sub_parser = subparsers.add_parser(
        "subscribe",
        help="Subscribe to ZMQ stream (for testing/verification)",
        description="Connect to a ZMQ publisher and display received frames",
    )
    sub_parser.add_argument(
        "address",
        help="ZMQ address to connect (e.g., tcp://localhost:5555)",
    )
    sub_parser.add_argument(
        "--no-window", action="store_true",
        help="Headless mode (no GUI, just stats)",
    )

    # webrtc command (browser streaming)
    webrtc_parser = subparsers.add_parser(
        "webrtc",
        help="Stream video to browser via WebRTC",
        description="Start WebRTC server for browser-based video viewing",
    )
    webrtc_parser.add_argument(
        "source",
        help="Video source: file, camera (0 or /dev/video0), or RTSP URL",
    )
    webrtc_parser.add_argument(
        "--port", "-p", type=int, default=8080,
        help="HTTP server port (default: 8080)",
    )
    webrtc_parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="HTTP server host (default: 0.0.0.0)",
    )
    webrtc_parser.add_argument(
        "--fps", type=int, default=30,
        help="Target FPS (default: 30)",
    )
    webrtc_parser.add_argument(
        "--decoder", "-d", type=str, default="auto",
        choices=["auto", "nvdec", "vaapi", "cpu"],
        help="Video decoder (default: auto). nvdec=NVIDIA, vaapi=Intel/AMD, cpu=software",
    )

    # ingest command (A-B*-C architecture)
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Run ingest process (A module) for A-B*-C architecture",
    )
    ingest_parser.add_argument(
        "source",
        help="Video source: camera device (0 or /dev/video0) or RTSP URL",
    )
    ingest_parser.add_argument(
        "--trig-socket", type=str, default="/tmp/trig.sock",
        help="UDS socket path for TRIG messages (default: /tmp/trig.sock)",
    )
    ingest_parser.add_argument(
        "--proxy", action="append", default=[],
        metavar="NAME:PATH:W:H:FPS",
        help="Proxy output: name:fifo_path:width:height:fps (can repeat)",
    )
    ingest_parser.add_argument(
        "--buffer", "-b", type=float, default=120.0,
        help="Ring buffer retention in seconds (default: 120)",
    )
    ingest_parser.add_argument(
        "--output-dir", "-o", type=str, default="./clips",
        help="Output directory for clips (default: ./clips)",
    )

    args = parser.parse_args()

    if args.command == "play":
        _cmd_play(args)
    elif args.command == "info":
        _cmd_info(args)
    elif args.command == "clip":
        _cmd_clip(args)
    elif args.command == "daemon":
        _cmd_daemon(args)
    elif args.command == "subscribe":
        _cmd_subscribe(args)
    elif args.command == "webrtc":
        _cmd_webrtc(args)
    elif args.command == "ingest":
        _cmd_ingest(args)
    else:
        parser.print_help()
        sys.exit(1)


def _parse_resolution(res_str: str) -> tuple[int, int]:
    """Parse resolution string like '640x480' to (width, height)."""
    try:
        w, h = res_str.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise ValueError(f"Invalid resolution format: {res_str} (expected WIDTHxHEIGHT)")


def _cmd_play(args):
    """Play video from file, camera, or RTSP stream."""
    source_type = _detect_source_type(args.source)

    # Parse resolution if provided
    resolution = None
    if args.resolution:
        try:
            resolution = _parse_resolution(args.resolution)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    if source_type == "file":
        _play_file(args, resolution)
    elif source_type == "camera":
        _play_camera(args, resolution)
    elif source_type == "rtsp":
        _play_rtsp(args, resolution)


def _play_file(args, resolution):
    """Play video file."""
    from visualbase import VisualBase, FileSource

    fps = int(args.fps) if args.fps > 0 else 0
    decoder = getattr(args, 'decoder', 'auto')

    print(f"Playing: {args.source}")
    print(f"Decoder: {decoder}")

    try:
        source = FileSource(args.source, decoder=decoder)
        source.open()
        print(f"Using decoder: {source.decoder}")
    except IOError as e:
        print(f"Error: {e}")
        sys.exit(1)

    vb = VisualBase(clip_output_dir=args.output_dir)
    vb.connect(source, ring_buffer_retention_sec=args.buffer)

    _print_stream_info(vb, "File", args)
    _run_stream(vb, args, "File")


def _play_camera(args, resolution):
    """Play from USB camera."""
    from visualbase import VisualBase, CameraSource

    # Parse device ID
    source = args.source
    if source.startswith("/dev/video"):
        device_id = int(source.replace("/dev/video", ""))
    else:
        device_id = int(source)

    width, height = resolution if resolution else (None, None)
    fps = int(args.fps) if args.fps > 0 else 30

    print(f"Connecting to camera /dev/video{device_id}...")

    try:
        source_obj = CameraSource(
            device_id=device_id,
            width=width,
            height=height,
            fps=fps,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    vb = VisualBase(clip_output_dir=args.output_dir)

    try:
        vb.connect(source_obj, ring_buffer_retention_sec=args.buffer)
    except IOError as e:
        print(f"Error: {e}")
        sys.exit(1)

    _print_stream_info(vb, "Camera", args)
    _run_stream(vb, args, "Camera")


def _play_rtsp(args, resolution):
    """Play from RTSP stream."""
    from visualbase import VisualBase, RTSPSource

    fps = int(args.fps) if args.fps > 0 else 30
    decoder = getattr(args, 'decoder', 'auto')

    print(f"Connecting to {args.source}...")
    print(f"Decoder: {decoder}")

    try:
        source_obj = RTSPSource(
            url=args.source,
            reconnect=not args.no_reconnect,
            decoder=decoder,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    vb = VisualBase(clip_output_dir=args.output_dir)

    try:
        vb.connect(source_obj, ring_buffer_retention_sec=args.buffer)
        print(f"Using decoder: {source_obj.decoder}")
    except IOError as e:
        print(f"Error: {e}")
        sys.exit(1)

    _print_stream_info(vb, "RTSP", args)
    _run_stream(vb, args, "RTSP")


def _print_stream_info(vb, source_type: str, args):
    """Print stream connection info."""
    print(f"Connected: {vb.source.width}x{vb.source.height} @ {vb.source.fps:.1f}fps")
    print(f"Buffer: {args.buffer}s | Output: {args.output_dir}")
    if not args.no_window:
        print("Keys: [q] quit  [s] save clip  [i] buffer info")
    print("-" * 50)


def _run_stream(vb, args, source_type: str):
    """Run stream with optional GUI."""
    if args.no_window:
        _run_headless(vb, args)
    else:
        _run_with_window(vb, args, source_type)


def _run_with_window(vb, args, source_type: str):
    """Run with GUI window."""
    import cv2
    from visualbase import Trigger

    fps = int(args.fps) if args.fps > 0 else 0  # 0 = use source fps
    frame_count = 0
    clip_count = 0
    window_name = f"VisualBase - {source_type}"

    try:
        for frame in vb.get_stream(fps=fps):
            frame_count += 1
            display = frame.data.copy()

            # Overlay info
            if not args.no_overlay:
                info = vb.get_buffer_info()
                text = f"Frame {frame_count} | {frame.t_src_ns / 1e9:.1f}s | Buffer {info.duration_sec:.1f}s | Clips {clip_count}"
                cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # q or ESC
                break
            elif key == ord("s"):
                trigger = Trigger.point(
                    event_time_ns=frame.t_src_ns,
                    pre_sec=3.0,
                    post_sec=1.0,
                    label="manual",
                )
                result = vb.trigger(trigger)
                if result.success:
                    clip_count += 1
                    print(f"Saved: {result.output_path} ({result.duration_sec:.1f}s)")
                else:
                    print(f"Failed: {result.error}")
            elif key == ord("i"):
                info = vb.get_buffer_info()
                print(f"Buffer: [{info.start_ns/1e9:.1f}s - {info.end_ns/1e9:.1f}s] = {info.duration_sec:.1f}s")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        vb.disconnect()
        print(f"Finished: {frame_count} frames, {clip_count} clips")


def _run_headless(vb, args):
    """Run in headless mode."""
    fps = int(args.fps) if args.fps > 0 else 0  # 0 = use source fps
    frame_count = 0
    last_report = 0

    print("Headless mode - Ctrl+C to stop")

    try:
        for frame in vb.get_stream(fps=fps):
            frame_count += 1
            if frame_count - last_report >= 30:
                info = vb.get_buffer_info()
                print(f"Frame {frame_count}: {frame.t_src_ns/1e9:.1f}s, buffer={info.duration_sec:.1f}s")
                last_report = frame_count
    except KeyboardInterrupt:
        pass
    finally:
        vb.disconnect()
        print(f"Finished: {frame_count} frames")


def _cmd_info(args):
    """Show video file information."""
    from visualbase import FileSource

    try:
        with FileSource(args.path) as source:
            print(f"File:       {source.path}")
            print(f"Resolution: {source.width}x{source.height}")
            print(f"FPS:        {source.fps:.2f}")
            print(f"Frames:     {source.frame_count}")
            print(f"Duration:   {source.duration_sec:.2f}s")
    except IOError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_clip(args):
    """Extract a clip from video file."""
    from visualbase import VisualBase, FileSource, Trigger

    event_time_ns = int(args.time * 1_000_000_000)

    try:
        with VisualBase(clip_output_dir=args.output_dir) as vb:
            vb.connect(FileSource(args.path))

            trigger = Trigger.point(
                event_time_ns=event_time_ns,
                pre_sec=args.pre,
                post_sec=args.post,
                label=args.label,
            )

            print(f"Source: {args.path}")
            print(f"Event:  {args.time:.2f}s")
            print(f"Range:  [{trigger.clip_start_sec:.2f}s - {trigger.clip_end_sec:.2f}s]")

            result = vb.trigger(trigger)

            if result.success:
                print(f"Output: {result.output_path}")
                print(f"Duration: {result.duration_sec:.2f}s")
            else:
                print(f"Error: {result.error}")
                sys.exit(1)

    except IOError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_daemon(args):
    """Run as ZMQ streaming daemon."""
    from visualbase.sources.camera import CameraSource
    from visualbase.sources.rtsp import RTSPSource
    from visualbase.daemon import VideoDaemon

    source_type = _detect_source_type(args.source)

    # Create source
    if source_type == "camera":
        if args.source.startswith("/dev/video"):
            device_id = int(args.source.replace("/dev/video", ""))
        else:
            device_id = int(args.source)
        print(f"Source: Camera /dev/video{device_id}")
        source = CameraSource(device_id=device_id)
    elif source_type == "rtsp":
        print(f"Source: RTSP {args.source}")
        source = RTSPSource(url=args.source)
    else:
        print(f"Error: Daemon requires camera or RTSP source, not file")
        sys.exit(1)

    fps = int(args.fps) if args.fps > 0 else 0

    print(f"ZMQ PUB: {args.pub}")
    print(f"HWM: {args.hwm}, Target FPS: {fps if fps else 'source'}")
    print("-" * 50)
    print("Daemon running - Ctrl+C to stop")
    print("Subscribers can connect to:", args.pub.replace("*", "localhost"))

    daemon = VideoDaemon(
        source=source,
        pub_address=args.pub,
        fps=fps,
        hwm=args.hwm,
    )

    try:
        daemon.run()
    except KeyboardInterrupt:
        pass
    finally:
        stats = daemon.get_stats()
        print(f"\nStats: {stats['frame_count']} frames @ {stats['fps']:.1f}fps")


def _cmd_webrtc(args):
    """Stream video to browser via WebRTC."""
    try:
        from visualbase.webrtc import WebRTCServer
    except ImportError:
        print("Error: WebRTC not available. Install with: uv sync --extra webrtc")
        sys.exit(1)

    source_type = _detect_source_type(args.source)

    decoder = getattr(args, 'decoder', 'auto')

    # Create video source
    if source_type == "file":
        from visualbase import FileSource
        source = FileSource(args.source, decoder=decoder)
        print(f"Source: File {args.source}")
        print(f"Decoder: {decoder}")
    elif source_type == "camera":
        from visualbase import CameraSource
        if args.source.startswith("/dev/video"):
            device_id = int(args.source.replace("/dev/video", ""))
        else:
            device_id = int(args.source)
        source = CameraSource(device_id=device_id, fps=args.fps)
        print(f"Source: Camera /dev/video{device_id}")
    elif source_type == "rtsp":
        from visualbase import RTSPSource
        source = RTSPSource(url=args.source, decoder=decoder)
        print(f"Source: RTSP {args.source}")
        print(f"Decoder: {decoder}")
    else:
        print(f"Error: Unknown source type")
        sys.exit(1)

    # Open source
    try:
        source.open()
    except IOError as e:
        print(f"Error: Cannot open source: {e}")
        sys.exit(1)

    # Show actual decoder used
    if hasattr(source, 'decoder'):
        print(f"Using decoder: {source.decoder}")

    print(f"Resolution: {source.width}x{source.height} @ {source.fps:.1f}fps")
    print(f"WebRTC server: http://{args.host}:{args.port}")
    print("-" * 50)

    # Frame callback for WebRTC
    import threading
    import time
    frame_lock = threading.Lock()
    current_frame = [None]

    # Calculate frame interval for rate limiting (file sources need this)
    target_fps = args.fps if args.fps > 0 else (source.fps if source.fps else 30)
    frame_interval = 1.0 / target_fps

    def update_frame():
        """Background thread to read frames."""
        next_frame_time = time.time()
        while True:
            frame = source.read()
            if frame is None:
                # For file sources, loop back to beginning
                if source_type == "file":
                    source.seek(0)
                    frame = source.read()
                    if frame is None:
                        break
                else:
                    break
            with frame_lock:
                current_frame[0] = frame.data

            # Rate limiting for file sources
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_frame():
        """Get current frame for WebRTC."""
        with frame_lock:
            return current_frame[0]

    # Start frame reader thread
    reader_thread = threading.Thread(target=update_frame, daemon=True)
    reader_thread.start()

    # Start WebRTC server
    server = WebRTCServer(
        frame_callback=get_frame,
        host=args.host,
        port=args.port,
        fps=args.fps,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        source.close()


def _cmd_subscribe(args):
    """Subscribe to ZMQ stream for testing."""
    import time

    try:
        from visualbase.ipc.zmq_transport import ZMQVideoSubscriber
    except ImportError:
        print("Error: ZMQ not available. Install with: uv sync --extra zmq")
        sys.exit(1)

    print(f"Subscribing to: {args.address}")
    print("-" * 50)

    sub = ZMQVideoSubscriber(args.address, timeout_ms=1000)
    if not sub.open():
        print(f"Error: Cannot connect to {args.address}")
        sys.exit(1)

    print(f"Connected. Waiting for frames...")
    if not args.no_window:
        print("Keys: [q] quit")

    frame_count = 0
    start_time = time.time()

    try:
        if args.no_window:
            # Headless mode - just print stats
            while True:
                frame = sub.read()
                if frame is not None:
                    frame_count += 1
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"\rFrames: {frame_count} | {frame.width}x{frame.height} | {fps:.1f} fps", end="", flush=True)
        else:
            # GUI mode
            import cv2
            window_name = "ZMQ Subscriber"
            while True:
                frame = sub.read()
                if frame is not None:
                    frame_count += 1
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0

                    display = frame.data.copy()
                    text = f"Frame {frame_count} | {frame.width}x{frame.height} | {fps:.1f} fps"
                    cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        if not args.no_window:
            import cv2
            cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print()
        print(f"Received {frame_count} frames in {elapsed:.1f}s ({fps:.1f} fps)")


def _cmd_ingest(args):
    """Run ingest process for A-B*-C architecture."""
    from pathlib import Path
    from visualbase.sources.camera import CameraSource
    from visualbase.sources.rtsp import RTSPSource
    from visualbase.streaming.fanout import ProxyConfig
    from visualbase.process.ingest import IngestProcess

    source_type = _detect_source_type(args.source)

    # Create source
    if source_type == "camera":
        if args.source.startswith("/dev/video"):
            device_id = int(args.source.replace("/dev/video", ""))
        else:
            device_id = int(args.source)
        print(f"Source: Camera /dev/video{device_id}")
        source = CameraSource(device_id=device_id)
    elif source_type == "rtsp":
        print(f"Source: RTSP {args.source}")
        source = RTSPSource(url=args.source)
    else:
        print(f"Error: Ingest requires camera or RTSP source, not file")
        sys.exit(1)

    # Parse proxy configs
    proxy_configs = []
    for proxy_str in args.proxy:
        try:
            parts = proxy_str.split(":")
            if len(parts) != 5:
                raise ValueError("Expected 5 parts")
            name, path, w, h, fps = parts
            proxy_configs.append(ProxyConfig(name, path, int(w), int(h), float(fps)))
        except ValueError as e:
            print(f"Error: Invalid proxy '{proxy_str}' - expected NAME:PATH:W:H:FPS")
            sys.exit(1)

    # Default proxies if none specified
    if not proxy_configs:
        proxy_configs = [
            ProxyConfig("face", "/tmp/vid_face.mjpg", 640, 480, 10),
            ProxyConfig("pose", "/tmp/vid_pose.mjpg", 640, 480, 10),
            ProxyConfig("quality", "/tmp/vid_quality.mjpg", 320, 240, 5),
        ]
        print("Proxies: (default)")
        for cfg in proxy_configs:
            print(f"  {cfg.name}: {cfg.fifo_path} {cfg.width}x{cfg.height}@{cfg.fps}fps")

    print(f"Buffer: {args.buffer}s")
    print(f"TRIG socket: {args.trig_socket}")
    print(f"Output: {args.output_dir}")
    print("-" * 50)

    process = IngestProcess(
        source=source,
        proxy_configs=proxy_configs,
        trig_socket=args.trig_socket,
        clip_output_dir=Path(args.output_dir),
        ring_buffer_retention_sec=args.buffer,
    )

    try:
        process.run()
    except KeyboardInterrupt:
        pass
    finally:
        stats = process.get_stats()
        print(f"\nStats: {stats['frames_captured']} frames @ {stats['fps']:.1f}fps")
        print(f"       {stats['triggers_received']} triggers, {stats['clips_extracted']} clips")


if __name__ == "__main__":
    main()
