"""CLI utility functions and compatibility layer.

Third-party log noise suppression
==================================

CLI 실행 시 서드파티 라이브러리의 C++ stdout/stderr 출력과 내부 로거의
과도한 INFO 메시지를 억제한다. 모든 억제는 -v (verbose) 플래그로 해제 가능.

조치 전체 목록:

1. 환경변수 (suppress_thirdparty_noise)                          ← 이 파일
   - QT_LOGGING_RULES, QT_QPA_PLATFORM  : Qt debug/wayland 경고
   - OPENCV_LOG_LEVEL=ERROR              : OpenCV 내부 로그
   - TF_CPP_MIN_LOG_LEVEL=2              : TF Lite C++ INFO 억제
   - GLOG_minloglevel=2                  : abseil/MediaPipe C++ INFO 억제
   - ORT_LOG_LEVEL=ERROR                 : ONNX Runtime C++ 로그

2. Python warnings 필터 (suppress_thirdparty_noise)              ← 이 파일
   - FutureWarning from skimage          : insightface face_align 사용
   - FutureWarning from insightface      : 내부 deprecation

3. fd-level stderr 필터 (StderrFilter)                           ← 이 파일
   C++ 라이브러리는 fd 2에 직접 write하므로 os.dup2+pipe로 가로챈다.
   - QFontDatabase: / Qt no longer ships fonts  : OpenCV Qt 폰트 경고
   - XDG_SESSION_TYPE=wayland / qt.qpa.         : Qt 플랫폼 경고
   - Created TensorFlow Lite                    : TF Lite delegate 메시지
   - inference_feedback_manager                 : MediaPipe C++ warning
   - landmark_projection_calculator             : MediaPipe C++ warning
   - Applied providers:                         : ONNX Runtime provider 목록
   - absl::InitializeLog()                      : abseil 시작 배너

4. Python 로거 레벨 조정 (configure_log_levels)                  ← 이 파일
   비-verbose 모드에서 INFO → WARNING 승격:
   - vpx.*                                : analyzer init/cleanup 메시지
   - facemoment.main                      : CUDA conflict 등 내부 로그
   - facemoment.moment_detector.analyzers: classifier init/cleanup
   - facemoment.pipeline.pathway_pipeline : debug header 중복
   - visualpath.process.launcher          : worker 시작/handshake
   - visualbase.sources.decoder           : NVDEC/VAAPI 선택 로그

5. stdout 리다이렉트 (개별 백엔드 파일)
   insightface/onnxruntime/hsemotion 초기화 시 print() 억제:
   - vpx/face-detect/.../insightface.py   : FaceAnalysis()+prepare()
     + ort.set_default_logger_severity(3)   (ONNX Python 로거 ERROR만)
     + contextlib.redirect_stdout           (find model, set det-size, Applied providers)
   - vpx/expression/.../hsemotion.py      : HSEmotionRecognizer()
     + contextlib.redirect_stdout           (ONNX provider 출력)

6. 로그 레벨 변경 (개별 소스 파일)
   - visualbase/sources/decoder.py        : NVDEC/VAAPI 성공 로그 INFO→DEBUG
   - vpx/face-detect/.../insightface.py   : Available ONNX providers INFO→DEBUG
"""

import logging
import os
import sys
import warnings

# ANSI formatting (disabled when stdout is not a terminal)
_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
BOLD = "\033[1m" if _TTY else ""
DIM = "\033[2m" if _TTY else ""
ITALIC = "\033[3m" if _TTY else ""
RESET = "\033[0m" if _TTY else ""


def suppress_thirdparty_noise():
    """Suppress third-party C++/stdout/stderr noise for cleaner CLI output.

    See module docstring for full suppression inventory.
    """
    # [1] 환경변수 — C++ 라이브러리 로그 레벨
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("ORT_LOG_LEVEL", "ERROR")

    # [2] Python warnings 필터
    warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")
    warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")


# Backwards-compatible alias
suppress_qt_warnings = suppress_thirdparty_noise


def configure_log_levels():
    """Adjust internal logger levels to reduce INFO noise.

    Call after logging.basicConfig(). See module docstring [4] for full list.
    Non-verbose 모드에서만 적용. -v 시 logging.DEBUG로 모든 메시지 노출.
    """
    for name in (
        "vpx",
        "facemoment.main",
        "facemoment.moment_detector.analyzers",
        "facemoment.pipeline.pathway_pipeline",
        "visualpath.process.launcher",
        "visualbase.sources.decoder",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


class StderrFilter:
    """Filter native C++ stderr at the file descriptor level.

    C++ 라이브러리는 Python sys.stderr를 우회하여 fd 2에 직접 write한다.
    os.dup2 + pipe + reader thread로 fd 2를 가로채 패턴 매칭 필터링.
    See module docstring [3] for pattern inventory.
    """

    # [3] fd-level stderr 필터 패턴
    SUPPRESS_PATTERNS = (
        "QFontDatabase:",                   # OpenCV Qt 폰트 경고
        "Note that Qt no longer ships fonts",
        "XDG_SESSION_TYPE=wayland",         # Qt 플랫폼 경고
        "qt.qpa.",
        "Created TensorFlow Lite",          # TF Lite delegate
        "inference_feedback_manager",       # MediaPipe C++
        "landmark_projection_calculator",   # MediaPipe C++
        "Applied providers:",               # ONNX Runtime
        "absl::InitializeLog()",            # abseil 시작 배너
    )

    def install(self):
        """Replace fd 2 with a filtered pipe."""
        import threading

        try:
            sys.stderr.flush()
            self._original_fd = os.dup(2)
            pipe_r, pipe_w = os.pipe()
            os.dup2(pipe_w, 2)
            os.close(pipe_w)
            sys.stderr = open(2, "w", closefd=False)
            threading.Thread(
                target=self._reader, args=(pipe_r,), daemon=True
            ).start()
        except OSError:
            pass  # fallback: no fd-level filtering (e.g. redirected stderr)

    def _reader(self, pipe_r):
        """Read lines from pipe, suppress matching patterns, forward the rest."""
        try:
            buf = b""
            while True:
                chunk = os.read(pipe_r, 8192)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace")
                    if not any(p in text for p in self.SUPPRESS_PATTERNS):
                        os.write(self._original_fd, line + b"\n")
            if buf:
                text = buf.decode("utf-8", errors="replace")
                if not any(p in text for p in self.SUPPRESS_PATTERNS):
                    os.write(self._original_fd, buf)
        except OSError:
            pass
        finally:
            os.close(pipe_r)


class VideoSourceInfo:
    """Video source metadata container for compatibility layer."""

    def __init__(self, fps: float, frame_count: int, width: int, height: int):
        self.fps = fps
        self.frame_count = frame_count
        self.width = width
        self.height = height


class LegacyVideoStream:
    """Fallback video stream using cv2.VideoCapture when visualbase API is incompatible."""

    def __init__(self, path: str, fps: float, resolution: tuple = None):
        import cv2
        from visualbase import Frame

        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_skip = max(1, int(self._video_fps / fps)) if fps > 0 else 1
        self._resolution = resolution
        self._frame_id = 0
        self._Frame = Frame

        self.info = VideoSourceInfo(
            fps=self._video_fps,
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def __iter__(self):
        return self

    def __next__(self):
        import cv2

        while True:
            ret, image = self._cap.read()
            if not ret:
                raise StopIteration

            frame_id = self._frame_id
            self._frame_id += 1

            if frame_id % self._frame_skip != 0:
                continue

            if self._resolution:
                image = cv2.resize(image, self._resolution)

            t_ns = int(frame_id / self._video_fps * 1e9)
            return self._Frame.from_array(image, frame_id=frame_id, t_src_ns=t_ns)

    def disconnect(self):
        if self._cap:
            self._cap.release()
            self._cap = None


def create_video_stream(path: str, fps: float = 10.0, resolution: tuple = None):
    """Create visualbase video stream from file path.

    Falls back to cv2.VideoCapture if visualbase API is incompatible.

    Args:
        path: Path to the video file.
        fps: Target frame rate for processing.
        resolution: Optional (width, height) for resizing.

    Returns:
        Tuple of (vb, source, stream) where:
        - vb: VisualBase instance or LegacyVideoStream (call vb.disconnect() when done)
        - source: Object with metadata (fps, frame_count, width, height)
        - stream: Iterator[Frame] for processing
    """
    try:
        from visualbase import VisualBase, FileSource

        source = FileSource(path)
        source.open()

        required_attrs = ['fps', 'frame_count', 'width', 'height']
        for attr in required_attrs:
            if not hasattr(source, attr):
                raise AttributeError(f"FileSource missing '{attr}' attribute")

        vb = VisualBase()

        if not hasattr(vb, 'connect') or not hasattr(vb, 'get_stream'):
            raise AttributeError("VisualBase missing required methods")

        vb.connect(source)
        stream = vb.get_stream(fps=int(fps), resolution=resolution)
        return vb, source, stream

    except (ImportError, AttributeError, TypeError) as e:
        import logging
        logging.warning(f"visualbase API incompatible, using fallback: {e}")

        legacy = LegacyVideoStream(path, fps, resolution)
        return legacy, legacy.info, legacy


def check_ml_dependencies(module_name: str, require_expression: bool = False) -> bool:
    """Check if ML dependencies are available."""
    deps = {
        "face": ["insightface", "onnxruntime"],
        "face_expression": ["insightface", "onnxruntime", "feat"],
        "pose": ["ultralytics"],
    }

    if module_name == "face" and require_expression:
        module_name = "face_expression"
    missing = []
    for dep in deps.get(module_name, []):
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)

    if missing:
        print(f"Error: ML dependencies not installed for {module_name} analyzer.")
        print(f"Missing: {', '.join(missing)}")
        print()
        print("Install with:")
        print("  uv sync --extra ml")
        print()
        print("Or install individually:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True


def setup_observability(trace_level: str, trace_output: str = None):
    """Configure observability based on CLI arguments.

    Args:
        trace_level: Trace level string ("off", "minimal", "normal", "verbose").
        trace_output: Optional path to output JSONL file.

    Returns:
        Tuple of (hub, file_sink) for cleanup. file_sink may be None.
    """
    from facemoment.observability import ObservabilityHub, TraceLevel, FileSink, ConsoleSink

    level_map = {
        "off": TraceLevel.OFF,
        "minimal": TraceLevel.MINIMAL,
        "normal": TraceLevel.NORMAL,
        "verbose": TraceLevel.VERBOSE,
    }

    level = level_map.get(trace_level, TraceLevel.OFF)
    hub = ObservabilityHub.get_instance()

    if level == TraceLevel.OFF:
        return hub, None

    sinks = []
    sinks.append(ConsoleSink())

    file_sink = None
    if trace_output:
        file_sink = FileSink(trace_output)
        sinks.append(file_sink)

    hub.configure(level=level, sinks=sinks)
    print(f"Observability: level={trace_level}", end="")
    if trace_output:
        print(f", output={trace_output}")
    else:
        print()

    return hub, file_sink


def cleanup_observability(hub, file_sink):
    """Clean up observability resources."""
    if hub is not None:
        hub.shutdown()


def detect_distributed_mode(args) -> bool:
    """Check if distributed mode is requested via CLI args.

    Returns True if --distributed, --venv-*, or --config is specified.
    """
    distributed = getattr(args, 'distributed', False)
    config_path = getattr(args, 'config', None)
    venv_face = getattr(args, 'venv_face', None)
    venv_pose = getattr(args, 'venv_pose', None)
    venv_gesture = getattr(args, 'venv_gesture', None)

    if venv_face or venv_pose or venv_gesture or config_path:
        distributed = True

    return distributed


def detect_ml_mode(args) -> str:
    """Detect ML mode from CLI args.

    Returns:
        "auto", "enabled", or "disabled".
    """
    use_ml = getattr(args, 'use_ml', None)
    if use_ml is None:
        return "auto"
    return "enabled" if use_ml else "disabled"


def probe_analyzers(use_ml=None, device="cuda:0", roi=None) -> dict:
    """Probe which analyzers are available.

    Args:
        use_ml: None (auto), True (force ML), False (disable ML).
        device: Device for ML backends.
        roi: Optional ROI tuple for face analyzer.

    Returns:
        Dict with keys "face", "pose", "gesture", "quality" (bool),
        "names" (list of available analyzer names),
        and "face_mode" ("enabled", "disabled", or "dummy").
    """
    result = {
        "face": False,
        "pose": False,
        "gesture": False,
        "quality": True,
        "names": [],
        "face_mode": "disabled",
    }

    if use_ml is not False:
        # Try face analyzer
        try:
            from vpx.face import FaceAnalyzer
            FaceAnalyzer()
            result["face"] = True
            result["names"].append("face")
            result["face_mode"] = "enabled"
        except Exception:
            pass

        # Try pose analyzer
        try:
            from vpx.pose import PoseAnalyzer
            PoseAnalyzer()
            result["pose"] = True
            result["names"].append("pose")
        except Exception:
            pass

        # Try gesture analyzer
        try:
            from vpx.gesture import GestureAnalyzer
            GestureAnalyzer()
            result["gesture"] = True
            result["names"].append("gesture")
        except Exception:
            pass

    # Fall back to dummy if no face analyzer
    if not result["face"]:
        result["names"].insert(0, "dummy")
        result["face_mode"] = "dummy" if use_ml is False else "disabled"

    result["names"].append("quality")

    return result


def create_video_writer(path, fps, width, height):
    """Create cv2 VideoWriter for debug/output video.

    Args:
        path: Output file path.
        fps: Target FPS.
        width: Frame width.
        height: Frame height.

    Returns:
        cv2.VideoWriter instance.
    """
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def score_frame(scorer, observations):
    """Score a frame using FrameScorer with available observations.

    Args:
        scorer: FrameScorer instance (or None).
        observations: Dict of {analyzer_name: observation}.

    Returns:
        ScoreResult or None.
    """
    if scorer is None:
        return None
    return scorer.score(
        face_obs=observations.get("face") or observations.get("dummy"),
        pose_obs=observations.get("pose"),
        quality_obs=observations.get("quality"),
    )
