from visualbase.api import VisualBase
from visualbase.core.frame import Frame
from visualbase.core.buffer import BaseBuffer, FileBuffer, BufferInfo
from visualbase.core.ring_buffer import RingBuffer
from visualbase.packaging.trigger import Trigger, TriggerType
from visualbase.packaging.clipper import ClipResult
from visualbase.sources.base import BaseSource
from visualbase.sources.file import FileSource
from visualbase.sources.camera import CameraSource
from visualbase.sources.rtsp import RTSPSource
from visualbase.tools.viewer import FrameViewer, play

# IPC modules for A-B*-C architecture
from visualbase.ipc import (
    FIFOVideoWriter,
    FIFOVideoReader,
    UDSServer,
    UDSClient,
    OBSMessage,
    TRIGMessage,
    parse_obs_message,
    parse_trig_message,
)
from visualbase.streaming import ProxyFanout, ProxyConfig
from visualbase.process import IngestProcess
from visualbase.daemon import VideoDaemon

__all__ = [
    # Main API
    "VisualBase",
    "Frame",
    # Buffers
    "BaseBuffer",
    "FileBuffer",
    "RingBuffer",
    "BufferInfo",
    # Triggers and clips
    "Trigger",
    "TriggerType",
    "ClipResult",
    # Sources
    "BaseSource",
    "FileSource",
    "CameraSource",
    "RTSPSource",
    # Tools
    "FrameViewer",
    "play",
    # IPC (Phase 8)
    "FIFOVideoWriter",
    "FIFOVideoReader",
    "UDSServer",
    "UDSClient",
    "OBSMessage",
    "TRIGMessage",
    "parse_obs_message",
    "parse_trig_message",
    # Streaming (Phase 8)
    "ProxyFanout",
    "ProxyConfig",
    # Process (Phase 8)
    "IngestProcess",
    # Daemon (Phase 8.7)
    "VideoDaemon",
]

# Optional WebRTC (requires aiortc, aiohttp)
try:
    from visualbase.webrtc import WebRTCServer
    __all__.append("WebRTCServer")
except ImportError:
    pass  # WebRTC dependencies not installed
