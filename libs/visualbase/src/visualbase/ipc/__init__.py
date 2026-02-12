"""IPC (Inter-Process Communication) module for A-B*-C architecture.

This module provides IPC primitives for inter-process communication:
- FIFO: Named pipes for video stream fan-out (A→B*)
- UDS: Unix Domain Sockets for message passing (B*→C, C→A)
- ZMQ: ZeroMQ PUB/SUB for dynamic connections (optional, requires pyzmq)
- ZMQ RPC: ZeroMQ REQ-REP for synchronous RPC (optional, requires pyzmq)
- Messages: OBS/TRIG message parsing and serialization
- Interfaces: ABCs for swappable transport implementations
- Factory: Unified transport creation
- Codec: Frame encode/decode for IPC transmission
- Utilities: ZMQ availability check, IPC address generation

Example using interfaces (recommended for production code):
    >>> from visualbase.ipc import VideoReader, TransportFactory
    >>>
    >>> def process_frames(reader: VideoReader):
    ...     reader.open()
    ...     for frame in reader:
    ...         process(frame)
    ...     reader.close()
    >>>
    >>> reader = TransportFactory.create_video_reader("fifo", "/tmp/vid.mjpg")
    >>> process_frames(reader)

Example using ZMQ (requires: uv sync --extra zmq):
    >>> reader = TransportFactory.create_video_reader("zmq", "tcp://localhost:5555")
    >>> writer = TransportFactory.create_video_writer("zmq", "tcp://*:5555")
"""

# Interfaces (ABCs)
from visualbase.ipc.interfaces import (
    VideoReader,
    VideoWriter,
    MessageReceiver,
    MessageSender,
    RPCServer,
    RPCClient,
)

# Concrete implementations
from visualbase.ipc.fifo import FIFOVideoWriter, FIFOVideoReader
from visualbase.ipc.uds import UDSServer, UDSClient

# Factory
from visualbase.ipc.factory import TransportFactory

# Codec
from visualbase.ipc.codec import encode_frame, decode_frame

# Utilities
from visualbase.ipc._util import check_zmq_available, generate_ipc_address

# Messages
from visualbase.ipc.messages import (
    OBSMessage,
    TRIGMessage,
    parse_obs_message,
    parse_trig_message,
    FaceOBS,
    PoseOBS,
    QualityOBS,
)

__all__ = [
    # Interfaces
    "VideoReader",
    "VideoWriter",
    "MessageReceiver",
    "MessageSender",
    "RPCServer",
    "RPCClient",
    # Factory
    "TransportFactory",
    # Codec
    "encode_frame",
    "decode_frame",
    # Utilities
    "check_zmq_available",
    "generate_ipc_address",
    # FIFO
    "FIFOVideoWriter",
    "FIFOVideoReader",
    # UDS
    "UDSServer",
    "UDSClient",
    # Messages
    "OBSMessage",
    "TRIGMessage",
    "parse_obs_message",
    "parse_trig_message",
    "FaceOBS",
    "PoseOBS",
    "QualityOBS",
]

# Optional ZMQ transports (requires pyzmq)
try:
    from visualbase.ipc.zmq_transport import (
        ZMQVideoPublisher,
        ZMQVideoSubscriber,
        ZMQMessagePublisher,
        ZMQMessageSubscriber,
    )
    from visualbase.ipc.zmq_rpc import (
        ZMQRPCServer,
        ZMQRPCClient,
    )
    __all__.extend([
        "ZMQVideoPublisher",
        "ZMQVideoSubscriber",
        "ZMQMessagePublisher",
        "ZMQMessageSubscriber",
        "ZMQRPCServer",
        "ZMQRPCClient",
    ])
except ImportError:
    pass  # pyzmq not installed
