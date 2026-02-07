"""Abstract interfaces for IPC transports.

Defines ABCs for video streaming and message passing to enable
swappable transport implementations (FIFO/UDS, ZMQ, etc.).

Video Streaming (A->B*):
- VideoReader: Receive video frames
- VideoWriter: Send video frames

Message Passing (B*->C, C->A):
- MessageReceiver: Receive text messages (server-side)
- MessageSender: Send text messages (client-side)

Example:
    # Code depends on interface, not implementation
    def process_frames(reader: VideoReader) -> None:
        reader.open()
        while reader.is_open:
            frame = reader.read()
            if frame:
                process(frame)
        reader.close()

    # At runtime, inject concrete implementation
    reader = FIFOVideoReader("/tmp/vid.mjpg")  # or ZMQVideoReader(...)
    process_frames(reader)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from visualbase.core.frame import Frame


class VideoReader(ABC):
    """Abstract interface for reading video frames.

    Used by Extractor processes (B*) to receive frames from Ingest (A).
    Implementations must be iterable over frames.

    Example:
        >>> reader = FIFOVideoReader("/tmp/vid.mjpg")
        >>> if reader.open(timeout_sec=5.0):
        ...     for frame in reader:
        ...         process(frame)
        ...     reader.close()
    """

    @abstractmethod
    def open(self, timeout_sec: Optional[float] = None) -> bool:
        """Open the reader for receiving frames.

        Args:
            timeout_sec: Optional timeout in seconds. None for indefinite wait.

        Returns:
            True if opened successfully, False on failure or timeout.
        """
        ...

    @abstractmethod
    def read(self) -> Optional["Frame"]:
        """Read the next frame.

        Returns:
            Frame if available, None on EOF or error.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the reader and release resources."""
        ...

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Check if the reader is open and ready."""
        ...

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __iter__(self) -> "VideoReader":
        return self

    def __next__(self) -> "Frame":
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame


class VideoWriter(ABC):
    """Abstract interface for writing video frames.

    Used by Ingest process (A) to send frames to Extractors (B*).

    Example:
        >>> writer = FIFOVideoWriter("/tmp/vid.mjpg")
        >>> writer.open()  # blocks until reader connects
        >>> for frame in source:
        ...     writer.write(frame)
        >>> writer.close()
    """

    @abstractmethod
    def open(self) -> None:
        """Open the writer for sending frames.

        This may block until a reader connects (depending on implementation).
        """
        ...

    @abstractmethod
    def write(self, frame: "Frame") -> bool:
        """Write a frame.

        Args:
            frame: Frame to write.

        Returns:
            True if write succeeded, False on error.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the writer and release resources."""
        ...

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Check if the writer is open and ready."""
        ...

    def __enter__(self) -> "VideoWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class MessageReceiver(ABC):
    """Abstract interface for receiving messages (server-side).

    Used by Fusion (C) to receive OBS messages and Ingest (A) for TRIG.

    Example:
        >>> server = UDSServer("/tmp/obs.sock")
        >>> server.start()
        >>> while server.is_running:
        ...     msg = server.recv(timeout=1.0)
        ...     if msg:
        ...         process(msg)
        >>> server.stop()
    """

    @abstractmethod
    def start(self) -> None:
        """Start the receiver (bind and listen)."""
        ...

    @abstractmethod
    def recv(self, timeout: Optional[float] = None) -> Optional[str]:
        """Receive a single message.

        Args:
            timeout: Timeout in seconds. None for non-blocking.

        Returns:
            Message string if received, None on timeout or error.
        """
        ...

    @abstractmethod
    def recv_all(self, max_messages: int = 100) -> List[str]:
        """Receive all pending messages.

        Args:
            max_messages: Maximum messages to receive in one call.

        Returns:
            List of message strings (may be empty).
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the receiver and release resources."""
        ...

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the receiver is running."""
        ...

    def __enter__(self) -> "MessageReceiver":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class MessageSender(ABC):
    """Abstract interface for sending messages (client-side).

    Used by Extractors (B*) to send OBS and Fusion (C) for TRIG.

    Example:
        >>> client = UDSClient("/tmp/obs.sock")
        >>> if client.connect():
        ...     client.send("OBS src=face ...")
        ...     client.disconnect()
    """

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the message receiver.

        Returns:
            True if connected successfully, False on error.
        """
        ...

    @abstractmethod
    def send(self, message: str) -> bool:
        """Send a message.

        Args:
            message: Message string to send.

        Returns:
            True if sent successfully, False on error.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect and release resources."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected."""
        ...

    def __enter__(self) -> "MessageSender":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
