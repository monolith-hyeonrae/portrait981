"""Factory for creating IPC transport instances.

Provides a unified way to create transport implementations by type,
enabling configuration-driven transport selection.

Example:
    >>> # Create FIFO-based video transport
    >>> reader = TransportFactory.create_video_reader("fifo", "/tmp/vid.mjpg")
    >>> writer = TransportFactory.create_video_writer("fifo", "/tmp/vid.mjpg")

    >>> # Create UDS-based message transport
    >>> server = TransportFactory.create_message_receiver("uds", "/tmp/obs.sock")
    >>> client = TransportFactory.create_message_sender("uds", "/tmp/obs.sock")

    >>> # Create ZMQ-based transports (requires pyzmq)
    >>> reader = TransportFactory.create_video_reader("zmq", "tcp://localhost:5555")
    >>> writer = TransportFactory.create_video_writer("zmq", "tcp://*:5555")
"""

from typing import Optional, Any, Dict
import logging

from visualbase.ipc.interfaces import (
    VideoReader,
    VideoWriter,
    MessageReceiver,
    MessageSender,
)
from visualbase.ipc.fifo import FIFOVideoReader, FIFOVideoWriter
from visualbase.ipc.uds import UDSServer, UDSClient

logger = logging.getLogger(__name__)

# Try to import ZMQ transports (optional dependency)
try:
    from visualbase.ipc.zmq_transport import (
        ZMQVideoPublisher,
        ZMQVideoSubscriber,
        ZMQMessagePublisher,
        ZMQMessageSubscriber,
    )
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False


class TransportFactory:
    """Factory for creating IPC transport instances.

    Provides static methods to create transport implementations by type.
    Supports FIFO/UDS transports, with optional ZMQ support.

    Supported transport types:
        Video (VideoReader/VideoWriter):
            - "fifo": Named pipe (FIFO) based transport
            - "zmq": ZeroMQ PUB/SUB transport (requires pyzmq)

        Message (MessageReceiver/MessageSender):
            - "uds": Unix Domain Socket datagram transport
            - "zmq": ZeroMQ PUB/SUB transport (requires pyzmq)
    """

    # Registry of video transport implementations
    _video_readers: Dict[str, type] = {
        "fifo": FIFOVideoReader,
    }

    _video_writers: Dict[str, type] = {
        "fifo": FIFOVideoWriter,
    }

    # Registry of message transport implementations
    _message_receivers: Dict[str, type] = {
        "uds": UDSServer,
    }

    _message_senders: Dict[str, type] = {
        "uds": UDSClient,
    }

    # Register ZMQ transports if available
    if HAS_ZMQ:
        _video_readers["zmq"] = ZMQVideoSubscriber
        _video_writers["zmq"] = ZMQVideoPublisher
        _message_receivers["zmq"] = ZMQMessageSubscriber
        _message_senders["zmq"] = ZMQMessagePublisher

    @classmethod
    def create_video_reader(
        cls,
        transport_type: str,
        path: str,
        **kwargs: Any,
    ) -> VideoReader:
        """Create a video reader instance.

        Args:
            transport_type: Transport type ("fifo", "zmq" in future).
            path: Path or address for the transport.
            **kwargs: Additional arguments for the specific transport.

        Returns:
            VideoReader instance.

        Raises:
            ValueError: If transport type is not supported.
        """
        impl_class = cls._video_readers.get(transport_type)
        if impl_class is None:
            raise ValueError(
                f"Unknown video reader transport: {transport_type}. "
                f"Supported: {list(cls._video_readers.keys())}"
            )
        return impl_class(path, **kwargs)

    @classmethod
    def create_video_writer(
        cls,
        transport_type: str,
        path: str,
        **kwargs: Any,
    ) -> VideoWriter:
        """Create a video writer instance.

        Args:
            transport_type: Transport type ("fifo", "zmq" in future).
            path: Path or address for the transport.
            **kwargs: Additional arguments for the specific transport.

        Returns:
            VideoWriter instance.

        Raises:
            ValueError: If transport type is not supported.
        """
        impl_class = cls._video_writers.get(transport_type)
        if impl_class is None:
            raise ValueError(
                f"Unknown video writer transport: {transport_type}. "
                f"Supported: {list(cls._video_writers.keys())}"
            )
        return impl_class(path, **kwargs)

    @classmethod
    def create_message_receiver(
        cls,
        transport_type: str,
        path: str,
        **kwargs: Any,
    ) -> MessageReceiver:
        """Create a message receiver instance.

        Args:
            transport_type: Transport type ("uds", "zmq" in future).
            path: Path or address for the transport.
            **kwargs: Additional arguments for the specific transport.

        Returns:
            MessageReceiver instance.

        Raises:
            ValueError: If transport type is not supported.
        """
        impl_class = cls._message_receivers.get(transport_type)
        if impl_class is None:
            raise ValueError(
                f"Unknown message receiver transport: {transport_type}. "
                f"Supported: {list(cls._message_receivers.keys())}"
            )
        return impl_class(path, **kwargs)

    @classmethod
    def create_message_sender(
        cls,
        transport_type: str,
        path: str,
        **kwargs: Any,
    ) -> MessageSender:
        """Create a message sender instance.

        Args:
            transport_type: Transport type ("uds", "zmq" in future).
            path: Path or address for the transport.
            **kwargs: Additional arguments for the specific transport.

        Returns:
            MessageSender instance.

        Raises:
            ValueError: If transport type is not supported.
        """
        impl_class = cls._message_senders.get(transport_type)
        if impl_class is None:
            raise ValueError(
                f"Unknown message sender transport: {transport_type}. "
                f"Supported: {list(cls._message_senders.keys())}"
            )
        return impl_class(path, **kwargs)

    @classmethod
    def register_video_reader(cls, name: str, impl_class: type) -> None:
        """Register a custom video reader implementation.

        Args:
            name: Transport type name.
            impl_class: Class implementing VideoReader.
        """
        cls._video_readers[name] = impl_class
        logger.info(f"Registered video reader: {name}")

    @classmethod
    def register_video_writer(cls, name: str, impl_class: type) -> None:
        """Register a custom video writer implementation.

        Args:
            name: Transport type name.
            impl_class: Class implementing VideoWriter.
        """
        cls._video_writers[name] = impl_class
        logger.info(f"Registered video writer: {name}")

    @classmethod
    def register_message_receiver(cls, name: str, impl_class: type) -> None:
        """Register a custom message receiver implementation.

        Args:
            name: Transport type name.
            impl_class: Class implementing MessageReceiver.
        """
        cls._message_receivers[name] = impl_class
        logger.info(f"Registered message receiver: {name}")

    @classmethod
    def register_message_sender(cls, name: str, impl_class: type) -> None:
        """Register a custom message sender implementation.

        Args:
            name: Transport type name.
            impl_class: Class implementing MessageSender.
        """
        cls._message_senders[name] = impl_class
        logger.info(f"Registered message sender: {name}")

    @classmethod
    def list_video_transports(cls) -> Dict[str, list]:
        """List available video transport types.

        Returns:
            Dict with "readers" and "writers" keys listing available types.
        """
        return {
            "readers": list(cls._video_readers.keys()),
            "writers": list(cls._video_writers.keys()),
        }

    @classmethod
    def list_message_transports(cls) -> Dict[str, list]:
        """List available message transport types.

        Returns:
            Dict with "receivers" and "senders" keys listing available types.
        """
        return {
            "receivers": list(cls._message_receivers.keys()),
            "senders": list(cls._message_senders.keys()),
        }
