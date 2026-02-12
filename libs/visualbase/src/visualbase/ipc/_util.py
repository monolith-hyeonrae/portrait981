"""IPC utility functions.

Shared utilities for IPC address generation and dependency checking.
"""

import os
import tempfile


def check_zmq_available() -> bool:
    """Check if pyzmq is available.

    Returns:
        True if pyzmq can be imported.
    """
    try:
        import zmq  # noqa: F401
        return True
    except ImportError:
        return False


def generate_ipc_address(prefix: str = "visualbase") -> tuple[str, str]:
    """Generate a unique IPC address for ZMQ communication.

    Creates a temporary file path suitable for ZMQ IPC transport.

    Args:
        prefix: Prefix for the socket file name.

    Returns:
        Tuple of (zmq_address, file_path) where zmq_address is like
        "ipc:///tmp/visualbase-12345-xxxx.sock" and file_path is the
        underlying socket file path.
    """
    ipc_file = tempfile.mktemp(
        prefix=f"{prefix}-{os.getpid()}-",
        suffix=".sock",
    )
    ipc_address = f"ipc://{ipc_file}"
    return ipc_address, ipc_file
