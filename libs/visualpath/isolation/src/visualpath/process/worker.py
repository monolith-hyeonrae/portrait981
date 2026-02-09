"""Subprocess entry point for VenvWorker.

This module provides the subprocess side of VenvWorker's ZMQ-based
IPC mechanism. It loads an analyzer via entry points and processes
frames received over ZMQ.

Usage:
    python -m visualpath.process.worker --analyzer face --ipc-address ipc:///tmp/xxx.sock

Or via the entry point:
    visualpath-worker --analyzer face --ipc-address ipc:///tmp/xxx.sock
"""

import argparse
import base64
import logging
import sys
import traceback
from typing import Any, Dict, Optional

import numpy as np

from visualpath.process.serialization import (
    serialize_observation,
    deserialize_observation,
)

logger = logging.getLogger(__name__)


def _deserialize_frame(data: Dict[str, Any]) -> "Frame":
    """Deserialize frame from ZMQ message.

    Args:
        data: Dict containing serialized frame data.

    Returns:
        Reconstructed Frame object.
    """
    import cv2
    from visualbase import Frame

    # Decode JPEG data
    jpeg_bytes = base64.b64decode(data["data_b64"])
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode frame image data")

    return Frame.from_array(
        img,
        frame_id=data["frame_id"],
        t_src_ns=data["t_src_ns"],
    )


def _deserialize_observation_in_worker(data: Dict[str, Any]) -> "Observation":
    """Deserialize an Observation from ZMQ deps message."""
    return deserialize_observation(data)


def run_worker(analyzer_name: str, ipc_address: str) -> int:
    """Run the worker process main loop.

    Args:
        analyzer_name: Name of the analyzer to load via entry points.
        ipc_address: ZMQ IPC address to bind to.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    import zmq
    from visualpath.plugin import create_analyzer
    from visualpath.core import Observation  # noqa: F401 - used in type hints

    # Create ZMQ socket (REP pattern - reply to requests)
    context = zmq.Context()
    socket = context.socket(zmq.REP)

    try:
        socket.bind(ipc_address)
        logger.info(f"Worker bound to {ipc_address}")
    except zmq.ZMQError as e:
        logger.error(f"Failed to bind to {ipc_address}: {e}")
        return 1

    # Load and initialize analyzer
    try:
        analyzer = create_analyzer(analyzer_name)
        analyzer.initialize()
        logger.info(f"Loaded analyzer: {analyzer_name}")
    except Exception as e:
        logger.error(f"Failed to load analyzer '{analyzer_name}': {e}")
        socket.close()
        context.term()
        return 1

    try:
        while True:
            try:
                # Receive message (blocking)
                message = socket.recv_json()
            except zmq.ZMQError as e:
                logger.error(f"ZMQ receive error: {e}")
                break

            msg_type = message.get("type")

            if msg_type == "ping":
                # Handshake
                socket.send_json({"type": "pong", "analyzer": analyzer_name})
                continue

            if msg_type == "shutdown":
                # Clean shutdown
                socket.send_json({"type": "ack"})
                logger.info("Received shutdown signal")
                break

            if msg_type == "analyze":
                # Process frame
                try:
                    frame = _deserialize_frame(message["frame"])

                    # Deserialize deps if present
                    analyzer_deps = None
                    if "deps" in message and message["deps"]:
                        analyzer_deps = {}
                        for name, obs_data in message["deps"].items():
                            if obs_data is not None:
                                analyzer_deps[name] = _deserialize_observation_in_worker(obs_data)

                    observation = analyzer.process(frame, analyzer_deps)

                    socket.send_json({
                        "observation": serialize_observation(observation),
                    })
                except Exception as e:
                    logger.error(f"Analysis error: {e}")
                    socket.send_json({
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })
                continue

            # Unknown message type
            logger.warning(f"Unknown message type: {msg_type}")
            socket.send_json({"error": f"Unknown message type: {msg_type}"})

    finally:
        # Cleanup
        try:
            analyzer.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        socket.close()
        context.term()
        logger.info("Worker shutdown complete")

    return 0


def main() -> int:
    """Main entry point for the worker subprocess."""
    parser = argparse.ArgumentParser(
        description="visualpath worker subprocess",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--analyzer",
        required=True,
        help="Name of the analyzer to load (via entry points)",
    )
    parser.add_argument(
        "--ipc-address",
        required=True,
        help="ZMQ IPC address to bind to (e.g., ipc:///tmp/worker.sock)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    return run_worker(args.analyzer, args.ipc_address)


if __name__ == "__main__":
    sys.exit(main())
