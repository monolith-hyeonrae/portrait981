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
import json
import logging
import sys
import traceback
from typing import Any, Dict, Optional

from visualbase.ipc import decode_frame
from visualpath.process.serialization import (
    serialize_observation,
    deserialize_observation,
)

logger = logging.getLogger(__name__)


def run_worker(analyzer_name: str, ipc_address: str) -> int:
    """Run the worker process main loop.

    Args:
        analyzer_name: Name of the analyzer to load via entry points.
        ipc_address: ZMQ IPC address to bind to.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    from visualbase.ipc import ZMQRPCServer
    from visualpath.plugin import create_analyzer

    # Create RPC server
    server = ZMQRPCServer()

    try:
        server.bind(ipc_address)
        logger.info(f"Worker bound to {ipc_address}")
    except Exception as e:
        logger.error(f"Failed to bind to {ipc_address}: {e}")
        return 1

    # Load and initialize analyzer
    try:
        analyzer = create_analyzer(analyzer_name)
        analyzer.initialize()
        logger.info(f"Loaded analyzer: {analyzer_name}")
    except Exception as e:
        logger.error(f"Failed to load analyzer '{analyzer_name}': {e}")
        server.close()
        return 1

    try:
        while True:
            try:
                raw = server.recv()
                if raw is None:
                    continue
                message = json.loads(raw)
            except Exception as e:
                logger.error(f"Receive error: {e}")
                break

            msg_type = message.get("type")

            if msg_type == "ping":
                # Handshake
                server.send(json.dumps({"type": "pong", "analyzer": analyzer_name}).encode())
                continue

            if msg_type == "shutdown":
                # Clean shutdown
                server.send(json.dumps({"type": "ack"}).encode())
                logger.info("Received shutdown signal")
                break

            if msg_type == "analyze":
                # Process frame
                try:
                    frame = decode_frame(message["frame"])

                    # Deserialize deps if present
                    analyzer_deps = None
                    if "deps" in message and message["deps"]:
                        analyzer_deps = {}
                        for name, obs_data in message["deps"].items():
                            if obs_data is not None:
                                analyzer_deps[name] = deserialize_observation(obs_data)

                    observation = analyzer.process(frame, analyzer_deps)

                    server.send(json.dumps({
                        "observation": serialize_observation(observation),
                    }).encode())
                except Exception as e:
                    logger.error(f"Analysis error: {e}")
                    server.send(json.dumps({
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }).encode())
                continue

            if msg_type == "analyze_batch":
                # Process batch of frames
                try:
                    frames = [decode_frame(f) for f in message["frames"]]
                    deps_list = []
                    for deps_data in message.get("deps_list", []):
                        if deps_data:
                            deps = {}
                            for name, obs_data in deps_data.items():
                                if obs_data is not None:
                                    deps[name] = deserialize_observation(obs_data)
                            deps_list.append(deps)
                        else:
                            deps_list.append(None)

                    # Pad deps_list if shorter than frames
                    while len(deps_list) < len(frames):
                        deps_list.append(None)

                    observations = analyzer.process_batch(frames, deps_list)

                    server.send(json.dumps({
                        "observations": [
                            serialize_observation(obs) for obs in observations
                        ],
                    }).encode())
                except Exception as e:
                    logger.error(f"Batch analysis error: {e}")
                    server.send(json.dumps({
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }).encode())
                continue

            # Unknown message type
            logger.warning(f"Unknown message type: {msg_type}")
            server.send(json.dumps({"error": f"Unknown message type: {msg_type}"}).encode())

    finally:
        # Cleanup
        try:
            analyzer.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        server.close()
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
