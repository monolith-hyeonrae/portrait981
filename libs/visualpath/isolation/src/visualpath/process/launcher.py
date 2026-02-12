"""Worker launchers for different isolation levels.

This module provides worker launchers that execute analyzers with
different isolation strategies:

- InlineWorker: Same process, same thread (IsolationLevel.INLINE)
- ThreadWorker: Same process, different thread (IsolationLevel.THREAD)
- ProcessWorker: Same venv, different process (IsolationLevel.PROCESS)
- VenvWorker: Different venv, different process (IsolationLevel.VENV)

Example:
    >>> from visualpath.process.launcher import WorkerLauncher
    >>> from visualpath.core import IsolationLevel
    >>>
    >>> # Create launcher based on isolation level
    >>> launcher = WorkerLauncher.create(
    ...     level=IsolationLevel.THREAD,
    ...     analyzer=my_analyzer,
    ... )
    >>> launcher.start()
    >>> result = launcher.process(frame)
    >>> launcher.stop()

For VenvWorker/ProcessWorker with ZMQ:
    >>> # Run analyzer in a separate venv
    >>> worker = WorkerLauncher.create(
    ...     level=IsolationLevel.VENV,
    ...     analyzer=None,  # Will be loaded in subprocess
    ...     venv_path="/path/to/venv",
    ...     analyzer_name="face",  # Entry point name
    ... )
    >>> worker.start()
    >>> result = worker.process(frame)
    >>> worker.stop()
"""

import json
import logging
import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

from visualbase.ipc import check_zmq_available, generate_ipc_address, encode_frame
from visualpath.core.observation import Observation
from visualpath.core.module import Module
from visualpath.core.isolation import IsolationLevel
from visualpath.process.serialization import (
    serialize_observation,
    deserialize_observation,
)

if TYPE_CHECKING:
    from visualbase import Frame

logger = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    """Public worker status information for observability.

    Attributes:
        isolation_level: Isolation level name ("INLINE", "THREAD", "PROCESS", "VENV").
        pid: Worker process PID (0 = not yet started).
        venv_path: Virtual environment path ("" = same venv).
    """
    isolation_level: str
    pid: int
    venv_path: str


@dataclass
class WorkerResult:
    """Result from a worker processing a frame.

    Attributes:
        observation: The extracted observation, or None if extraction failed.
        error: Optional error message if extraction failed.
        timing_ms: Processing time in milliseconds.
    """
    observation: Optional[Observation]
    error: Optional[str] = None
    timing_ms: float = 0.0


class BaseWorker(ABC):
    """Abstract base class for workers.

    Workers handle frame processing with a specific isolation strategy.
    """

    @abstractmethod
    def start(self) -> None:
        """Start the worker."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the worker and clean up resources."""
        ...

    @abstractmethod
    def process(self, frame: "Frame", deps: Optional[Dict[str, Observation]] = None) -> WorkerResult:
        """Process a frame and return the result.

        Args:
            frame: The frame to process.
            deps: Optional dict of observations from dependent analyzers.

        Returns:
            WorkerResult with observation or error.
        """
        ...

    def process_batch(
        self,
        frames: list["Frame"],
        deps_list: list[Optional[Dict[str, Observation]]],
    ) -> list[WorkerResult]:
        """Process a batch of frames and return results.

        Default implementation calls process() sequentially.
        Override for IPC-level batch optimization.

        Args:
            frames: List of frames to process.
            deps_list: Corresponding deps for each frame.

        Returns:
            List of WorkerResult, same length as frames.
        """
        return [
            self.process(frame, deps)
            for frame, deps in zip(frames, deps_list)
        ]

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the worker is running."""
        ...

    @property
    @abstractmethod
    def worker_info(self) -> WorkerInfo:
        """Return public worker status information for observability."""
        ...


class InlineWorker(BaseWorker):
    """Worker that runs extraction inline (same process, same thread).

    This is the simplest worker with zero overhead, but no isolation.
    Suitable for lightweight analyzers that don't need isolation.
    """

    def __init__(self, analyzer: Module):
        """Initialize the inline worker.

        Args:
            analyzer: The analyzer to run.
        """
        self._analyzer = analyzer
        self._running = False

    def start(self) -> None:
        """Start the worker (initializes analyzer)."""
        self._analyzer.initialize()
        self._running = True

    def stop(self) -> None:
        """Stop the worker (cleans up analyzer)."""
        self._analyzer.cleanup()
        self._running = False

    def process(self, frame: "Frame", deps: Optional[Dict[str, Observation]] = None) -> WorkerResult:
        """Process a frame inline."""
        import time
        start = time.perf_counter()

        try:
            analyzer_deps = None
            if deps and self._analyzer.depends:
                analyzer_deps = {
                    name: deps[name]
                    for name in self._analyzer.depends
                    if name in deps
                }
            obs = self._analyzer.process(frame, analyzer_deps)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return WorkerResult(observation=obs, timing_ms=elapsed_ms)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return WorkerResult(observation=None, error=str(e), timing_ms=elapsed_ms)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def worker_info(self) -> WorkerInfo:
        return WorkerInfo(isolation_level="INLINE", pid=os.getpid(), venv_path="")


class ThreadWorker(BaseWorker):
    """Worker that runs extraction in a separate thread.

    Provides thread-level isolation. Useful for I/O-bound work or
    analyzers that can benefit from concurrent execution.
    """

    def __init__(
        self,
        analyzer: Module,
        queue_size: int = 10,
    ):
        """Initialize the thread worker.

        Args:
            analyzer: The analyzer to run.
            queue_size: Maximum pending frames in queue.
        """
        self._analyzer = analyzer
        self._queue_size = queue_size

        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = False

    def start(self) -> None:
        """Start the worker thread pool."""
        self._analyzer.initialize()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._running = True

    def stop(self) -> None:
        """Stop the worker and clean up."""
        self._running = False
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._analyzer.cleanup()

    def process(self, frame: "Frame", deps: Optional[Dict[str, Observation]] = None) -> WorkerResult:
        """Submit frame for processing and wait for result."""
        import time

        if not self._running or self._executor is None:
            return WorkerResult(observation=None, error="Worker not running")

        start = time.perf_counter()

        try:
            analyzer_deps = None
            if deps and self._analyzer.depends:
                analyzer_deps = {
                    name: deps[name]
                    for name in self._analyzer.depends
                    if name in deps
                }

            def _do_analyze():
                return self._analyzer.process(frame, analyzer_deps)

            future = self._executor.submit(_do_analyze)
            obs = future.result()  # Blocking wait
            elapsed_ms = (time.perf_counter() - start) * 1000
            return WorkerResult(observation=obs, timing_ms=elapsed_ms)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return WorkerResult(observation=None, error=str(e), timing_ms=elapsed_ms)

    def process_async(self, frame: "Frame") -> Future:
        """Submit frame for processing without waiting.

        Args:
            frame: The frame to process.

        Returns:
            Future that will contain the Observation.
        """
        if not self._running or self._executor is None:
            raise RuntimeError("Worker not running")
        return self._executor.submit(self._analyzer.process, frame)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def worker_info(self) -> WorkerInfo:
        return WorkerInfo(isolation_level="THREAD", pid=os.getpid(), venv_path="")




class VenvWorker(BaseWorker):
    """Worker that runs extraction in a separate venv subprocess.

    Uses ZMQ for bidirectional communication:
    - Main -> Subprocess: Frame (REQ)
    - Subprocess -> Main: Observation (REP)

    This provides full dependency isolation by running the analyzer
    in a different Python virtual environment with its own set of
    installed packages.

    Requirements:
    - pyzmq must be installed in both the main process and the target venv
    - The target venv must have visualpath and the analyzer's dependencies

    Example:
        >>> worker = VenvWorker(
        ...     analyzer=None,  # Will be loaded in subprocess
        ...     venv_path="/path/to/venv-face",
        ...     analyzer_name="face",
        ... )
        >>> worker.start()
        >>> result = worker.process(frame)
        >>> worker.stop()
    """

    def __init__(
        self,
        analyzer: Optional[Module],
        venv_path: str,
        analyzer_name: Optional[str] = None,
        queue_size: int = 10,
        timeout_sec: float = 30.0,
        handshake_timeout_sec: float = 30.0,
        jpeg_quality: int = 95,
    ):
        """Initialize the venv worker.

        Args:
            analyzer: Optional analyzer instance (for backwards compatibility).
                       If provided and analyzer_name is None, falls back to inline.
            venv_path: Path to the virtual environment.
            analyzer_name: Entry point name of the analyzer to load in subprocess.
            queue_size: Maximum pending frames in queue (reserved for future use).
            timeout_sec: Timeout for processing requests in seconds.
            handshake_timeout_sec: Timeout for initial handshake in seconds.
            jpeg_quality: JPEG quality for frame compression (0-100).
        """
        self._analyzer = analyzer
        self._venv_path = venv_path
        self._analyzer_name = analyzer_name or (analyzer.name if analyzer else None)
        self._queue_size = queue_size
        self._timeout_sec = timeout_sec
        self._handshake_timeout_sec = handshake_timeout_sec
        self._jpeg_quality = jpeg_quality

        self._process: Optional[subprocess.Popen] = None
        self._rpc_client: Optional[Any] = None  # ZMQRPCClient
        self._ipc_address: str = ""
        self._ipc_file: Optional[str] = None
        self._running = False
        self._use_zmq = False

        # Fall back to inline if ZMQ not available or no analyzer_name
        self._inline: Optional[InlineWorker] = None

    def _should_use_zmq(self) -> bool:
        """Determine if ZMQ should be used."""
        if not check_zmq_available():
            logger.warning("pyzmq not available, falling back to inline execution")
            return False

        if not self._analyzer_name:
            logger.warning("No analyzer_name provided, falling back to inline execution")
            return False

        # Check if venv Python exists
        venv_python = os.path.join(self._venv_path, "bin", "python")
        if not os.path.isfile(venv_python):
            logger.warning(
                f"Venv Python not found at {venv_python}, falling back to inline execution"
            )
            return False

        return True

    def start(self) -> None:
        """Start the worker subprocess and establish ZMQ connection."""
        if self._running:
            return

        self._use_zmq = self._should_use_zmq()

        if not self._use_zmq:
            # Fall back to inline execution
            if self._analyzer is None:
                raise ValueError(
                    "Cannot fall back to inline execution without an analyzer instance"
                )
            self._inline = InlineWorker(self._analyzer)
            self._inline.start()
            self._running = True
            return

        from visualbase.ipc import ZMQRPCClient

        # Generate unique IPC address
        self._ipc_address, self._ipc_file = generate_ipc_address(
            prefix="visualpath-worker"
        )

        # Start subprocess
        venv_python = os.path.join(self._venv_path, "bin", "python")
        cmd = [
            venv_python,
            "-m", "visualpath.process.worker",
            "--analyzer", self._analyzer_name,
            "--ipc-address", self._ipc_address,
        ]

        logger.info(f"Starting worker subprocess: {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            self._cleanup_zmq()
            raise RuntimeError(f"Failed to start worker subprocess: {e}") from e

        # Give subprocess time to start and bind
        time.sleep(0.1)

        # Create RPC client and connect
        self._rpc_client = ZMQRPCClient(
            send_timeout_ms=int(self._timeout_sec * 1000),
            recv_timeout_ms=int(self._timeout_sec * 1000),
        )
        try:
            self._rpc_client.connect(self._ipc_address)
        except Exception as e:
            self._terminate_process()
            self._cleanup_zmq()
            raise RuntimeError(f"Failed to connect to worker: {e}") from e

        # Perform handshake with timeout
        try:
            self._rpc_client.send(json.dumps({"type": "ping"}).encode())
            raw = self._rpc_client.recv(
                timeout_ms=int(self._handshake_timeout_sec * 1000)
            )

            if raw is None:
                raise RuntimeError(
                    f"Worker handshake timed out after {self._handshake_timeout_sec}s"
                )

            response = json.loads(raw)

            if response.get("type") != "pong":
                raise RuntimeError(f"Unexpected handshake response: {response}")

            logger.info(f"Worker handshake successful, analyzer: {response.get('analyzer')}")

        except RuntimeError:
            self._terminate_process()
            self._cleanup_zmq()
            raise
        except Exception as e:
            self._terminate_process()
            self._cleanup_zmq()
            raise RuntimeError(f"Worker handshake failed: {e}") from e

        self._running = True

    def stop(self) -> None:
        """Stop the worker subprocess and clean up resources."""
        if not self._running:
            return

        if self._inline is not None:
            self._inline.stop()
            self._inline = None
            self._running = False
            return

        # Send shutdown signal
        if self._rpc_client is not None and self._rpc_client.is_connected:
            try:
                self._rpc_client.send(json.dumps({"type": "shutdown"}).encode())
                self._rpc_client.recv(timeout_ms=5000)
            except Exception as e:
                logger.warning(f"Error during shutdown signal: {e}")

        self._terminate_process()
        self._cleanup_zmq()
        self._running = False

    def _terminate_process(self) -> None:
        """Terminate the subprocess."""
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("Worker process did not terminate, killing")
                self._process.kill()
                self._process.wait()
            except Exception as e:
                logger.warning(f"Error terminating worker process: {e}")
            finally:
                self._process = None

    def _cleanup_zmq(self) -> None:
        """Clean up RPC client and IPC resources."""
        if self._rpc_client is not None:
            try:
                self._rpc_client.close()
            except Exception:
                pass
            self._rpc_client = None

        # Remove IPC socket file
        if self._ipc_file and os.path.exists(self._ipc_file):
            try:
                os.unlink(self._ipc_file)
            except Exception:
                pass
            self._ipc_file = None

    def process(self, frame: "Frame", deps: Optional[Dict[str, Observation]] = None) -> WorkerResult:
        """Send frame to subprocess and receive observation.

        Args:
            frame: Frame to process.
            deps: Optional dict of observations from dependent analyzers.

        Returns:
            WorkerResult with observation or error.
        """
        start_time = time.perf_counter()

        if not self._running:
            return WorkerResult(
                observation=None,
                error="Worker not running",
                timing_ms=0.0,
            )

        if self._inline is not None:
            return self._inline.process(frame, deps)

        try:
            # Serialize and send frame
            frame_data = encode_frame(frame, self._jpeg_quality)
            message: Dict[str, Any] = {
                "type": "analyze",
                "frame": frame_data,
            }
            # Include deps if provided
            if deps:
                message["deps"] = {
                    name: serialize_observation(obs)
                    for name, obs in deps.items()
                }
            self._rpc_client.send(json.dumps(message).encode())

            # Receive response
            raw = self._rpc_client.recv()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if raw is None:
                return WorkerResult(
                    observation=None,
                    error=f"Worker timeout after {self._timeout_sec}s",
                    timing_ms=elapsed_ms,
                )

            response = json.loads(raw)

            if "error" in response:
                return WorkerResult(
                    observation=None,
                    error=response["error"],
                    timing_ms=elapsed_ms,
                )

            observation = deserialize_observation(response.get("observation"))
            return WorkerResult(
                observation=observation,
                timing_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return WorkerResult(
                observation=None,
                error=str(e),
                timing_ms=elapsed_ms,
            )

    def process_batch(
        self,
        frames: list["Frame"],
        deps_list: list[Optional[Dict[str, Observation]]],
    ) -> list[WorkerResult]:
        """Send a batch of frames to subprocess and receive observations.

        Uses a single IPC round-trip with the ``analyze_batch`` message type
        for reduced overhead. Falls back to sequential if inline.

        Args:
            frames: List of frames to process.
            deps_list: Corresponding deps for each frame.

        Returns:
            List of WorkerResult, same length as frames.
        """
        start_time = time.perf_counter()

        if not self._running:
            return [
                WorkerResult(observation=None, error="Worker not running", timing_ms=0.0)
                for _ in frames
            ]

        if self._inline is not None:
            return [
                self._inline.process(f, d) for f, d in zip(frames, deps_list)
            ]

        try:
            # Serialize batch message
            serialized_frames = [encode_frame(f, self._jpeg_quality) for f in frames]
            serialized_deps = []
            for deps in deps_list:
                if deps:
                    serialized_deps.append({
                        name: serialize_observation(obs)
                        for name, obs in deps.items()
                    })
                else:
                    serialized_deps.append(None)

            message: Dict[str, Any] = {
                "type": "analyze_batch",
                "frames": serialized_frames,
                "deps_list": serialized_deps,
            }
            self._rpc_client.send(json.dumps(message).encode())

            # Receive batch response
            raw = self._rpc_client.recv()
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if raw is None:
                return [
                    WorkerResult(
                        observation=None,
                        error=f"Worker timeout after {self._timeout_sec}s",
                        timing_ms=elapsed_ms,
                    )
                    for _ in frames
                ]

            response = json.loads(raw)

            if "error" in response:
                return [
                    WorkerResult(
                        observation=None,
                        error=response["error"],
                        timing_ms=elapsed_ms,
                    )
                    for _ in frames
                ]

            observations_data = response.get("observations", [])
            results = []
            per_frame_ms = elapsed_ms / max(len(frames), 1)
            for obs_data in observations_data:
                obs = deserialize_observation(obs_data)
                results.append(WorkerResult(observation=obs, timing_ms=per_frame_ms))
            return results

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return [
                WorkerResult(observation=None, error=str(e), timing_ms=elapsed_ms)
                for _ in frames
            ]

    @property
    def is_running(self) -> bool:
        """Check if the worker is running."""
        return self._running

    @property
    def worker_info(self) -> WorkerInfo:
        pid = self._process.pid if self._process else 0
        return WorkerInfo(isolation_level="VENV", pid=pid, venv_path=self._venv_path)

    def get_subprocess_output(self) -> tuple[str, str]:
        """Get stdout and stderr from the subprocess.

        Useful for debugging when the worker fails.

        Returns:
            Tuple of (stdout, stderr) strings.
        """
        if self._process is None:
            return "", ""

        stdout = ""
        stderr = ""

        if self._process.stdout:
            try:
                stdout = self._process.stdout.read()
            except Exception:
                pass

        if self._process.stderr:
            try:
                stderr = self._process.stderr.read()
            except Exception:
                pass

        return stdout or "", stderr or ""


class ProcessWorker(BaseWorker):
    """Worker that runs extraction in a separate process (same venv).

    This is a convenience wrapper around VenvWorker that uses the current
    Python environment. Provides process-level isolation without dependency
    isolation.

    Requirements:
    - pyzmq must be installed

    Example:
        >>> worker = ProcessWorker(
        ...     analyzer=None,
        ...     analyzer_name="face",
        ... )
        >>> worker.start()
        >>> result = worker.process(frame)
        >>> worker.stop()
    """

    def __init__(
        self,
        analyzer: Optional[Module] = None,
        analyzer_name: Optional[str] = None,
        queue_size: int = 10,
        timeout_sec: float = 30.0,
    ):
        """Initialize the process worker.

        Args:
            analyzer: Optional analyzer instance (for backwards compatibility).
            analyzer_name: Entry point name of the analyzer.
            queue_size: Maximum pending frames in queue.
            timeout_sec: Timeout for processing requests in seconds.
        """
        # Get the current venv path
        venv_path = os.path.dirname(os.path.dirname(sys.executable))

        self._delegate = VenvWorker(
            analyzer=analyzer,
            venv_path=venv_path,
            analyzer_name=analyzer_name,
            queue_size=queue_size,
            timeout_sec=timeout_sec,
        )

    def start(self) -> None:
        """Start the worker process."""
        self._delegate.start()

    def stop(self) -> None:
        """Stop the worker process."""
        self._delegate.stop()

    def process(self, frame: "Frame", deps: Optional[Dict[str, Observation]] = None) -> WorkerResult:
        """Process a frame in the worker process."""
        return self._delegate.process(frame, deps)

    def process_batch(
        self,
        frames: list["Frame"],
        deps_list: list[Optional[Dict[str, Observation]]],
    ) -> list[WorkerResult]:
        """Process a batch of frames in the worker process."""
        return self._delegate.process_batch(frames, deps_list)

    @property
    def is_running(self) -> bool:
        """Check if the worker is running."""
        return self._delegate.is_running

    @property
    def worker_info(self) -> WorkerInfo:
        info = self._delegate.worker_info
        return WorkerInfo(isolation_level="PROCESS", pid=info.pid, venv_path=info.venv_path)


class WorkerLauncher:
    """Factory for creating workers based on isolation level.

    Example:
        >>> # Simple usage with analyzer instance
        >>> launcher = WorkerLauncher.create(
        ...     level=IsolationLevel.THREAD,
        ...     analyzer=my_analyzer,
        ... )
        >>> with launcher:
        ...     result = launcher.process(frame)

        >>> # VenvWorker with subprocess
        >>> launcher = WorkerLauncher.create(
        ...     level=IsolationLevel.VENV,
        ...     analyzer=None,
        ...     venv_path="/path/to/venv",
        ...     analyzer_name="face",
        ... )
    """

    @staticmethod
    def create(
        level: IsolationLevel,
        analyzer: Optional[Module],
        venv_path: Optional[str] = None,
        analyzer_name: Optional[str] = None,
        **kwargs,
    ) -> BaseWorker:
        """Create a worker for the specified isolation level.

        Args:
            level: The isolation level to use.
            analyzer: The analyzer to run. Can be None for PROCESS/VENV levels
                      if analyzer_name is provided (loaded via entry points).
            venv_path: Path to venv (required for VENV level).
            analyzer_name: Entry point name of the analyzer. Required for
                           PROCESS/VENV levels when analyzer is None.
            **kwargs: Additional arguments passed to the worker.

        Returns:
            A worker instance for the specified isolation level.

        Raises:
            ValueError: If required parameters are missing.
        """
        if level == IsolationLevel.INLINE:
            if analyzer is None:
                raise ValueError("analyzer is required for INLINE isolation level")
            return InlineWorker(analyzer)

        elif level == IsolationLevel.THREAD:
            if analyzer is None:
                raise ValueError("analyzer is required for THREAD isolation level")
            return ThreadWorker(analyzer, **kwargs)

        elif level == IsolationLevel.PROCESS:
            return ProcessWorker(
                analyzer=analyzer,
                analyzer_name=analyzer_name,
                **kwargs,
            )

        elif level == IsolationLevel.VENV:
            if venv_path is None:
                raise ValueError("venv_path is required for VENV isolation level")
            return VenvWorker(
                analyzer=analyzer,
                venv_path=venv_path,
                analyzer_name=analyzer_name,
                **kwargs,
            )

        elif level == IsolationLevel.CONTAINER:
            # Container isolation not yet implemented
            raise NotImplementedError("CONTAINER isolation level not yet implemented")

        else:
            raise ValueError(f"Unknown isolation level: {level}")
