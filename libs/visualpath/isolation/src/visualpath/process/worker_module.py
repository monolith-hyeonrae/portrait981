"""WorkerModule — Module wrapper that delegates to a BaseWorker.

WorkerModule implements the Module interface so that isolated modules
(running in subprocesses or separate venvs) are transparent to the
SimpleInterpreter and GraphExecutor. They just call module.process()
without knowing about isolation.

Example:
    >>> from visualpath.process.worker_module import WorkerModule
    >>> from visualpath.process.launcher import WorkerLauncher
    >>> from visualpath.core import IsolationLevel
    >>>
    >>> worker = WorkerLauncher.create(
    ...     level=IsolationLevel.PROCESS,
    ...     analyzer=None,
    ...     analyzer_name="pose",
    ... )
    >>> module = WorkerModule(name="pose", worker=worker, depends=[])
    >>> module.initialize()  # starts the worker
    >>> obs = module.process(frame, deps={"face_detect": face_obs})
    >>> module.cleanup()     # stops the worker
"""

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from visualpath.core.module import Module, DepsContext

if TYPE_CHECKING:
    from visualbase import Frame
    from visualpath.core.capabilities import ModuleCapabilities
    from visualpath.core.observation import Observation
    from visualpath.process.launcher import BaseWorker

logger = logging.getLogger(__name__)


class WorkerModule(Module):
    """Module that delegates processing to a BaseWorker.

    This allows isolated workers (ProcessWorker, VenvWorker) to be used
    transparently within FlowGraph pipelines. The SimpleInterpreter calls
    module.process() as usual, and WorkerModule forwards it to the worker.

    Args:
        name: Module name (must match the analyzer's name).
        worker: BaseWorker instance to delegate to.
        depends: List of module names this module depends on.
    """

    def __init__(
        self,
        name: str,
        worker: "BaseWorker",
        depends: Optional[List[str]] = None,
        *,
        optional_depends: Optional[List[str]] = None,
        stateful: bool = False,
        capabilities: Optional["ModuleCapabilities"] = None,
    ):
        self._name = name
        self._worker = worker
        self.depends = depends or []
        self.optional_depends = optional_depends or []
        self.stateful = stateful
        self._capabilities = capabilities

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> "ModuleCapabilities":
        if self._capabilities is not None:
            return self._capabilities
        from visualpath.core.capabilities import ModuleCapabilities
        return ModuleCapabilities()

    def initialize(self) -> None:
        """Start the worker and emit observability record."""
        if not self._worker.is_running:
            self._worker.start()
        # Emit observability record
        try:
            from visualpath.observability import ObservabilityHub
            hub = ObservabilityHub.get_instance()
            if hub.enabled:
                from visualpath.observability.records import WorkerStartRecord
                info = self._worker.worker_info
                hub.emit(WorkerStartRecord(
                    module_name=self._name,
                    isolation_level=info.isolation_level,
                    pid=info.pid,
                    venv_path=info.venv_path,
                ))
        except Exception:
            pass  # observability failure should not block execution

    def cleanup(self) -> None:
        """Stop the worker."""
        if self._worker.is_running:
            self._worker.stop()

    def process(
        self,
        frame: "Frame",
        deps: Optional[Dict[str, "Observation"]] = None,
    ) -> Optional["Observation"]:
        """Process a frame via the worker.

        Args:
            frame: The frame to process.
            deps: Optional dependency observations.

        Returns:
            Observation from the worker, or None on error.
        """
        result = self._worker.process(frame, deps=deps)

        if result.error:
            logger.warning(
                "WorkerModule '%s' error: %s", self._name, result.error
            )
            return None

        return result.observation

    def process_batch(
        self,
        frames: list["Frame"],
        deps_list: list[DepsContext],
    ) -> list[Optional["Observation"]]:
        """Process a batch of frames via the worker.

        Delegates to worker.process_batch() for IPC-level batching.

        Args:
            frames: List of frames to process.
            deps_list: Corresponding dependency observations.

        Returns:
            List of Observations, same length as frames.
        """
        batch_results = self._worker.process_batch(frames, deps_list)

        observations: list[Optional["Observation"]] = []
        for result in batch_results:
            if result.error:
                logger.warning(
                    "WorkerModule '%s' batch error: %s", self._name, result.error
                )
                observations.append(None)
            else:
                observations.append(result.observation)

        return observations

    @property
    def runtime_info(self):
        from visualpath.core.module import RuntimeInfo

        info = self._worker.worker_info
        return RuntimeInfo(isolation=info.isolation_level, pid=info.pid)

    @property
    def worker(self) -> "BaseWorker":
        """Access the underlying worker."""
        return self._worker


__all__ = ["WorkerModule"]
