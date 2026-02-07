"""WorkerBackend — isolation-aware execution backend.

WorkerBackend wraps modules that need isolation (PROCESS, VENV, etc.)
in WorkerModule, then delegates execution to SimpleBackend.

This means all execution goes through FlowGraph → Backend.execute(),
regardless of whether modules need isolation or not.

Example:
    >>> from visualpath.backends.worker import WorkerBackend
    >>> backend = WorkerBackend()
    >>> result = backend.execute(frames, graph)
"""

from visualpath.backends.worker.backend import WorkerBackend

__all__ = ["WorkerBackend"]
