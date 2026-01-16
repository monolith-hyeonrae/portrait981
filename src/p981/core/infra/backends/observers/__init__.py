"""Observer 백엔드 구현을 모은다."""

from p981.core.infra.backends.observers.loguru import LoguruObserverBackend
from p981.core.infra.backends.observers.noop import NoopObserverBackend
from p981.core.infra.backends.observers.opencv import OpenCvObserverBackend
from p981.core.infra.backends.observers.pixeltable import PixeltableObserverBackend
from p981.core.infra.backends.observers.rerun import RerunObserverBackend
from p981.core.infra.backends.observers.multi import MultiObserverBackend

__all__ = [
    "LoguruObserverBackend",
    "NoopObserverBackend",
    "OpenCvObserverBackend",
    "PixeltableObserverBackend",
    "RerunObserverBackend",
    "MultiObserverBackend",
]
