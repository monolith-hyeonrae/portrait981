"""코어 실행을 감싸는 공용 진입점.

CLI/API/GUI/Runtime 등 다양한 실행 주체가 동일한 코어 조립 로직을 재사용하도록 한다.
"""

from __future__ import annotations

from p981.core.application.executor import StageExecutor
from p981.core.application.protocols import ObserverProtocol
from p981.core.application.stage.discover import DiscoverInput, DiscoverOutput
from p981.core.application.stage.synthesize import SynthesizeInput, SynthesizeOutput
from p981.core.wiring import build_executor, build_observer


class CoreApp:
    """코어 실행 흐름을 하나로 묶는 공용 진입점."""

    def __init__(self, mode: str, observer: ObserverProtocol, executor: StageExecutor) -> None:
        self._mode = mode
        self._observer = observer
        self._executor = executor

    @classmethod
    def from_wiring(cls, mode: str, observers: list[str]) -> "CoreApp":
        """wiring 기반으로 observer/executor를 조립해 CoreApp을 만든다."""
        observer = build_observer(observers, mode)
        executor = build_executor(mode, observer)
        return cls(mode=mode, observer=observer, executor=executor)

    @property
    def mode(self) -> str:
        """현재 실행 모드를 반환한다."""
        return self._mode

    @property
    def observer(self) -> ObserverProtocol:
        """코어에서 사용하는 observer를 반환한다."""
        return self._observer

    def run_discover(
        self,
        request: DiscoverInput,
        progress=None,
    ) -> DiscoverOutput:
        """Discover 스테이지를 실행한다."""
        return self._executor.run_discover(request, progress=progress)

    def run_synthesize(
        self,
        request: SynthesizeInput,
        progress=None,
    ) -> SynthesizeOutput:
        """Synthesize 스테이지를 실행한다."""
        return self._executor.run_synthesize(request, progress=progress)
