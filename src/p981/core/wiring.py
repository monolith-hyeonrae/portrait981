"""코어 서비스와 옵저버 백엔드를 조립하는 composition root."""

from __future__ import annotations

from p981.core.infra.services import (
    FFmpegMediaService,
    InMemoryAssetService,
    InMemoryMediaService,
    InMemoryMomentService,
    InMemoryStateService,
    InMemorySynthesisService,
)
from p981.core.application.executor import StageExecutor
from p981.core.infra.backends import (
    InMemoryAssetIndex,
    InMemoryBlobStore,
    InMemoryMetaStore,
    LoguruObserverBackend,
    MultiObserverBackend,
    NoopObserverBackend,
    OpenCvObserverBackend,
    PixeltableObserverBackend,
    RerunObserverBackend,
)
from p981.core.application.protocols import (
    AssetProtocols,
    MediaProtocols,
    MomentProtocols,
    ObserverProtocol,
    StateProtocols,
    SynthesisProtocols,
)
from p981.core.application.stage import (
    DiscoverStageDeps,
    SimpleDiscoverStage,
    SimpleSynthesizeStage,
    SynthesizeStageDeps,
)


def build_executor(mode: str, observer: ObserverProtocol) -> StageExecutor:
    """프로토콜 바인딩 후 도메인 서비스와 스테이지 실행기를 구성한다."""

    # 코어 스토어: 메타/블랍/인덱스 등 인프라 프로토콜 구현체(스켈레톤은 인메모리).
    blob_store = InMemoryBlobStore()
    meta_store = InMemoryMetaStore()
    asset_index = InMemoryAssetIndex()

    # 프로토콜 바인딩: 애플리케이션 서비스가 사용할 프로토콜 의존성 묶음을 구성한다.
    media_protocols = MediaProtocols(blob_store=blob_store, observer=observer)
    state_protocols = StateProtocols(meta_store=meta_store, observer=observer)
    moment_protocols = MomentProtocols(observer=observer)
    asset_protocols = AssetProtocols(meta_store=meta_store, asset_index=asset_index)
    synthesis_protocols = SynthesisProtocols(blob_store=blob_store)

    # 애플리케이션 서비스: 프로토콜 바인딩을 주입해 실제 구현체를 준비한다.
    asset_service = InMemoryAssetService(protocols=asset_protocols)
    if mode == "stub":
        media_service = InMemoryMediaService(protocols=media_protocols)
    else:
        media_service = FFmpegMediaService(protocols=media_protocols)
    state_service = InMemoryStateService(protocols=state_protocols)
    moment_service = InMemoryMomentService(protocols=moment_protocols)
    synthesis_service = InMemorySynthesisService(protocols=synthesis_protocols)

    # 스테이지 조립: 애플리케이션 서비스를 워크플로 단위로 묶어 StageExecutor에 제공한다.
    discover_stage = SimpleDiscoverStage(
        DiscoverStageDeps(
            media=media_service,
            state=state_service,
            moment=moment_service,
            asset=asset_service,
        )
    )
    synthesize_stage = SimpleSynthesizeStage(
        SynthesizeStageDeps(asset=asset_service, synthesis=synthesis_service)
    )
    return StageExecutor(discover_stage=discover_stage, synthesize_stage=synthesize_stage)


def build_observer(selected: list[str], mode: str) -> ObserverProtocol:
    """CLI 선택값으로 옵저버 백엔드 체인을 구성한다."""
    log_level = "DEBUG" if mode == "debug" else "INFO"
    if not selected:
        if mode == "debug":
            return LoguruObserverBackend(level=log_level)
        return NoopObserverBackend()

    factories = {
        "log": lambda: LoguruObserverBackend(level=log_level),
        "frames": PixeltableObserverBackend,
        "pixeltable": PixeltableObserverBackend,
        "opencv": OpenCvObserverBackend,
        "rerun": RerunObserverBackend,
        "noop": NoopObserverBackend,
    }
    backends = [factories.get(name, NoopObserverBackend)() for name in selected]
    if len(backends) == 1:
        return backends[0]
    return MultiObserverBackend(backends)
