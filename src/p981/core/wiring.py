"""코어 서비스와 옵저버 어댑터를 조립하는 composition root."""

from __future__ import annotations

from .domain.impl import (
    FFmpegMediaService,
    InMemoryAssetService,
    InMemoryMediaService,
    InMemoryMomentService,
    InMemoryStateService,
    InMemorySynthesisService,
)
from .executor import StageExecutor
from .ports import (
    AssetPorts,
    InMemoryAssetIndex,
    InMemoryBlobStore,
    InMemoryMetaStore,
    LoguruObservationPort,
    MediaPorts,
    MomentPorts,
    MultiObservationPort,
    NoopObservationPort,
    ObservationPort,
    PixeltableObservationPort,
    RerunObservationPort,
    StatePorts,
    SynthesisPorts,
)
from .stage import DiscoverStageDeps, SimpleDiscoverStage, SimpleSynthesizeStage, SynthesizeStageDeps


def build_executor(mode: str, observer: ObservationPort) -> StageExecutor:
    """포트 바인딩 후 도메인 서비스와 스테이지 실행기를 구성한다."""
    # 코어 스토어.
    blob_store = InMemoryBlobStore()
    meta_store = InMemoryMetaStore()
    asset_index = InMemoryAssetIndex()

    # 포트 바인딩.
    media_ports = MediaPorts(blob_store=blob_store, observer=observer)
    state_ports = StatePorts(meta_store=meta_store, observer=observer)
    moment_ports = MomentPorts(observer=observer)
    asset_ports = AssetPorts(meta_store=meta_store, asset_index=asset_index)
    synthesis_ports = SynthesisPorts(blob_store=blob_store)

    # 도메인 서비스.
    asset_service = InMemoryAssetService(ports=asset_ports)
    if mode == "stub":
        media_service = InMemoryMediaService(ports=media_ports)
    else:
        media_service = FFmpegMediaService(ports=media_ports)
    state_service = InMemoryStateService(ports=state_ports)
    moment_service = InMemoryMomentService(ports=moment_ports)
    synthesis_service = InMemorySynthesisService(ports=synthesis_ports)

    # 스테이지 조립.
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


def build_observer(selected: list[str], mode: str) -> ObservationPort:
    """CLI 선택값으로 옵저버 어댑터 체인을 구성한다."""
    if not selected:
        if mode == "debug":
            return LoguruObservationPort()
        return NoopObservationPort()

    factories = {
        "log": LoguruObservationPort,
        "frames": PixeltableObservationPort,
        "rerun": RerunObservationPort,
        "noop": NoopObservationPort,
    }
    ports = [factories.get(name, NoopObservationPort)() for name in selected]
    if len(ports) == 1:
        return ports[0]
    return MultiObservationPort(ports)
