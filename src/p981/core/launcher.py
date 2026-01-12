from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from loguru import logger

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
    InMemoryAssetIndex,
    InMemoryBlobStore,
    InMemoryMetaStore,
    LoguruObservationPort,
    MultiObservationPort,
    NoopObservationPort,
    ObservationPort,
    PixeltableObservationPort,
    RerunObservationPort,
)
from .stage import DiscoverStageDeps, SimpleDiscoverStage, SimpleSynthesizeStage, SynthesizeStageDeps
from .common import LoguruProgressSink, configure_logging
from .types import DiscoverInput, SynthesizeInput


def build_executor(mode: str, observer: ObservationPort) -> StageExecutor:
    blob_store = InMemoryBlobStore()
    meta_store = InMemoryMetaStore()
    asset_index = InMemoryAssetIndex()

    asset_service = InMemoryAssetService(meta_store=meta_store, asset_index=asset_index)
    if mode == "stub":
        media_service = InMemoryMediaService(blob_store=blob_store, observer=observer)
    else:
        media_service = FFmpegMediaService(blob_store=blob_store, observer=observer)
    state_service = InMemoryStateService(meta_store=meta_store)
    moment_service = InMemoryMomentService()
    synthesis_service = InMemorySynthesisService(blob_store=blob_store)

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


def main() -> None:
    parser = argparse.ArgumentParser(prog="p981.core.launcher")
    parser.add_argument(
        "--mode",
        default="dev",
        choices=["prod", "dev", "stub", "debug"],
        help="Execution mode (prod/dev/stub/debug).",
    )
    parser.add_argument(
        "--observer",
        action="append",
        help="Observation adapter (noop/log/pixeltable/rerun). Can be repeated or comma-separated.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover")
    discover_parser.add_argument("--video-ref", required=True)
    discover_parser.add_argument("--customer-id")
    discover_parser.add_argument("--progress", action="store_true")

    synthesize_parser = subparsers.add_parser("synthesize")
    synthesize_parser.add_argument(
        "--style", required=True, choices=["base", "closeup", "fullbody", "cinematic"]
    )
    synthesize_parser.add_argument("--progress", action="store_true")
    synthesize_parser.add_argument("--moment-ref")
    synthesize_parser.add_argument("--base-portrait-ref")
    synthesize_parser.add_argument("--closeup-image-ref")
    synthesize_parser.add_argument("--fullbody-image-ref")

    args = parser.parse_args()
    log_level = "DEBUG" if args.mode == "debug" else "INFO"
    configure_logging(level=log_level)
    if args.mode == "prod":
        logger.warning("prod mode uses in-memory stores in the skeleton implementation.")
    observer = _build_observer(_parse_observers(args.observer), args.mode)
    executor = build_executor(args.mode, observer)

    progress = LoguruProgressSink() if args.progress or args.mode == "debug" else None
    if args.command == "discover":
        logger.info(
            "p981-core discover start | mode={} | video_ref={} | customer_id={}",
            args.mode,
            args.video_ref,
            args.customer_id or "none",
        )
        output = executor.run_discover(
            DiscoverInput(video_ref=args.video_ref, customer_id=args.customer_id),
            progress=progress,
        )
    else:
        _validate_synthesize_args(args)
        output = executor.run_synthesize(
            SynthesizeInput(
                style=args.style,
                moment_ref=args.moment_ref,
                base_portrait_ref=args.base_portrait_ref,
                closeup_image_ref=args.closeup_image_ref,
                fullbody_image_ref=args.fullbody_image_ref,
            ),
            progress=progress,
        )

    print(json.dumps(asdict(output), indent=2, sort_keys=True))


def _validate_synthesize_args(args: argparse.Namespace) -> None:
    if args.style == "base" and not args.moment_ref:
        raise SystemExit("--moment-ref is required for style=base")
    if args.style in {"closeup", "fullbody"} and not args.base_portrait_ref:
        raise SystemExit("--base-portrait-ref is required for closeup/fullbody")
    if args.style == "cinematic" and (not args.closeup_image_ref or not args.fullbody_image_ref):
        raise SystemExit("--closeup-image-ref and --fullbody-image-ref are required for cinematic")


def _parse_observers(values: list[str] | None) -> list[str]:
    if not values:
        return []
    selected: list[str] = []
    for item in values:
        if not item:
            continue
        selected.extend([part.strip() for part in item.split(",") if part.strip()])
    return selected


def _build_observer(selected: list[str], mode: str) -> ObservationPort:
    if not selected:
        if mode == "debug":
            return LoguruObservationPort()
        return NoopObservationPort()
    ports: list[ObservationPort] = []
    for name in selected:
        if name == "log":
            ports.append(LoguruObservationPort())
        elif name == "pixeltable":
            ports.append(PixeltableObservationPort())
        elif name == "rerun":
            ports.append(RerunObservationPort())
        elif name == "noop":
            ports.append(NoopObservationPort())
        else:
            ports.append(NoopObservationPort())
    if len(ports) == 1:
        return ports[0]
    return MultiObservationPort(ports)


if __name__ == "__main__":
    main()
