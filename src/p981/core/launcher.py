"""p981-core 스테이지를 CLI로 실행하는 엔트리포인트.

흐름:
1) CLI 인자 파싱 및 로깅 설정
2) wiring을 통해 옵저버/실행기 생성
3) discover/synthesize 실행 후 결과 JSON 출력
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from loguru import logger

from .common import LoguruProgressSink, configure_logging
from .stage.discover import DiscoverInput
from .stage.synthesize import SynthesizeInput
from .wiring import build_executor, build_observer


def _build_parser() -> argparse.ArgumentParser:
    """스테이지 실행용 CLI 플래그와 서브커맨드를 정의한다."""

    # 메인 파서

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
        help="Observation adapter (noop/log/frames/rerun). Can be repeated or comma-separated.",
    )

    # 서브커맨드

    subparsers = parser.add_subparsers(dest="command", required=True)

    # discover 명령
    discover_parser = subparsers.add_parser("discover")
    discover_parser.add_argument("--video-ref", required=True)
    discover_parser.add_argument(
        "--member-id",
        "--customer-id",
        dest="member_id",
        help="Member identifier used for history-aware dedupe (optional).",
    )
    discover_parser.add_argument("--progress", action="store_true")

    # synthesize 명령
    synthesize_parser = subparsers.add_parser("synthesize")
    synthesize_parser.add_argument(
        "--style", required=True, choices=["base", "closeup", "fullbody", "cinematic"]
    )
    synthesize_parser.add_argument("--progress", action="store_true")
    synthesize_parser.add_argument("--moment-ref")
    synthesize_parser.add_argument("--base-portrait-ref")
    synthesize_parser.add_argument("--closeup-image-ref")
    synthesize_parser.add_argument("--fullbody-image-ref")

    return parser


def _build_progress_sink(args: argparse.Namespace) -> LoguruProgressSink | None:
    """요청 시(또는 debug) 진행률 출력 핸들러를 만든다."""

    if args.progress or args.mode == "debug":
        return LoguruProgressSink()
    return None


def _validate_synthesize_args(args: argparse.Namespace) -> None:
    """스타일별 필수 입력을 검증한다."""

    if args.style == "base" and not args.moment_ref:
        raise SystemExit("--moment-ref is required for style=base")

    if args.style in {"closeup", "fullbody"} and not args.base_portrait_ref:
        raise SystemExit("--base-portrait-ref is required for closeup/fullbody")

    if args.style == "cinematic" and (not args.closeup_image_ref or not args.fullbody_image_ref):
        raise SystemExit("--closeup-image-ref and --fullbody-image-ref are required for cinematic")


def _parse_observers(values: list[str] | None) -> list[str]:
    """반복/쉼표 구분 옵션을 평탄화해 리스트로 만든다."""

    if not values:
        return []
    selected: list[str] = []
    for item in values:
        if not item:
            continue
        selected.extend([part.strip() for part in item.split(",") if part.strip()])
    return selected


def main() -> None:
    """CLI 인자를 파싱하고 의존성을 조립한 뒤 선택된 스테이지를 실행한다."""

    # arguments
    parser = _build_parser()
    args = parser.parse_args()

    # mode setting
    log_level = "DEBUG" if args.mode == "debug" else "INFO"
    configure_logging(level=log_level)

    if args.mode == "prod":
        logger.warning("prod mode uses in-memory stores in the skeleton implementation.")

    # observer setting
    observer = build_observer(_parse_observers(args.observer), args.mode)
    executor = build_executor(args.mode, observer)
    progress = _build_progress_sink(args)

    # 커맨드 디스패치.
    if args.command == "discover":

        logger.info(
            "p981-core discover start | mode={} | video_ref={} | member_id={}",
            args.mode,
            args.video_ref,
            args.member_id or "none",
        )

        output = executor.run_discover(
            DiscoverInput(video_ref=args.video_ref, member_id=args.member_id),
            progress=progress,
        )

    elif args.command == "synthesize":

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

    else:
        logger.warning("Unknown command: %s", args.command)
        return

    print(f"\n{args.command.upper()} Stage Output:")
    print(json.dumps(asdict(output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
