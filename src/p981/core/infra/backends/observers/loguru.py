"""Loguru 기반 observer 백엔드."""

from __future__ import annotations

import sys

from loguru import logger

from p981.core.application.protocols.observer import ObserverEvent


def _configure_loguru(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


class LoguruObserverBackend:
    _STEP_WIDTH = 36  # 진행률 로그에서 스텝명을 정렬하기 위한 폭.

    def __init__(self, level: str = "INFO") -> None:
        self._level = level
        _configure_loguru(level)

    def emit(self, event: ObserverEvent) -> None:
        if event.kind == "log":
            self._emit_log(event)
            return
        if event.kind == "progress":
            self._emit_progress(event)
            return

        payload = event.payload
        frame_index = payload.get("frame_index")
        timestamp_ms = payload.get("timestamp_ms")
        avg_luma = payload.get("avg_luma")
        logger.info(
            "obs {} | frame={} ts_ms={} avg_luma={}",
            event.kind,
            frame_index,
            timestamp_ms,
            avg_luma,
        )

    def _emit_log(self, event: ObserverEvent) -> None:
        payload = event.payload
        level = str(payload.get("level", self._level)).upper()
        message = str(payload.get("message", ""))
        extra = {
            key: value
            for key, value in payload.items()
            if key not in {"level", "message"} and value is not None
        }
        if extra:
            message = f"{message} | {extra}"
        logger.log(level, message)

    def _emit_progress(self, event: ObserverEvent) -> None:
        payload = event.payload
        level = str(payload.get("level", self._level)).upper()
        stage = payload.get("stage", "stage")
        step = payload.get("step", "")
        message = payload.get("message") or step
        progress = payload.get("progress")
        progress_pct = "--%"
        if isinstance(progress, (int, float)):
            progress_pct = f"{progress * 100:>3.0f}%"

        step_label = f"{step:<{self._STEP_WIDTH}}"
        logger.log(level, f"[{stage}] {step_label} | {progress_pct:>4} | {message}")
