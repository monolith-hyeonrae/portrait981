from __future__ import annotations

import sys

from loguru import logger

from .progress import ProgressSink, ProgressUpdate


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


class LoguruProgressSink(ProgressSink):
    _STEP_WIDTH = 36

    def __init__(self, level: str = "INFO") -> None:
        self._level = level

    def emit(self, update: ProgressUpdate) -> None:
        progress_pct = "--%"
        if update.progress is not None:
            progress_pct = f"{update.progress * 100:>3.0f}%"
        message = update.message or update.step
        step = f"{update.step:<{self._STEP_WIDTH}}"
        logger.log(
            self._level,
            f"[{update.stage}] {step} | {progress_pct:>4} | {message}",
        )
