"""CLI command handlers."""

from facemoment.cli.commands.debug import run_debug
from facemoment.cli.commands.process import run_process
from facemoment.cli.commands.info import run_info

__all__ = [
    "run_debug",
    "run_process",
    "run_info",
]
