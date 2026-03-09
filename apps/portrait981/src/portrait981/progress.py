"""Rich-based live progress display for portrait981 batch runs."""

from __future__ import annotations

import time
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.progress import Progress as RichProgress
from rich.table import Table
from rich.text import Text

from portrait981.types import StepEvent

# Step status → icon + style
_STYLES = {
    "started": ("[bold cyan]>>[/]",),
    "completed": ("[bold green]OK[/]",),
    "failed": ("[bold red]FAIL[/]",),
    "skipped": ("[dim]--[/]",),
    "progress": ("[bold cyan]>>[/]",),
}

_STEP_ORDER = ("scan", "lookup", "generate")

# Spinner frames for active scan
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class BatchProgress:
    """Live-updating Rich display tracking batch pipeline progress.

    Usage::

        progress = BatchProgress(total=5)
        pipeline = Portrait981Pipeline(config, on_step=progress.on_step)
        with progress:
            results = pipeline.run_batch(jobs)
    """

    def __init__(self, total: int, console: Optional[Console] = None) -> None:
        self._total = total
        self._console = console or Console(stderr=True)
        self._live: Optional[Live] = None
        # Per-job settled state: {job_id: {step: (status, detail, elapsed)}}
        self._jobs: dict[str, dict[str, tuple[str, str, float]]] = {}
        # Per-job scan progress: {job_id: (frame_id, elapsed, fps)}
        self._scan_progress: dict[str, tuple[int, float, float]] = {}
        # Frame counting for FPS: {job_id: (frame_count, first_frame_time)}
        self._scan_counters: dict[str, tuple[int, float]] = {}
        # job_id → (video_name, index)
        self._meta: dict[str, tuple[str, int]] = {}
        self._completed = 0
        self._tick = 0

    def __enter__(self) -> BatchProgress:
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.__exit__(*args)

    def on_step(self, event: StepEvent) -> None:
        """StepCallback-compatible handler."""
        jid = event.job_id

        if jid not in self._meta:
            self._meta[jid] = (event.video_name or event.member_id, event.job_index)
            self._jobs[jid] = {}

        if event.status == "progress":
            # Update scan progress tracking
            now = time.monotonic()
            if jid not in self._scan_counters:
                self._scan_counters[jid] = (0, now)
            count, first_time = self._scan_counters[jid]
            count += 1
            self._scan_counters[jid] = (count, first_time)
            wall = now - first_time
            fps = count / wall if wall > 0.1 else 0.0
            self._scan_progress[jid] = (event.frame_id, event.elapsed_sec, fps)
        else:
            # Settled status (started/completed/failed/skipped)
            self._jobs[jid][event.step] = (event.status, event.detail, event.elapsed_sec)
            # Clear scan progress on completion
            if event.step == "scan" and event.status != "started":
                self._scan_progress.pop(jid, None)
                self._scan_counters.pop(jid, None)

        # Count completed jobs
        if event.step == "generate" and event.status in ("completed", "failed", "skipped"):
            self._completed = sum(
                1 for j in self._jobs.values()
                if "generate" in j and j["generate"][0] in ("completed", "failed", "skipped")
            )

        self._tick += 1
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Table:
        table = Table(
            title=f"Batch  {self._completed}/{self._total}",
            title_style="bold",
            expand=False,
            show_edge=False,
            show_header=True,
            show_lines=False,
            pad_edge=False,
            box=None,
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Video", min_width=14, max_width=28, no_wrap=True)
        table.add_column("Scan", justify="left", min_width=24)
        table.add_column("Generate", justify="center", width=14)

        # Sort by job index
        sorted_jobs = sorted(self._meta.items(), key=lambda x: x[1][1])

        for job_id, (video_name, index) in sorted_jobs:
            steps = self._jobs.get(job_id, {})
            scan_prog = self._scan_progress.get(job_id)

            row: list = [
                str(index + 1),
                Text(video_name, overflow="ellipsis"),
            ]

            # Scan column — special handling for live progress
            if scan_prog is not None:
                row.append(self._format_scan_progress(*scan_prog))
            elif "scan" in steps:
                row.append(self._format_cell(*steps["scan"]))
            else:
                row.append(Text("", style="dim"))

            # Generate column
            if "generate" in steps:
                row.append(self._format_cell(*steps["generate"]))
            else:
                row.append(Text("", style="dim"))

            table.add_row(*row)

        # Remaining unstarted jobs
        remaining = self._total - len(self._meta)
        if remaining > 0:
            table.add_row(
                "",
                Text(f"  +{remaining} waiting", style="dim italic"),
                "", "",
            )

        return table

    def _format_scan_progress(self, frame_id: int, elapsed: float, fps: float) -> Text:
        """Format the scan cell with live frame counter + spinner."""
        spinner_char = _SPINNER[self._tick % len(_SPINNER)]
        fps_str = f"{fps:.1f}fps" if fps > 0 else ""
        elapsed_str = f"{elapsed:.0f}s"
        return Text.from_markup(
            f"[bold cyan]{spinner_char}[/] frame [bold]{frame_id}[/]"
            f"  [dim]{elapsed_str}  {fps_str}[/]"
        )

    @staticmethod
    def _format_cell(status: str, detail: str, elapsed: float) -> Text:
        icon_markup = _STYLES.get(status, ("[dim]??[/]",))[0]
        parts = [icon_markup]
        if elapsed > 0:
            parts.append(f" [dim]{elapsed:.1f}s[/]")
        if detail and status in ("completed", "failed"):
            short = detail[:22] + ".." if len(detail) > 24 else detail
            parts.append(f" [dim]{short}[/]")
        return Text.from_markup(" ".join(parts))
