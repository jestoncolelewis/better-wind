"""Live "ribbon" panel for the `eval` command.

A `rich.live.Live` region renders a task-list panel: completed phases stay
visible with their result + timing, the active phase shows a spinner with
optional sub-progress, pending phases are listed dim. Top of the panel has a
bar with %, step counter, elapsed, ETA. Bottom has log path, memory, pid.
After the last baseline scores, a ranking row sorts baselines by RMSE.

`cli.eval_cmd` instantiates one of these and wires it to `evaluate_airport`'s
phase callbacks. Falls back to no-op when stderr isn't a TTY.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

_SPARK = "▁▂▃▄▅▆▇█"


@dataclass
class _Completed:
    label: str
    result: str
    skill_text: str | None
    skill_style: str
    secs: float
    rmse: float | None


class EvalRibbon:
    """Phased progress UI for `eval`. Use as a context manager.

    >>> with EvalRibbon(airport_icao="KMAN", all_phases=[...], log_path=p,
    ...                  baseline_names=["hrrr"]) as r:
    ...     evaluate_airport(airport, on_phase_start=r.begin,
    ...                                on_phase_done=r.end_phase,
    ...                                on_load_progress=r.sub_progress)
    """

    def __init__(
        self,
        *,
        airport_icao: str,
        all_phases: list[str],
        log_path: Path | str,
        baseline_names: list[str],
    ) -> None:
        self.airport_icao = airport_icao
        self.all_phases = all_phases
        self.log_path = str(log_path)
        self.baseline_names = baseline_names

        self._completed: list[_Completed] = []
        self._active: str | None = None
        self._active_start: float = 0.0
        self._sub_done: int = 0
        self._sub_total: int = 0
        self._started: float = 0.0
        self._hrrr_rmse: float | None = None

        # Lifecycle members (initialized in __enter__).
        self._console: Console | None = None
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task_id: int | None = None

    # ── context manager ─────────────────────────────────────────────────────
    def __enter__(self) -> EvalRibbon:
        self._console = Console(stderr=True)
        self._progress = Progress(
            TextColumn(f"[bold cyan]eval {self.airport_icao}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("· step {task.completed}/{task.total}"),
            TextColumn("· [dim]elapsed[/] {task.fields[elapsed]}"),
            TextColumn("· [dim]eta[/] {task.fields[eta]}"),
            console=self._console,
        )
        self._task_id = self._progress.add_task(
            "eval", total=len(self.all_phases), elapsed="0.0s", eta="—",
        )
        self._started = time.monotonic()
        self._live = Live(
            self._render(), console=self._console, refresh_per_second=12,
        )
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.__exit__(exc_type, exc, tb)

    # ── phase callbacks (match harness.PhaseStart / PhaseDone signatures) ───
    def begin(self, phase: str) -> None:
        self._active = phase
        self._active_start = time.monotonic()
        self._sub_done = 0
        self._sub_total = 0
        self._refresh()

    def end_phase(self, phase: str, info: dict[str, Any]) -> None:
        secs = time.monotonic() - self._active_start
        result = _format_result(phase, info)
        rmse_raw = info.get("rmse")
        rmse: float | None = float(rmse_raw) if isinstance(rmse_raw, (int, float)) else None

        skill_text: str | None = None
        skill_style = "dim"
        if phase.startswith("scoring "):
            if phase == "scoring hrrr":
                self._hrrr_rmse = rmse
                skill_text = "baseline"
                skill_style = "dim"
            elif rmse is not None and self._hrrr_rmse is not None and self._hrrr_rmse > 0:
                skill = 1 - rmse / self._hrrr_rmse
                sign = "+" if skill > 0 else ""
                skill_text = f"{sign}{skill * 100:.1f}% vs HRRR"
                skill_style = "green" if skill > 0 else "red"

        self._completed.append(_Completed(
            label=phase,
            result=result,
            skill_text=skill_text,
            skill_style=skill_style,
            secs=secs,
            rmse=rmse,
        ))
        self._active = None

        # Top progress bar: advance, refresh elapsed + ETA.
        elapsed = time.monotonic() - self._started
        done = len(self._completed)
        if done < len(self.all_phases) and done > 0:
            avg = elapsed / done
            remaining = avg * (len(self.all_phases) - done)
            eta = f"~{remaining:.1f}s"
        else:
            eta = "—"
        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id, advance=1, elapsed=f"{elapsed:.1f}s", eta=eta,
            )
        self._refresh()

    def sub_progress(self, done: int, total: int) -> None:
        """Hook for `on_load_progress` — updates the active phase's sub-counter."""
        self._sub_done = done
        self._sub_total = total
        self._refresh()

    # ── rendering ───────────────────────────────────────────────────────────
    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Panel:
        max_secs = max((c.secs for c in self._completed), default=0.0)

        def spark(secs: float) -> str:
            if max_secs <= 0:
                return _SPARK[0]
            idx = min(len(_SPARK) - 1, int(secs / max_secs * (len(_SPARK) - 1)))
            return _SPARK[idx]

        rows = Table.grid(padding=(0, 1), expand=True)
        rows.add_column(width=2)                    # status glyph
        rows.add_column(ratio=3)                    # phase label
        rows.add_column(ratio=4)                    # result + skill
        rows.add_column(width=1)                    # spark
        rows.add_column(justify="right", width=6)   # timing

        for c in self._completed:
            result = Text(c.result, style="dim")
            if c.skill_text is not None:
                result.append("  ")
                result.append(c.skill_text, style=c.skill_style)
            rows.add_row(
                Text("✓", style="green"),
                Text(c.label),
                result,
                Text(spark(c.secs), style="cyan"),
                Text(f"{c.secs:.1f}s", style="dim"),
            )

        if self._active is not None:
            if self._sub_total > 0:
                result_cell: Text = Text(
                    f"{self._sub_done:,}/{self._sub_total:,} files", style="cyan",
                )
            else:
                result_cell = Text("…", style="dim")
            rows.add_row(
                Spinner("dots", style="cyan"),
                Text(self._active, style="bold"),
                result_cell,
                Text(""),
                Text(""),
            )

        done_labels = {c.label for c in self._completed}
        for label in self.all_phases:
            if label in done_labels or label == self._active:
                continue
            rows.add_row(
                Text("◯", style="dim"),
                Text(label, style="dim"),
                Text("pending", style="dim"),
                Text(""),
                Text(""),
            )

        # Ranking row once every baseline has scored.
        ranking: Text | None = None
        scored = [
            (c.label.removeprefix("scoring "), c.rmse)
            for c in self._completed
            if c.label.startswith("scoring ") and c.rmse is not None
        ]
        if len(scored) == len(self.baseline_names) and scored:
            scored.sort(key=lambda t: t[1] if t[1] is not None else float("inf"))
            ranking = Text("ranking: ", style="bold")
            for i, (name, rmse) in enumerate(scored):
                if i:
                    ranking.append("  →  ", style="dim")
                ranking.append(f"{name} ", style="cyan")
                ranking.append(f"({rmse:.2f} kt)", style="dim")

        footer = Text.from_markup(
            f"[dim]log:[/] {self.log_path}  ·  "
            f"[dim]mem:[/] {_memory_mb()} MB  ·  "
            f"[dim]pid:[/] {os.getpid()}"
        )

        assert self._progress is not None
        body: list[Any] = [self._progress, Text(""), rows]
        if ranking is not None:
            body += [Text(""), ranking]
        body += [Text(""), footer]

        return Panel(
            Group(*body),
            border_style="cyan",
            title=f"[bold]wind-forecast[/] · [bold]eval {self.airport_icao}[/]",
            title_align="left",
            subtitle=f"[dim]{len(self._completed)}/{len(self.all_phases)} done[/]",
            subtitle_align="right",
        )


# ── helpers ────────────────────────────────────────────────────────────────
def _format_result(phase: str, info: dict[str, Any]) -> str:
    """Turn an info dict from `harness.evaluate_airport` into a display string."""
    if phase == "loading HRRR forecasts":
        return f"{int(info.get('rows', 0)):,} rows, {int(info.get('cycles', 0)):,} cycles"
    if phase == "loading METAR observations":
        return f"{int(info.get('obs', 0)):,} obs"
    if phase == "pairing forecasts to obs":
        return f"{int(info.get('paired', 0)):,} paired"
    if phase == "chronological split":
        return (
            f"train={int(info.get('train', 0)):,} "
            f"val={int(info.get('val', 0)):,} "
            f"test={int(info.get('test', 0)):,}"
        )
    if phase.startswith("scoring "):
        rmse = info.get("rmse")
        if isinstance(rmse, (int, float)):
            return f"RMSE {rmse:.2f} kt"
        return "—"
    return ""


def _memory_mb() -> int:
    """Best-effort RSS for the footer. Returns 0 if we can't read it."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kibibytes.
        return kb // (1024 * 1024) if sys.platform == "darwin" else kb // 1024
    except (OSError, ImportError):
        return 0


def phases_for(baseline_names: list[str]) -> list[str]:
    """Phase labels in the exact order `evaluate_airport` will emit them."""
    return [
        "loading HRRR forecasts",
        "loading METAR observations",
        "pairing forecasts to obs",
        "chronological split",
    ] + [f"scoring {n}" for n in baseline_names]


@contextlib.contextmanager
def maybe_ribbon(
    *,
    airport_icao: str,
    baseline_names: list[str],
    log_path: Path | str,
) -> Iterator[EvalRibbon | None]:
    """Yield an `EvalRibbon` when stderr is a TTY, otherwise `None`.

    Lets the CLI write `if r: r.begin(...)` without a separate code path for
    non-interactive runs (CI, redirected output).
    """
    if not sys.stderr.isatty():
        yield None
        return
    with EvalRibbon(
        airport_icao=airport_icao,
        all_phases=phases_for(baseline_names),
        log_path=log_path,
        baseline_names=baseline_names,
    ) as ribbon:
        yield ribbon
