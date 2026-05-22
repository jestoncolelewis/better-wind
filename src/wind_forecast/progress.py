"""Live "ribbon" panel used by the CLI's long-running commands.

A `rich.live.Live` region renders a task-list panel: completed phases stay
visible with their result + timing, the active phase shows a spinner with
optional sub-progress, pending phases are listed dim. Top of the panel has a
bar with %, step counter, elapsed, ETA. Bottom shows log path, RSS, pid.

Subclasses inject command-specific result formatting (`_format_result`) and
optional extras above the footer (`_build_extras`). The default
`_build_completed_content` is enough for ingest-style commands; eval
overrides it to annotate each baseline with a colored skill delta vs HRRR.

All ribbon classes are used as context managers via `maybe_ribbon(factory)`,
which yields `None` when stderr isn't a TTY — CI runs stay silent.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, ProgressColumn, TaskID, TextColumn
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

_SPARK = "▁▂▃▄▅▆▇█"


class _DynamicTextColumn(ProgressColumn):
    """Column whose markup is computed by a callable on each render — lets
    counters tick at Live's refresh rate without push updates."""

    def __init__(self, compute: Callable[[], str]) -> None:
        super().__init__()
        self._compute = compute

    def render(self, task: Any) -> Text:
        del task
        return Text.from_markup(self._compute())


def _fmt_duration(secs: float) -> str:
    """Compact human-readable duration. Top two units, e.g. `2d07h`, `1h05m`."""
    if secs < 0:
        secs = 0
    if secs < 60:
        return f"{secs:.1f}s"
    s = int(round(secs))
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


@dataclass
class _Completed:
    label: str
    result: str
    skill_text: str | None
    skill_style: str
    secs: float
    rmse: float | None


class JobRibbon:
    """General phased progress UI. Subclass for command-specific behaviour."""

    def __init__(
        self,
        *,
        title: str,
        all_phases: list[str],
        log_path: Path | str,
        sub_unit: str = "items",
    ) -> None:
        self.title = title
        self.all_phases = all_phases
        self.log_path = str(log_path)
        self.sub_unit = sub_unit

        self._completed: list[_Completed] = []
        self._active: str | None = None
        self._active_start: float = 0.0
        self._sub_done: int = 0
        self._sub_total: int = 0
        self._sub_summary: str = ""
        self._started: float = 0.0

        # Lifecycle members initialized in __enter__.
        self._console: Console | None = None
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._log_redirect: contextlib.AbstractContextManager[None] | None = None

    # ── context manager ─────────────────────────────────────────────────────
    def __enter__(self) -> JobRibbon:
        self._console = Console(stderr=True)
        self._started = time.monotonic()
        # Dynamic columns are evaluated each render, so the counters tick at
        # Live's refresh rate without us having to push updates.
        self._progress = Progress(
            TextColumn(f"[bold cyan]{self.title}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            _DynamicTextColumn(self._compute_step_text),
            _DynamicTextColumn(self._compute_elapsed),
            _DynamicTextColumn(self._compute_eta),
            console=self._console,
        )
        self._task_id = self._progress.add_task("job", total=len(self.all_phases))

        # Route logs through rich so warnings don't tear the live region.
        self._log_redirect = rich_progress_logging(self._console)
        self._log_redirect.__enter__()

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
        if self._log_redirect is not None:
            self._log_redirect.__exit__(exc_type, exc, tb)

    # ── phase callbacks ─────────────────────────────────────────────────────
    def begin(self, phase: str) -> None:
        self._active = phase
        self._active_start = time.monotonic()
        self._sub_done = 0
        self._sub_total = 0
        self._sub_summary = ""
        self._sync_task_progress()
        self._refresh()

    def end_phase(self, phase: str, info: dict[str, Any]) -> None:
        secs = time.monotonic() - self._active_start
        result, skill_text, skill_style = self._build_completed_content(phase, info)
        rmse = _as_float(info.get("rmse"))

        self._completed.append(_Completed(
            label=phase,
            result=result,
            skill_text=skill_text,
            skill_style=skill_style,
            secs=secs,
            rmse=rmse,
        ))
        self._active = None

        self._sync_task_progress()
        self._refresh()

    def _sync_task_progress(self) -> None:
        """Push a fractional `completed` to the Progress task so the bar +
        percentage reflect both finished phases and sub-progress within the
        current phase."""
        if self._progress is None or self._task_id is None:
            return
        progress = float(len(self._completed))
        if self._active and self._sub_total > 0 and self._sub_done > 0:
            progress += min(1.0, self._sub_done / self._sub_total)
        self._progress.update(self._task_id, completed=progress)

    def _compute_step_text(self) -> str:
        return f"· step {len(self._completed)}/{len(self.all_phases)}"

    def _compute_elapsed(self) -> str:
        elapsed = time.monotonic() - self._started
        return f"· [dim]elapsed[/] {_fmt_duration(elapsed)}"

    def _compute_eta(self) -> str:
        """Best-effort ETA, recomputed on every render.

        Within an active phase with sub-progress data, extrapolates from the
        observed rate. Otherwise falls back to average completed-phase
        duration × remaining phases.
        """
        remaining: float | None = None
        if self._active and self._sub_total > 0 and self._sub_done > 0:
            active_elapsed = time.monotonic() - self._active_start
            if active_elapsed > 0:
                rate = self._sub_done / active_elapsed
                if rate > 0:
                    remaining = (self._sub_total - self._sub_done) / rate

        if remaining is None:
            done = len(self._completed)
            if 0 < done < len(self.all_phases):
                elapsed = time.monotonic() - self._started
                avg = elapsed / done
                remaining = avg * (len(self.all_phases) - done)

        if remaining is None:
            return "· [dim]eta[/] —"
        return f"· [dim]eta[/] ~{_fmt_duration(remaining)}"

    def sub_progress(
        self,
        done: int,
        total: int,
        *,
        summary: str = "",
    ) -> None:
        """Update the active phase's sub-counter and optional summary text."""
        self._sub_done = done
        self._sub_total = total
        self._sub_summary = summary
        self._sync_task_progress()
        self._refresh()

    # ── hooks for subclasses ────────────────────────────────────────────────
    def _build_completed_content(
        self, phase: str, info: dict[str, Any]
    ) -> tuple[str, str | None, str]:
        """Return `(result_text, skill_text, skill_style)` for one completed phase."""
        return self._format_result(phase, info), None, "dim"

    def _format_result(self, phase: str, info: dict[str, Any]) -> str:
        """Override for command-specific result formatting."""
        return str(info.get("result", ""))

    def _build_extras(self) -> list[Text]:
        """Override to render extra rows above the footer (e.g. ranking)."""
        return []

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
                    f"{self._sub_done:,}/{self._sub_total:,} {self.sub_unit}",
                    style="cyan",
                )
                if self._sub_summary:
                    result_cell.append(" · ", style="dim")
                    result_cell.append(self._sub_summary, style="dim")
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

        extras = self._build_extras()
        footer = Text.from_markup(
            f"[dim]log:[/] {self.log_path}  ·  "
            f"[dim]mem:[/] {_memory_mb()} MB  ·  "
            f"[dim]pid:[/] {os.getpid()}"
        )

        assert self._progress is not None
        body: list[Any] = [self._progress, Text(""), rows]
        for extra in extras:
            body += [Text(""), extra]
        body += [Text(""), footer]

        return Panel(
            Group(*body),
            border_style="cyan",
            title=f"[bold]wind-forecast[/] · [bold]{self.title}[/]",
            title_align="left",
            subtitle=f"[dim]{len(self._completed)}/{len(self.all_phases)} done[/]",
            subtitle_align="right",
        )


# ── command-specific ribbons ───────────────────────────────────────────────
class EvalRibbon(JobRibbon):
    """Ribbon for `wind-forecast eval`. Adds skill-vs-HRRR deltas and a ranking row."""

    def __init__(
        self,
        *,
        airport_icao: str,
        baseline_names: list[str],
        log_path: Path | str,
    ) -> None:
        super().__init__(
            title=f"eval {airport_icao}",
            all_phases=eval_phases_for(baseline_names),
            log_path=log_path,
            sub_unit="files",
        )
        self.baseline_names = baseline_names
        self._hrrr_rmse: float | None = None

    def _build_completed_content(
        self, phase: str, info: dict[str, Any]
    ) -> tuple[str, str | None, str]:
        result = self._format_result(phase, info)
        if not phase.startswith("scoring "):
            return result, None, "dim"

        rmse = _as_float(info.get("rmse"))
        if phase == "scoring hrrr":
            self._hrrr_rmse = rmse
            return result, "baseline", "dim"
        if rmse is None or self._hrrr_rmse is None or self._hrrr_rmse <= 0:
            return result, None, "dim"
        skill = 1 - rmse / self._hrrr_rmse
        sign = "+" if skill > 0 else ""
        return (
            result,
            f"{sign}{skill * 100:.1f}% vs HRRR",
            "green" if skill > 0 else "red",
        )

    def _format_result(self, phase: str, info: dict[str, Any]) -> str:
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
            rmse = _as_float(info.get("rmse"))
            return f"RMSE {rmse:.2f} kt" if rmse is not None else "—"
        return ""

    def _build_extras(self) -> list[Text]:
        scored = [
            (c.label.removeprefix("scoring "), c.rmse)
            for c in self._completed
            if c.label.startswith("scoring ") and c.rmse is not None
        ]
        if len(scored) != len(self.baseline_names) or not scored:
            return []
        scored.sort(key=lambda t: t[1] if t[1] is not None else float("inf"))
        ranking = Text("ranking: ", style="bold")
        for i, (name, rmse) in enumerate(scored):
            if i:
                ranking.append("  →  ", style="dim")
            ranking.append(f"{name} ", style="cyan")
            assert rmse is not None
            ranking.append(f"({rmse:.2f} kt)", style="dim")
        return [ranking]


class IngestMetarRibbon(JobRibbon):
    """Ribbon for `wind-forecast ingest-metar`. Single phase, chunk-level sub-progress."""

    PHASE = "fetching METAR"

    def __init__(self, *, airport_icao: str, log_path: Path | str) -> None:
        super().__init__(
            title=f"ingest-metar {airport_icao}",
            all_phases=[self.PHASE],
            log_path=log_path,
            sub_unit="chunks",
        )

    def _format_result(self, phase: str, info: dict[str, Any]) -> str:
        parts: list[str] = []
        if "stations" in info:
            parts.append(f"{int(info['stations'])} stations")
        if "rows" in info:
            parts.append(f"{int(info['rows']):,} rows")
        if info.get("failed"):
            parts.append(f"{int(info['failed'])} failed")
        return " · ".join(parts) or "done"


class IngestHrrrRibbon(JobRibbon):
    """Ribbon for `wind-forecast ingest-hrrr`. Single phase, cycle-level sub-progress."""

    PHASE = "fetching HRRR cycles"

    def __init__(self, *, airport_icao: str, log_path: Path | str) -> None:
        super().__init__(
            title=f"ingest-hrrr {airport_icao}",
            all_phases=[self.PHASE],
            log_path=log_path,
            sub_unit="cycles",
        )

    def _format_result(self, phase: str, info: dict[str, Any]) -> str:
        parts: list[str] = []
        if "written" in info:
            parts.append(f"{int(info['written'])} written")
        if "skipped" in info:
            parts.append(f"{int(info['skipped'])} skipped")
        if info.get("failed"):
            parts.append(f"{int(info['failed'])} failed")
        if info.get("empty"):
            parts.append(f"{int(info['empty'])} empty")
        if "rows" in info:
            parts.append(f"{int(info['rows']):,} rows")
        return " · ".join(parts) or "done"


# ── helpers ────────────────────────────────────────────────────────────────
def eval_phases_for(baseline_names: list[str]) -> list[str]:
    """Phase labels in the exact order `evaluate_airport` will emit them."""
    return [
        "loading HRRR forecasts",
        "loading METAR observations",
        "pairing forecasts to obs",
        "chronological split",
    ] + [f"scoring {n}" for n in baseline_names]


def _as_float(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


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
        return kb // (1024 * 1024) if sys.platform == "darwin" else kb // 1024
    except (OSError, ImportError):
        return 0


@contextlib.contextmanager
def rich_progress_logging(console: Console) -> Iterator[None]:
    """Route console log records through rich while a Live region is active.

    A vanilla `logging.StreamHandler` writes directly to stderr and would
    tear up the live panel. We swap it for a `RichHandler` bound to the same
    console rich Live is rendering on — log lines render above the live
    region cleanly. Level and filters from the original handler are copied
    so verbosity behaviour matches what `setup_logging` configured.
    """
    root = logging.getLogger()
    swapped: list[tuple[logging.Handler, list[logging.Filter], int]] = []
    added: list[RichHandler] = []
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            swapped.append((h, list(h.filters), h.level))
            root.removeHandler(h)
            rh = RichHandler(
                console=console,
                show_path=False,
                show_time=False,
                rich_tracebacks=False,
                markup=False,
            )
            rh.setLevel(h.level)
            for f in h.filters:
                rh.addFilter(f)
            root.addHandler(rh)
            added.append(rh)
    try:
        yield
    finally:
        for rh in added:
            root.removeHandler(rh)
        for h, filters, level in swapped:
            h.setLevel(level)
            for f in filters:
                if f not in h.filters:
                    h.addFilter(f)
            root.addHandler(h)


@contextlib.contextmanager
def maybe_ribbon(factory: Callable[[], JobRibbon]) -> Iterator[JobRibbon | None]:
    """Yield a ribbon when stderr is a TTY, otherwise `None`.

    Usage:
        with maybe_ribbon(lambda: EvalRibbon(...)) as ribbon:
            do_work(on_progress=ribbon.sub_progress if ribbon else None)
    """
    if not sys.stderr.isatty():
        yield None
        return
    with factory() as ribbon:
        yield ribbon
