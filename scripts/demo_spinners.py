"""Side-by-side demo of progress-indicator styles for the `eval` command.

Run one or all:

    uv run python scripts/demo_spinners.py            # run every demo
    uv run python scripts/demo_spinners.py tqdm       # just option A
    uv run python scripts/demo_spinners.py rich-status

Each demo fakes the eval pipeline phases:
  load HRRR  →  load METAR  →  pair  →  split  →  score persistence
                                              →  score hrrr
                                              →  score climatology

This file is exploratory — pick a winner and the rest gets deleted.
"""
from __future__ import annotations

import sys
import time

PHASES = [
    ("loading HRRR forecasts", 0.7),
    ("loading METAR observations", 0.5),
    ("pairing forecasts to obs", 0.4),
    ("chronological split", 0.2),
    ("scoring persistence", 0.4),
    ("scoring hrrr", 0.4),
    ("scoring climatology", 0.4),
]

# Plausible per-phase results so the ribbon demo can show real-looking stats.
PHASE_RESULTS = {
    "loading HRRR forecasts": "12,453 rows",
    "loading METAR observations": "8,231 obs",
    "pairing forecasts to obs": "7,892 paired",
    "chronological split": "train=5523 val=1183 test=1186",
    "scoring persistence": "RMSE 4.21 kt",
    "scoring hrrr": "RMSE 3.87 kt",
    "scoring climatology": "RMSE 3.95 kt",
}


# ─── Option A: tqdm indeterminate ────────────────────────────────────────────
# Matches the existing ingest commands. Looks the same as the METAR/HRRR bars.
def demo_tqdm() -> None:
    from tqdm.auto import tqdm

    bar = tqdm(total=len(PHASES), desc="eval KMAN", unit="step", dynamic_ncols=True)
    for label, dur in PHASES:
        bar.set_postfix_str(label)
        time.sleep(dur)
        bar.update(1)
    bar.close()


# ─── Option B: rich Status (animated spinner + live text) ────────────────────
# One persistent line, glyph animates, text updates per phase. No bar.
def demo_rich_status() -> None:
    from rich.console import Console

    console = Console()
    # spinner names: dots, dots2, dots12, arc, line, bouncingBar, point, earth, …
    with console.status("[bold cyan]starting…[/]", spinner="dots") as status:
        for label, dur in PHASES:
            status.update(f"[bold cyan]eval KMAN[/] · {label}")
            time.sleep(dur)
    console.print("[green]✓[/] eval KMAN complete")


# ─── Option C: rich Progress with phase column ───────────────────────────────
# Bar + spinner + elapsed time + current-phase column. Most info-dense.
def demo_rich_progress() -> None:
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("[dim]{task.fields[phase]}"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("eval KMAN", total=len(PHASES), phase="starting")
        for label, dur in PHASES:
            progress.update(task, phase=label)
            time.sleep(dur)
            progress.advance(task)


# ─── Option D: yaspin spinner ────────────────────────────────────────────────
# Dedicated spinner lib; many glyph styles; very small footprint.
def demo_yaspin() -> None:
    from yaspin import yaspin
    from yaspin.spinners import Spinners

    with yaspin(Spinners.dots, text="eval KMAN") as sp:
        for label, dur in PHASES:
            sp.text = f"eval KMAN · {label}"
            time.sleep(dur)
        sp.ok("✓")


# ─── Option E: phased click.echo (no animation, just stages) ─────────────────
# No moving glyph; each phase prints when it starts and gets a ✓ when done.
# Cheapest, most "calm". Works fine in CI / non-TTY.
def demo_phased_echo() -> None:
    import click

    for label, dur in PHASES:
        click.echo(f"  … {label}", nl=False, err=True)
        time.sleep(dur)
        click.echo(f"\r  \033[32m✓\033[0m {label}", err=True)


# ─── Option F: rich Live ribbon banner ───────────────────────────────────────
# MAXIMAL version — every plausibly-useful piece of info, intended to be pared
# back. Includes: run-context title (airport, leads, date range), top progress
# bar with %, step counter, elapsed, ETA; per-phase rows with status glyph,
# label, result stats, color-coded skill delta vs HRRR, relative-duration
# sparkline, absolute duration; active phase shows nested sub-progress;
# pending phases shown dim; final ranking row; footer with log path + memory.
def demo_ribbon() -> None:
    import os

    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.text import Text

    console = Console()

    # ── Run-context constants the real eval would have on hand ──────────────
    AIRPORT = "KMAN"
    LEAD_RANGE = "1–18"
    DATE_RANGE = "2020-01-01 → 2024-12-31"
    LOG_PATH = "logs/wind-forecast-2026-05-21T19:59Z.log"
    HRRR_RMSE = 3.87  # used for skill-vs-HRRR coloring

    # Extra per-phase detail (sub-progress and a skill delta where applicable).
    PHASE_DETAIL: dict[str, dict[str, object]] = {
        "loading HRRR forecasts":     {"sub_total": 4151, "sub_unit": "cycles"},
        "loading METAR observations": {"sub_total": 5,    "sub_unit": "stations"},
        "pairing forecasts to obs":   {"unmatched": 339},
        "chronological split":        {},
        "scoring persistence":        {"rmse": 4.21, "crps": 2.41},
        "scoring hrrr":               {"rmse": 3.87, "crps": 2.18},
        "scoring climatology":        {"rmse": 3.95, "crps": 2.24},
    }

    def skill_delta(label: str) -> Text | None:
        rmse = PHASE_DETAIL.get(label, {}).get("rmse")
        if not isinstance(rmse, float):
            return None
        if label == "scoring hrrr":
            return Text("baseline", style="dim")
        skill = 1 - rmse / HRRR_RMSE  # >0 = better than HRRR
        color = "green" if skill > 0 else "red"
        sign = "+" if skill > 0 else ""
        return Text(f"{sign}{skill * 100:.1f}% vs HRRR", style=color)

    # Sparkline cells for relative phase cost. Filled in as phases finish.
    SPARK = "▁▂▃▄▅▆▇█"
    def spark(secs: float, max_secs: float) -> str:
        if max_secs <= 0:
            return SPARK[0]
        idx = min(len(SPARK) - 1, int(secs / max_secs * (len(SPARK) - 1)))
        return SPARK[idx]

    progress = Progress(
        TextColumn("[bold cyan]eval {task.fields[airport]}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("· step {task.completed}/{task.total}"),
        TextColumn("· [dim]elapsed[/] {task.fields[elapsed]}"),
        TextColumn("· [dim]eta[/] {task.fields[eta]}"),
    )
    task_id = progress.add_task(
        "eval", total=len(PHASES), airport=AIRPORT, elapsed="0.0s", eta="—",
    )

    completed: list[tuple[str, str, float]] = []  # (label, result, secs)
    started = time.monotonic()

    def memory_mb() -> int:
        # Rough RSS (Linux only); falls back to a fake number for portability.
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) // 1024
        except OSError:
            pass
        return 412

    def render(active: str | None, sub_done: int = 0) -> Panel:
        max_secs = max((s for _, _, s in completed), default=0.0)

        rows = Table.grid(padding=(0, 1), expand=True)
        rows.add_column(width=2)        # status glyph
        rows.add_column(ratio=3)        # phase label
        rows.add_column(ratio=4)        # result + skill
        rows.add_column(width=1)        # spark
        rows.add_column(justify="right", width=6)  # timing

        for label, result, secs in completed:
            result_text = Text(result, style="dim")
            delta = skill_delta(label)
            if delta is not None:
                result_text.append("  ")
                result_text.append_text(delta)
            rows.add_row(
                Text("✓", style="green"),
                Text(label),
                result_text,
                Text(spark(secs, max_secs), style="cyan"),
                Text(f"{secs:.1f}s", style="dim"),
            )

        if active is not None:
            detail = PHASE_DETAIL.get(active, {})
            sub_total = detail.get("sub_total")
            if isinstance(sub_total, int):
                sub_unit = detail.get("sub_unit", "")
                result_cell: Text = Text(
                    f"{sub_done:,}/{sub_total:,} {sub_unit}", style="cyan",
                )
            else:
                result_cell = Text("…", style="dim")
            rows.add_row(
                Spinner("dots", style="cyan"),
                Text(active, style="bold"),
                result_cell,
                Text(""),
                Text(""),
            )

        # Pending phases listed dim so the operator can see what's coming.
        done_labels = {lbl for lbl, _, _ in completed}
        for label, _ in PHASES:
            if label in done_labels or label == active:
                continue
            rows.add_row(
                Text("◯", style="dim"),
                Text(label, style="dim"),
                Text("pending", style="dim"),
                Text(""),
                Text(""),
            )

        # Final ranking row, only after every baseline has scored.
        ranking: Text | None = None
        if len(completed) == len(PHASES):
            scored = [
                (lbl.removeprefix("scoring "), PHASE_DETAIL[lbl]["rmse"])
                for lbl, _, _ in completed
                if "rmse" in PHASE_DETAIL.get(lbl, {})
            ]
            scored.sort(key=lambda t: t[1])  # type: ignore[arg-type]
            ranking = Text("ranking: ", style="bold")
            for i, (name, rmse) in enumerate(scored):
                if i:
                    ranking.append("  →  ", style="dim")
                ranking.append(f"{name} ", style="cyan")
                ranking.append(f"({rmse:.2f} kt)", style="dim")

        footer = Text.from_markup(
            f"[dim]log:[/] {LOG_PATH}  ·  [dim]mem:[/] {memory_mb()} MB  ·  "
            f"[dim]pid:[/] {os.getpid()}"
        )

        body_items = [progress, Text(""), rows]
        if ranking is not None:
            body_items += [Text(""), ranking]
        body_items += [Text(""), footer]

        return Panel(
            Group(*body_items),
            border_style="cyan",
            title=(
                f"[bold]wind-forecast[/] · [bold]eval {AIRPORT}[/] · "
                f"leads {LEAD_RANGE} · {DATE_RANGE}"
            ),
            title_align="left",
            subtitle=f"[dim]{len(completed)}/{len(PHASES)} done[/]",
            subtitle_align="right",
        )

    total_planned = sum(d for _, d in PHASES)

    with Live(render(PHASES[0][0]), console=console, refresh_per_second=12) as live:
        for label, dur in PHASES:
            phase_start = time.monotonic()
            detail = PHASE_DETAIL.get(label, {})
            sub_total = detail.get("sub_total")

            # If the phase has a sub-progress, tick through it visibly.
            if isinstance(sub_total, int):
                ticks = 12
                for i in range(1, ticks + 1):
                    live.update(render(label, sub_done=int(sub_total * i / ticks)))
                    time.sleep(dur / ticks)
            else:
                live.update(render(label))
                time.sleep(dur)

            secs = time.monotonic() - phase_start
            completed.append((label, PHASE_RESULTS[label], secs))
            elapsed = time.monotonic() - started
            remaining = max(0.0, total_planned - elapsed)
            progress.update(
                task_id,
                advance=1,
                elapsed=f"{elapsed:.1f}s",
                eta=f"~{remaining:.1f}s" if remaining > 0 else "—",
            )
            live.update(render(None))

        live.update(render(None))


DEMOS = {
    "tqdm": demo_tqdm,
    "rich-status": demo_rich_status,
    "rich-progress": demo_rich_progress,
    "yaspin": demo_yaspin,
    "phased": demo_phased_echo,
    "ribbon": demo_ribbon,
}


def main() -> None:
    names = sys.argv[1:] or list(DEMOS)
    for name in names:
        if name not in DEMOS:
            print(f"unknown demo {name!r}; pick from {list(DEMOS)}")
            continue
        print(f"\n── {name} " + "─" * (70 - len(name)))
        DEMOS[name]()


if __name__ == "__main__":
    main()
