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
# Full-panel "task list" — completed phases stay on screen with their result
# and timing, the active phase shows a live spinner, and a progress bar +
# elapsed clock pin to the top. Closer to `terraform plan` / `vercel deploy`.
def demo_ribbon() -> None:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.text import Text

    console = Console()

    progress = Progress(
        TextColumn("[bold cyan]eval KMAN"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("· step {task.completed}/{task.total}"),
        TextColumn("· [dim]elapsed[/] {task.fields[elapsed]}"),
    )
    task_id = progress.add_task("eval", total=len(PHASES), elapsed="0.0s")

    completed: list[tuple[str, str, float]] = []  # (label, result, secs)
    started = time.monotonic()

    def render(active: str | None) -> Panel:
        rows = Table.grid(padding=(0, 1), expand=True)
        rows.add_column(width=2)        # status glyph
        rows.add_column(ratio=2)        # phase label
        rows.add_column(ratio=2)        # result
        rows.add_column(justify="right")  # timing

        for label, result, secs in completed:
            rows.add_row(
                Text("✓", style="green"),
                Text(label),
                Text(result, style="dim"),
                Text(f"{secs:.1f}s", style="dim"),
            )
        if active is not None:
            rows.add_row(
                Spinner("dots", style="cyan"),
                Text(active, style="bold"),
                Text("…", style="dim"),
                Text(""),
            )

        body = Group(progress, Text(""), rows)
        return Panel(
            body,
            border_style="cyan",
            title="[bold]wind-forecast[/]",
            title_align="left",
            subtitle=f"[dim]{len(completed)}/{len(PHASES)} done[/]",
            subtitle_align="right",
        )

    with Live(render(PHASES[0][0]), console=console, refresh_per_second=12) as live:
        for label, dur in PHASES:
            phase_start = time.monotonic()
            live.update(render(label))
            time.sleep(dur)
            completed.append((label, PHASE_RESULTS[label], time.monotonic() - phase_start))
            progress.update(
                task_id,
                advance=1,
                elapsed=f"{time.monotonic() - started:.1f}s",
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
