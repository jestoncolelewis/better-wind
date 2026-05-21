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
# Full-width panel pinned at the bottom. Most "app-like", a bit heavy for a CLI.
def demo_ribbon() -> None:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.table import Table

    console = Console()

    def render(phase: str, done: int) -> Panel:
        t = Table.grid(padding=(0, 1))
        t.add_row(Spinner("dots", style="cyan"), f"[bold]eval KMAN[/] · {phase}")
        t.add_row("", f"[dim]{done}/{len(PHASES)} steps[/]")
        return Panel(t, border_style="cyan", title="wind-forecast", title_align="left")

    with Live(render("starting", 0), console=console, refresh_per_second=12) as live:
        for i, (label, dur) in enumerate(PHASES, 1):
            live.update(render(label, i - 1))
            time.sleep(dur)
        live.update(render("done", len(PHASES)))


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
