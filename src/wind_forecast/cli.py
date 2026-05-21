"""Command-line entry points.

Phase 1 surfaces `ingest-metar` and `ingest-hrrr`. Phase 2 adds `eval`, which
runs the deterministic baselines (persistence, raw HRRR, climatological bias
correction) for one airport. `run` glues all three together for a one-shot
"do the whole pipeline" command. Later phases wire up `build-features`,
`train`, and `predict` using the same `--airport` convention.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import click

from wind_forecast.config import DEFAULT_CONFIG_DIR, DEFAULT_DATA_ROOT, Airport
from wind_forecast.logging_setup import setup_logging

log = logging.getLogger("wind_forecast")


def _parse_date(ctx: click.Context, param: click.Parameter, value: str | None) -> date | None:
    del ctx, param
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise click.BadParameter(f"expected YYYY-MM-DD, got {value!r}") from e


def _parse_datetime(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> datetime | None:
    del ctx, param
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise click.BadParameter(f"expected ISO datetime, got {value!r}") from e


airport_option = click.option(
    "--airport",
    "airport_icao",
    required=True,
    help="ICAO code of a configured airport (e.g. KMAN).",
)
config_dir_option = click.option(
    "--config-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_CONFIG_DIR,
    show_default=True,
)
data_root_option = click.option(
    "--data-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_DATA_ROOT,
    show_default=True,
)


@click.group()
@click.option(
    "-v", "--verbose", count=True,
    help="Console verbosity: -v = INFO, -vv = DEBUG. Default WARNING.",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Write the full DEBUG log to this file (default: logs/wind-forecast-{timestamp}Z.log).",
)
@click.pass_context
def cli(ctx: click.Context, verbose: int, log_file: Path | None) -> None:
    """Site-specific wind forecasting — data pipeline + baseline evaluation."""
    log_path = setup_logging(verbose=verbose, log_file=log_file)
    click.echo(f"logging to {log_path}", err=True)
    ctx.ensure_object(dict)
    ctx.obj["log_path"] = log_path


@cli.command("list-airports")
@config_dir_option
def list_airports(config_dir: Path) -> None:
    """List every configured airport."""
    airports = Airport.list_all(config_dir)
    if not airports:
        click.echo(f"No airport configs found under {config_dir}")
        return
    for a in airports:
        click.echo(f"{a.icao}  {a.name}  ({a.latitude:.4f}, {a.longitude:.4f})")


# ── ingest-metar ───────────────────────────────────────────────────────────
def _run_ingest_metar(
    airport: Airport,
    *,
    start: date | None,
    end: date | None,
    workers: int,
    chunk_days: int,
    skip_existing: bool,
    data_root: Path,
    log_path: Path | str,
) -> dict[str, Path]:
    """Shared implementation used by both `ingest-metar` and `run`."""
    from wind_forecast.ingest import metar as metar_ingest  # deferred
    from wind_forecast.progress import IngestMetarRibbon, maybe_ribbon

    with maybe_ribbon(
        lambda: IngestMetarRibbon(airport_icao=airport.icao, log_path=log_path),
    ) as ribbon:
        if ribbon is not None:
            ribbon.begin(IngestMetarRibbon.PHASE)

        def _on_progress(done: int, total: int, counters: dict[str, Any]) -> None:
            if ribbon is None:
                return
            summary = (
                f"{int(counters.get('rows', 0)):,} rows · "
                f"{int(counters.get('failed', 0))} failed"
            )
            ribbon.sub_progress(done, total, summary=summary)

        written = metar_ingest.ingest_airport(
            airport,
            start=start,
            end=end,
            data_root=data_root,
            max_workers=workers,
            chunk_days=chunk_days,
            skip_existing=skip_existing,
            on_progress=_on_progress,
        )

        if ribbon is not None:
            ribbon.end_phase(
                IngestMetarRibbon.PHASE,
                {
                    "stations": len(written),
                    "rows": sum(_parquet_row_count(p) for p in written.values()),
                },
            )
    return written


@cli.command("ingest-metar")
@airport_option
@click.option("--start", callback=_parse_date, help="YYYY-MM-DD, default: airport.history_start.")
@click.option("--end", callback=_parse_date, help="YYYY-MM-DD, default: today UTC.")
@click.option(
    "--workers", type=int, default=4, show_default=True,
    help="Parallel (station, chunk) fetches.",
)
@click.option(
    "--chunk-days", type=int, default=366, show_default=True,
    help="Split each station's range into chunks at most this many days long.",
)
@click.option("--no-skip-existing", is_flag=True, help="Re-fetch stations already on disk.")
@config_dir_option
@data_root_option
@click.pass_context
def ingest_metar_cmd(
    ctx: click.Context,
    airport_icao: str,
    start: date | None,
    end: date | None,
    workers: int,
    chunk_days: int,
    no_skip_existing: bool,
    config_dir: Path,
    data_root: Path,
) -> None:
    """Download METAR obs for the airport + every neighbor station."""
    airport = Airport.load(airport_icao, config_dir)
    written = _run_ingest_metar(
        airport,
        start=start,
        end=end,
        workers=workers,
        chunk_days=chunk_days,
        skip_existing=not no_skip_existing,
        data_root=data_root,
        log_path=ctx.obj.get("log_path", ""),
    )
    for station, path in written.items():
        click.echo(f"{station}: {path}")


# ── ingest-hrrr ────────────────────────────────────────────────────────────
def _run_ingest_hrrr(
    airport: Airport,
    *,
    start: datetime,
    end: datetime,
    lead_min: int,
    lead_max: int,
    step_hours: int,
    grid_half: int,
    workers: int,
    cycle_workers: int,
    skip_existing: bool,
    data_root: Path,
    log_path: Path | str,
) -> Any:
    from wind_forecast.ingest import hrrr as hrrr_ingest  # deferred
    from wind_forecast.progress import IngestHrrrRibbon, maybe_ribbon

    with maybe_ribbon(
        lambda: IngestHrrrRibbon(airport_icao=airport.icao, log_path=log_path),
    ) as ribbon:
        if ribbon is not None:
            ribbon.begin(IngestHrrrRibbon.PHASE)

        def _on_progress(done: int, total: int, counters: dict[str, Any]) -> None:
            if ribbon is None:
                return
            summary = (
                f"{int(counters.get('written', 0))} written · "
                f"{int(counters.get('skipped', 0))} skipped · "
                f"{int(counters.get('failed', 0))} failed · "
                f"{int(counters.get('empty', 0))} empty"
            )
            ribbon.sub_progress(done, total, summary=summary)

        summary = hrrr_ingest.ingest_airport(
            airport,
            start=start,
            end=end,
            lead_hours=range(lead_min, lead_max + 1),
            grid_half=grid_half,
            cycle_step_hours=step_hours,
            skip_existing=skip_existing,
            data_root=data_root,
            max_workers=workers,
            cycle_workers=cycle_workers,
            on_progress=_on_progress,
        )

        if ribbon is not None:
            ribbon.end_phase(
                IngestHrrrRibbon.PHASE,
                {
                    "written": summary.written,
                    "skipped": summary.skipped,
                    "failed": summary.failed,
                    "empty": summary.empty,
                    "rows": summary.rows,
                },
            )
    return summary


@cli.command("ingest-hrrr")
@airport_option
@click.option(
    "--start",
    callback=_parse_datetime,
    required=True,
    help="First init cycle, ISO (e.g. 2024-01-01T00:00Z).",
)
@click.option(
    "--end",
    callback=_parse_datetime,
    required=True,
    help="Upper bound init cycle (exclusive).",
)
@click.option("--lead-min", type=int, default=1, show_default=True)
@click.option("--lead-max", type=int, default=18, show_default=True)
@click.option("--step-hours", type=int, default=1, show_default=True)
@click.option("--grid-half", type=int, default=2, show_default=True)
@click.option(
    "--workers", type=int, default=8, show_default=True,
    help="Parallel lead-hour fetches within one cycle (max useful ≈ leads per cycle).",
)
@click.option(
    "--cycle-workers", type=int, default=4, show_default=True,
    help="Cycles fetched in parallel. Peak in-flight requests ≈ cycle-workers × workers.",
)
@click.option("--no-skip-existing", is_flag=True, help="Re-fetch cycles already on disk.")
@config_dir_option
@data_root_option
@click.pass_context
def ingest_hrrr_cmd(
    ctx: click.Context,
    airport_icao: str,
    start: datetime,
    end: datetime,
    lead_min: int,
    lead_max: int,
    step_hours: int,
    grid_half: int,
    workers: int,
    cycle_workers: int,
    no_skip_existing: bool,
    config_dir: Path,
    data_root: Path,
) -> None:
    """Pull HRRR cycles and cache per-cycle Parquets."""
    airport = Airport.load(airport_icao, config_dir)
    summary = _run_ingest_hrrr(
        airport,
        start=start,
        end=end,
        lead_min=lead_min,
        lead_max=lead_max,
        step_hours=step_hours,
        grid_half=grid_half,
        workers=workers,
        cycle_workers=cycle_workers,
        skip_existing=not no_skip_existing,
        data_root=data_root,
        log_path=ctx.obj.get("log_path", ""),
    )
    click.echo(
        f"HRRR {airport.icao}: {summary.total} cycles "
        f"({summary.written} new, {summary.skipped} skipped, "
        f"{summary.failed} failed, {summary.empty} empty, "
        f"{summary.rows:,} rows)"
    )


# ── eval ───────────────────────────────────────────────────────────────────
def _run_eval(
    airport: Airport,
    *,
    baseline_names: list[str],
    train_frac: float,
    val_frac: float,
    data_root: Path,
    log_path: Path | str,
) -> Any:
    from wind_forecast.eval.harness import evaluate_airport
    from wind_forecast.progress import EvalRibbon, maybe_ribbon

    with maybe_ribbon(
        lambda: EvalRibbon(
            airport_icao=airport.icao,
            baseline_names=baseline_names,
            log_path=log_path,
        ),
    ) as ribbon:
        return evaluate_airport(
            airport,
            data_root=data_root,
            baselines=baseline_names,
            train_frac=train_frac,
            val_frac=val_frac,
            on_phase_start=ribbon.begin if ribbon else None,
            on_phase_done=ribbon.end_phase if ribbon else None,
            on_load_progress=ribbon.sub_progress if ribbon else None,
        )


@cli.command("eval")
@airport_option
@click.option(
    "--baseline",
    "baseline_choice",
    type=click.Choice(["persistence", "hrrr", "climatology", "all"]),
    default="all",
    show_default=True,
    help="Which baseline to score. `all` runs every baseline.",
)
@click.option(
    "--by-lead",
    is_flag=True,
    help="Print one row per forecast hour instead of the overall summary.",
)
@click.option(
    "--train-frac", type=float, default=0.70, show_default=True,
    help="Fraction of unique cycles used for training.",
)
@click.option(
    "--val-frac", type=float, default=0.15, show_default=True,
    help="Fraction of unique cycles used for validation. Test = remainder.",
)
@config_dir_option
@data_root_option
@click.pass_context
def eval_cmd(
    ctx: click.Context,
    airport_icao: str,
    baseline_choice: str,
    by_lead: bool,
    train_frac: float,
    val_frac: float,
    config_dir: Path,
    data_root: Path,
) -> None:
    """Score deterministic baselines (persistence / HRRR / climatology) on the test split."""
    from wind_forecast.eval import baselines as bl  # deferred
    from wind_forecast.eval.harness import format_table

    airport = Airport.load(airport_icao, config_dir)
    names = list(bl.ALL_BASELINES if baseline_choice == "all" else (baseline_choice,))
    metrics = _run_eval(
        airport,
        baseline_names=names,
        train_frac=train_frac,
        val_frac=val_frac,
        data_root=data_root,
        log_path=ctx.obj.get("log_path", ""),
    )
    click.echo(format_table(metrics, by_lead=by_lead))


# ── run (one-shot: ingest-metar → ingest-hrrr → eval) ──────────────────────
@cli.command("run")
@airport_option
@click.option(
    "--start", callback=_parse_date,
    help="HRRR + METAR start date (YYYY-MM-DD). Default: airport.history_start.",
)
@click.option(
    "--end", callback=_parse_date,
    help="HRRR + METAR end date (YYYY-MM-DD). Default: today UTC.",
)
@click.option("--lead-min", type=int, default=1, show_default=True)
@click.option("--lead-max", type=int, default=18, show_default=True)
@click.option(
    "--metar-workers", type=int, default=4, show_default=True,
    help="Parallel METAR (station, chunk) fetches.",
)
@click.option(
    "--hrrr-workers", type=int, default=8, show_default=True,
    help="Parallel lead-hour fetches within one HRRR cycle.",
)
@click.option(
    "--hrrr-cycle-workers", type=int, default=4, show_default=True,
    help="HRRR cycles fetched in parallel.",
)
@click.option("--no-skip-existing", is_flag=True, help="Re-fetch data already on disk.")
@click.option("--no-eval", is_flag=True, help="Skip the eval step (ingest only).")
@click.option(
    "--by-lead", is_flag=True,
    help="Print one eval row per forecast hour instead of the overall summary.",
)
@config_dir_option
@data_root_option
@click.pass_context
def run_cmd(
    ctx: click.Context,
    airport_icao: str,
    start: date | None,
    end: date | None,
    lead_min: int,
    lead_max: int,
    metar_workers: int,
    hrrr_workers: int,
    hrrr_cycle_workers: int,
    no_skip_existing: bool,
    no_eval: bool,
    by_lead: bool,
    config_dir: Path,
    data_root: Path,
) -> None:
    """One-shot pipeline: ingest METAR, ingest HRRR, then score baselines.

    Defaults to the airport's `history_start → today UTC`. Each step opens
    its own ribbon; the final eval table prints at the end.
    """
    from wind_forecast.eval import baselines as bl
    from wind_forecast.eval.harness import format_table

    airport = Airport.load(airport_icao, config_dir)
    resolved_start = start or airport.history_start or date(2020, 1, 1)
    resolved_end = end or datetime.now(tz=UTC).date()
    log_path = ctx.obj.get("log_path", "")
    skip_existing = not no_skip_existing

    _run_ingest_metar(
        airport,
        start=resolved_start,
        end=resolved_end,
        workers=metar_workers,
        chunk_days=366,
        skip_existing=skip_existing,
        data_root=data_root,
        log_path=log_path,
    )

    _run_ingest_hrrr(
        airport,
        start=datetime.combine(resolved_start, datetime.min.time(), tzinfo=UTC),
        end=datetime.combine(resolved_end, datetime.min.time(), tzinfo=UTC),
        lead_min=lead_min,
        lead_max=lead_max,
        step_hours=1,
        grid_half=2,
        workers=hrrr_workers,
        cycle_workers=hrrr_cycle_workers,
        skip_existing=skip_existing,
        data_root=data_root,
        log_path=log_path,
    )

    if no_eval:
        return

    metrics = _run_eval(
        airport,
        baseline_names=list(bl.ALL_BASELINES),
        train_frac=0.70,
        val_frac=0.15,
        data_root=data_root,
        log_path=log_path,
    )
    click.echo(format_table(metrics, by_lead=by_lead))


# ── helpers ────────────────────────────────────────────────────────────────
def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return 0


if __name__ == "__main__":
    cli()
