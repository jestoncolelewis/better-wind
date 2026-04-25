"""Command-line entry points.

Phase 1 surfaces `ingest-metar` and `ingest-hrrr`. Phase 2 adds `eval`, which
runs the deterministic baselines (persistence, raw HRRR, climatological bias
correction) for one airport. Later phases wire up `build-features`, `train`,
and `predict` using the same `--airport` convention.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

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
def ingest_metar_cmd(
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
    from wind_forecast.ingest import metar as metar_ingest  # deferred

    airport = Airport.load(airport_icao, config_dir)
    written = metar_ingest.ingest_airport(
        airport,
        start=start,
        end=end,
        data_root=data_root,
        max_workers=workers,
        chunk_days=chunk_days,
        skip_existing=not no_skip_existing,
    )
    for station, path in written.items():
        click.echo(f"{station}: {path}")


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
    help="Parallel lead-hour fetches per cycle.",
)
@click.option("--no-skip-existing", is_flag=True, help="Re-fetch cycles already on disk.")
@config_dir_option
@data_root_option
def ingest_hrrr_cmd(
    airport_icao: str,
    start: datetime,
    end: datetime,
    lead_min: int,
    lead_max: int,
    step_hours: int,
    grid_half: int,
    workers: int,
    no_skip_existing: bool,
    config_dir: Path,
    data_root: Path,
) -> None:
    """Pull HRRR cycles and cache per-cycle Parquets."""
    from wind_forecast.ingest import hrrr as hrrr_ingest  # deferred

    airport = Airport.load(airport_icao, config_dir)
    paths = hrrr_ingest.ingest_airport(
        airport,
        start=start,
        end=end,
        lead_hours=range(lead_min, lead_max + 1),
        grid_half=grid_half,
        cycle_step_hours=step_hours,
        skip_existing=not no_skip_existing,
        data_root=data_root,
        max_workers=workers,
    )
    click.echo(f"{len(paths)} cycles")


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
def eval_cmd(
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
    from wind_forecast.eval.harness import evaluate_airport, format_table

    airport = Airport.load(airport_icao, config_dir)
    names = bl.ALL_BASELINES if baseline_choice == "all" else (baseline_choice,)
    metrics = evaluate_airport(
        airport,
        data_root=data_root,
        baselines=names,
        train_frac=train_frac,
        val_frac=val_frac,
    )
    click.echo(format_table(metrics, by_lead=by_lead))


if __name__ == "__main__":
    cli()
