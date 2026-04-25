"""Command-line entry points.

Phase 1 surfaces `ingest-metar` and `ingest-hrrr` only. Later phases wire up
`build-features`, `train`, `eval`, and `predict` using the same `--airport`
convention.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import click

from wind_forecast.config import DEFAULT_CONFIG_DIR, DEFAULT_DATA_ROOT, Airport

log = logging.getLogger("wind_forecast")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


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
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Site-specific wind forecasting — phase 1 data pipeline."""
    _setup_logging(verbose)
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
@config_dir_option
@data_root_option
def ingest_metar_cmd(
    airport_icao: str,
    start: date | None,
    end: date | None,
    config_dir: Path,
    data_root: Path,
) -> None:
    """Download METAR obs for the airport + every neighbor station."""
    from wind_forecast.ingest import metar as metar_ingest  # deferred

    airport = Airport.load(airport_icao, config_dir)
    written = metar_ingest.ingest_airport(
        airport, start=start, end=end, data_root=data_root
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


if __name__ == "__main__":
    cli()
