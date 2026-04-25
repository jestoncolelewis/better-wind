# better-wind

Site-specific probabilistic wind forecasting for small airports. One codebase,
per-airport YAML configs, per-airport trained models. Learns the residual bias
of HRRR/TAF wind guidance instead of predicting raw wind.

See [CLAUD.md](CLAUD.md) for the full spec and phased build plan. Phase 1
(data pipeline + config plumbing) is what's implemented today.

## Prerequisites

- **Python 3.11** (pinned; other minor versions are not tested)
- **[uv](https://docs.astral.sh/uv/)** for dependency management. Install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Network access to:
  - `mesonet.agron.iastate.edu` (METAR bulk download)
  - NOAA HRRR on AWS Open Data (fetched through `herbie`)

No GPU required. Everything runs on a single workstation.

## Install

```bash
git clone https://github.com/jestoncolelewis/better-wind.git
cd better-wind

# Create a .venv and install runtime + dev deps (pytest, ruff, mypy, matplotlib)
uv sync --extra dev
```

`uv` creates `.venv/` in the repo root and resolves all deps from
`pyproject.toml`. Every command below is run through `uv run …` so you never
have to activate the venv manually.

## Verify it's working

1. **List configured airports.** Proves the config loader and `--airport`
   plumbing work end-to-end.
   ```bash
   uv run wind-forecast list-airports
   ```
   Expected output:
   ```
   logging to logs/wind-forecast-20260425T210000Z.log
   KBOI  Boise Air Terminal / Gowen Field  (43.5644, -116.2228)
   KMAN  Nampa Municipal Airport  (43.5817, -116.5225)
   ```

2. **Run the test suite.** 32 tests; should pass in under a second. Covers
   config validation, METAR schema stability across stations, wind u/v
   round-trips, cycle iteration, path partitioning, date chunking, and
   logging setup.
   ```bash
   uv run pytest
   ```

3. **See the CLI surface.**
   ```bash
   uv run wind-forecast --help
   uv run wind-forecast ingest-metar --help
   uv run wind-forecast ingest-hrrr  --help
   ```

## Pull METAR observations (ground truth)

```bash
uv run wind-forecast ingest-metar --airport KMAN
```

This hits the Iowa Mesonet ASOS endpoint for every station (`KMAN` plus every
entry in `neighbor_stations`), splits each station's date range into yearly
chunks, fetches all `(station, chunk)` pairs in parallel, and writes one
Parquet file per station to `data/raw/metar/KMAN/`. The date range defaults
to `history_start` from the YAML through "today UTC"; override with
`--start YYYY-MM-DD --end YYYY-MM-DD`.

Tuning knobs (defaults in parens):

| Flag | Default | What it does |
|---|---|---|
| `--workers N` | 4 | Parallel `(station, chunk)` requests |
| `--chunk-days N` | 366 | Days per request — Mesonet handles many small requests far better than one giant one |
| `--no-skip-existing` | off | Force re-fetch of stations that already have a Parquet on disk |

If Mesonet rate-limits you (HTTP 429), drop `--workers` to 2. If a single
chunk keeps timing out, shrink it: `--chunk-days 180`.

## Pull HRRR forecasts (predictors + baseline)

```bash
uv run wind-forecast ingest-hrrr \
    --airport KMAN \
    --start 2024-01-01T00:00Z \
    --end   2024-01-08T00:00Z
```

This iterates every hourly init cycle in `[start, end)`, pulls forecast hours
`+1..+18` for each cycle (in parallel), extracts a 5×5 grid box around the
airport for every required HRRR variable (`10m u/v`, gust, 925/850mb u/v/T,
2m T/Td, psfc, mslp, PBLH, CAPE, CIN), and writes one Parquet per cycle to
`data/raw/hrrr/KMAN/YYYY/YYYYMMDD_HHZ.parquet`.

Tuning knobs:

| Flag | Default | What it does |
|---|---|---|
| `--workers N` | 8 | Parallel lead-hour fetches per cycle |
| `--lead-min` / `--lead-max` | 1 / 18 | Forecast-hour range to pull |
| `--step-hours N` | 1 | Skip cycles (e.g. 3 = every 3rd init) |
| `--grid-half N` | 2 | `(2N+1) × (2N+1)` box around the airport |
| `--no-skip-existing` | off | Re-fetch cycles already on disk |

> Start with a **short window first** (e.g. one day) to confirm your bandwidth
> and disk can keep up. The full spec targets years of history. Re-runs skip
> cycles already on disk, so it's safe to interrupt and resume.

## Logging & progress

Every CLI invocation:

- Prints **only a progress bar plus warnings/errors** to the console. Per-task
  detail (chunk row counts, retry attempts, cycle timings) goes to a file.
- Writes a **full DEBUG log** to `logs/wind-forecast-<UTCtimestamp>Z.log`.
  This is the source of truth when something goes wrong — open it in another
  terminal with `tail -f logs/wind-forecast-*.log`.

Common controls (all on the top-level command, before the subcommand):

```bash
uv run wind-forecast -v   ingest-metar --airport KMAN          # console: INFO
uv run wind-forecast -vv  ingest-hrrr  --airport KMAN ...      # console: DEBUG
uv run wind-forecast --log-file run.log ingest-metar --airport KMAN
```

The `logs/` directory is gitignored.

## Inspect what landed

```bash
find data/raw -type f | head
uv run python -c "import pandas as pd; print(pd.read_parquet('data/raw/metar/KMAN/KMAN.parquet').head())"
uv run python -c "import pandas as pd; print(pd.read_parquet('data/raw/hrrr/KMAN/2024/20240101_00Z.parquet').head())"
```

`uv run python` uses the project venv where pandas/pyarrow are installed —
plain `python3` from your shell won't have them.

Canonical METAR schema (every station, every airport — this is the Phase 1
acceptance criterion):

```
station, valid_utc, drct, sknt, gust, u, v, tmpf, dwpf, alti, mslp, vsby, metar
```

Canonical HRRR schema:

```
cycle_utc, lead_hour, valid_utc, iy, ix, latitude, longitude,
u10, v10, gust, u925, v925, t925, u850, v850, t850,
t2m, d2m, psfc, mslp, pblh, cape, cin
```

The 5×5 grid box gives 25 rows per `(cycle, lead)`. The airport's nearest
grid point is `iy == ix == 2`.

## Adding a new airport

No code changes. Drop a YAML file under `config/airports/<ICAO>.yaml` (use
`KMAN.yaml` or `KBOI.yaml` as a template), then:

```bash
uv run wind-forecast list-airports               # new airport appears
uv run wind-forecast ingest-metar --airport KSUN
uv run wind-forecast ingest-hrrr  --airport KSUN --start 2024-01-01T00:00Z --end 2024-01-08T00:00Z
```

Required YAML fields: `icao`, `name`, `latitude`, `longitude`, `elevation_ft`,
`timezone`, and at least one runway. `neighbor_stations` is optional but
strongly recommended — those are the upstream ASOS/AWOS stations the feature
builder will use for tendency signals.

## Running everything across every configured airport

```bash
make ingest-all     # loops ingest-metar + ingest-hrrr for each YAML in config/airports/
make train-all      # (phase 3) trains per-airport models
make eval-all       # (phase 3) evaluates per-airport models
```

## Project layout

```
better-wind/
├── pyproject.toml              # deps, ruff/mypy/pytest config, `wind-forecast` script
├── Makefile                    # ingest-all / train-all / eval-all / lint / test
├── CLAUD.md                    # full spec + phased build plan
├── config/airports/            # one YAML per airport — the only place coords live
│   ├── KMAN.yaml
│   └── KBOI.yaml
├── src/wind_forecast/
│   ├── config.py               # Airport pydantic model + loader
│   ├── winds.py                # (u, v) <-> (direction-from, speed)
│   ├── logging_setup.py        # console + file logging, used by every CLI command
│   ├── cli.py                  # `wind-forecast` click entry points
│   └── ingest/
│       ├── metar.py            # Iowa Mesonet bulk downloader (chunked + parallel)
│       └── hrrr.py             # herbie wrapper, 5×5 grid extraction (parallel leads)
├── data/                       # gitignored; all data outputs land here
│   └── raw/
│       ├── metar/{ICAO}/{STATION}.parquet
│       └── hrrr/{ICAO}/{YYYY}/{YYYYMMDD_HHZ}.parquet
├── logs/                       # gitignored; one log file per CLI invocation
├── notebooks/
│   └── 01_data_eda.ipynb       # wind rose, diurnal cycle, HRRR bias stub
└── tests/                      # 32 tests covering config, schema, winds, chunks, logging
```

## Developer loop

```bash
uv run pytest                   # run tests
uv run ruff check src tests     # lint
uv run mypy --strict src        # type-check
```

All three must pass before commits.

## What's next

Phase 2 (baselines) and Phase 3 (LightGBM residual model) are not yet
implemented. See [CLAUDE.md](CLAUDE.md) for the full roadmap — the target for
Phase 3 is to beat raw HRRR by ≥15% RMSE on scalar wind speed at forecast
hour 6 for KMAN.
