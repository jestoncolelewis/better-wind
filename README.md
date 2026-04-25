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
   KBOI  Boise Air Terminal / Gowen Field  (43.5644, -116.2228)
   KMAN  Nampa Municipal Airport  (43.5817, -116.5225)
   ```

2. **Run the test suite.** 25 tests; should pass in under a second. Covers
   config validation, METAR schema stability across stations, wind u/v
   round-trips, cycle iteration, and path partitioning.
   ```bash
   uv run pytest
   ```

3. **See the CLI surface.**
   ```bash
   uv run wind-forecast --help
   uv run wind-forecast ingest-metar --help
   uv run wind-forecast ingest-hrrr  --help
   ```

## Pull data for the primary airport (KMAN)

METAR observations (ground truth, fast — one CSV per station):

```bash
uv run wind-forecast ingest-metar --airport KMAN
```

This hits the Iowa Mesonet ASOS endpoint once per station (`KMAN` plus every
entry in the airport's `neighbor_stations`) and writes one Parquet file per
station to `data/raw/metar/KMAN/`. The date range defaults to
`history_start` from the YAML through "today UTC"; override with
`--start YYYY-MM-DD --end YYYY-MM-DD`.

HRRR forecasts (predictors + baseline, **slow** — one GRIB fetch per cycle ×
lead × variable):

```bash
uv run wind-forecast ingest-hrrr \
    --airport KMAN \
    --start 2024-01-01T00:00Z \
    --end   2024-01-08T00:00Z
```

This iterates every hourly init cycle in `[start, end)`, pulls forecast hours
`+1..+18` for each cycle, extracts a 5×5 grid box around the airport for every
required HRRR variable (`10m u/v`, gust, 925/850mb u/v/T, 2m T/Td, psfc, mslp,
PBLH, CAPE, CIN), and writes one Parquet per cycle to
`data/raw/hrrr/KMAN/YYYY/YYYYMMDD_HHZ.parquet`.

> Start with a **short window first** (e.g. one week) to confirm your
> bandwidth and disk can keep up. The full spec targets years of history.

Re-runs are safe — existing cycle files are skipped. Pass
`--no-skip-existing` to force re-download.

## Verifying the data landed

```bash
find data/raw -type f | head
uv run python -c "import pandas as pd; print(pd.read_parquet('data/raw/metar/KMAN/KMAN.parquet').head())"
```

`uv run python` uses the project venv where pandas/pyarrow are installed —
plain `python3` from your shell won't have them.

You should see the canonical METAR schema:
`station, valid_utc, drct, sknt, gust, u, v, tmpf, dwpf, alti, mslp, vsby, metar`.
Every station under every airport produces the same columns and dtypes — this
is the Phase 1 acceptance criterion.

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
│   ├── cli.py                  # `wind-forecast` click entry points
│   └── ingest/
│       ├── metar.py            # Iowa Mesonet bulk downloader
│       └── hrrr.py             # herbie wrapper, 5x5 grid extraction
├── data/                       # gitignored; all outputs land here
│   └── raw/
│       ├── metar/{ICAO}/{STATION}.parquet
│       └── hrrr/{ICAO}/{YYYY}/{YYYYMMDD_HHZ}.parquet
├── notebooks/
│   └── 01_data_eda.ipynb       # wind rose, diurnal cycle, HRRR bias stub
└── tests/                      # config, schema stability, wind conversions
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
implemented. See [CLAUD.md](CLAUD.md) for the full roadmap — the target for
Phase 3 is to beat raw HRRR by ≥15% RMSE on scalar wind speed at forecast
hour 6 for KMAN.
