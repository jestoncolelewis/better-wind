# better-wind

Site-specific probabilistic wind forecasting for small airports. One codebase,
per-airport YAML configs, per-airport trained models. Learns the residual bias
of HRRR/TAF wind guidance instead of predicting raw wind.

See [CLAUDE.md](CLAUDE.md) for the full spec and phased build plan. Phase 1
(data pipeline + config plumbing) and Phase 2 (deterministic baselines +
evaluation harness) are implemented today.

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

# Install `wind-forecast` onto your PATH. `--editable` means the install
# tracks the source tree — `git pull` is enough to pick up new versions.
uv tool install --editable .

# One-time: tell uv to add ~/.local/bin to your shell's PATH if it isn't
# already there.
uv tool update-shell
```

After this, `wind-forecast --help` works from any directory. CLI commands
still default to `./config/airports/` and `./data/` (relative paths), so
run them from the repo root unless you pass `--config-dir` / `--data-root`.

For development (tests, lint, type-check) also create the local dev venv:

```bash
uv sync --extra dev
```

That puts `pytest`, `ruff`, `mypy`, and `matplotlib` in `.venv/`. Run them
with `uv run pytest`, `uv run ruff check src`, etc.

## Verify it's working

1. **List configured airports.** Proves the config loader and `--airport`
   plumbing work end-to-end.
   ```bash
   wind-forecast list-airports
   ```
   Expected output:
   ```
   logging to logs/wind-forecast-20260425T210000Z.log
   KBOI  Boise Air Terminal / Gowen Field  (43.5644, -116.2228)
   KMAN  Nampa Municipal Airport  (43.5817, -116.5225)
   ```

2. **Run the test suite.** 67 tests; should pass in under a second. Covers
   config validation, METAR schema stability across stations, wind u/v
   round-trips, cycle iteration, path partitioning, date chunking, logging
   setup, plus the Phase 2 eval harness (HRRR↔obs pairing, baselines, and
   metrics).
   ```bash
   uv run pytest
   ```

3. **See the CLI surface.**
   ```bash
   wind-forecast --help
   wind-forecast run          --help
   wind-forecast ingest-metar --help
   wind-forecast ingest-hrrr  --help
   wind-forecast eval         --help
   ```

## Run the whole pipeline in one shot

```bash
wind-forecast run --airport KMAN
```

`run` is the easy button: it ingests METAR, ingests HRRR, then scores the
baselines, all for one airport. Date range defaults to `history_start` from
the YAML through "today UTC". A live ribbon panel reports each phase as
it goes; the final eval table prints at the end.

Common variations:

```bash
wind-forecast run --airport KMAN --start 2024-01-01 --end 2024-02-01
wind-forecast run --airport KMAN --no-eval                  # ingest only
wind-forecast run --airport KMAN --by-lead                  # detailed eval table
```

The individual commands below give you finer control if you only need one
piece of the pipeline (e.g. re-running just the eval after a config tweak).

## Pull METAR observations (ground truth)

```bash
wind-forecast ingest-metar --airport KMAN
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
wind-forecast ingest-hrrr \
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
| `--cycle-workers N` | 4 | Cycles fetched concurrently |
| `--workers N` | 8 | Parallel lead-hour fetches **within** one cycle |
| `--lead-min` / `--lead-max` | 1 / 18 | Forecast-hour range to pull |
| `--step-hours N` | 1 | Skip cycles (e.g. 3 = every 3rd init) |
| `--grid-half N` | 2 | `(2N+1) × (2N+1)` box around the airport |
| `--no-skip-existing` | off | Re-fetch cycles already on disk |

### Throughput: tuning the two worker pools

HRRR ingest has two tiers of parallelism. `--workers` parallelizes the
~18 lead-hour fetches **inside one cycle** (capped by the lead-hour count, so
values above ~18 don't help). `--cycle-workers` runs multiple cycles
concurrently. Peak in-flight lead fetches ≈ `cycle-workers × workers`, so the
defaults (4 × 8 = 32) is what to expect against your bandwidth and CPU.

For a multi-year backfill, `--cycle-workers` is the knob that actually moves
wall time. Each cycle is ~30–60 s end-to-end (download + GRIB decode + Parquet
write); a 48 000-cycle backfill at default 4 concurrent cycles is roughly
4–10 days, while raising it to `--cycle-workers 16` cuts that to roughly
1–3 days. Suggested presets:

| Scenario | Suggested flags |
|---|---|
| Quick sanity check, modest laptop | `--cycle-workers 1 --workers 8` |
| Default workstation backfill | `--cycle-workers 4 --workers 8` (default) |
| Beefy machine, fat pipe | `--cycle-workers 16 --workers 8` |
| Hitting AWS rate limits / errors | back off `--cycle-workers` first |

If you see repeated `HRRR fetch failed` warnings or your disk can't keep up,
lower `--cycle-workers` before touching `--workers`. Re-runs skip cycles
already on disk, so it's always safe to interrupt and resume with a different
concurrency setting.

> Start with a **short window first** (e.g. one day) to confirm your bandwidth
> and disk can keep up. The full spec targets years of history.

## Logging & progress

Every long-running CLI command renders a **live ribbon panel** to stderr:
completed phases stay visible with their result + timing, the active phase
shows a spinner with sub-progress (cycles, chunks, files), pending phases
are listed dim, and `eval` adds a ranking row sorting baselines by RMSE
once they've all scored. Each command's footer carries the log file path,
RSS, and pid.

Non-interactive runs (CI, redirected stderr) skip the ribbon entirely so
your build logs stay clean.

Independent of the panel, every invocation writes a **full DEBUG log** to
`logs/wind-forecast-<UTCtimestamp>Z.log`. That's the source of truth when
something goes wrong — open it in another terminal with `tail -f
logs/wind-forecast-*.log`.

Common controls (all on the top-level command, before the subcommand):

```bash
wind-forecast -v   ingest-metar --airport KMAN          # console: INFO
wind-forecast -vv  ingest-hrrr  --airport KMAN ...      # console: DEBUG
wind-forecast --log-file run.log ingest-metar --airport KMAN
```

The `logs/` directory is gitignored.

## Inspect what landed

```bash
find data/raw -type f | head
uv run python -c "import pandas as pd; print(pd.read_parquet('data/raw/metar/KMAN/KMAN.parquet').head())"
uv run python -c "import pandas as pd; print(pd.read_parquet('data/raw/hrrr/KMAN/2024/20240101_00Z.parquet').head())"
```

`uv run python` uses the local `.venv/` (from `uv sync`) where pandas and
pyarrow live — plain `python3` from your shell won't have them.

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
wind-forecast list-airports               # new airport appears
wind-forecast ingest-metar --airport KSUN
wind-forecast ingest-hrrr  --airport KSUN --start 2024-01-01T00:00Z --end 2024-01-08T00:00Z
```

Required YAML fields: `icao`, `name`, `latitude`, `longitude`, `elevation_ft`,
`timezone`, and at least one runway. `neighbor_stations` is optional but
strongly recommended — those are the upstream ASOS/AWOS stations the feature
builder will use for tendency signals.

## Score the deterministic baselines (Phase 2)

Once you've ingested both METAR and HRRR for an airport, score the baselines
on a chronological 70/15/15 train/val/test split:

```bash
wind-forecast eval --airport KMAN                 # overall summary
wind-forecast eval --airport KMAN --by-lead       # one row per forecast hour
wind-forecast eval --airport KMAN --baseline climatology
```

Three baselines run by default:

- **`persistence`** — last METAR obs at cycle init time, carried forward to
  every lead hour.
- **`hrrr`** — the unmodified HRRR 10 m wind + surface gust at the airport's
  nearest grid point.
- **`climatology`** — HRRR minus the per-`(lead_hour, hour_of_day, month)`
  mean bias learned on the training split. Captures the simple diurnal /
  seasonal corrections the LightGBM model in Phase 3 has to beat.

The output table reports RMSE/MAE on `u`, `v`, scalar speed, gust; circular
direction MAE (masked at 3 kt observed wind); and CRPS on speed (which is just
MAE for deterministic baselines — the column lights up properly once Phase 3
adds quantile predictions).

## Running everything across every configured airport

```bash
make ingest-all     # loops ingest-metar + ingest-hrrr for each YAML in config/airports/
make eval-baselines # runs `wind-forecast eval --baseline all` for each airport
make train-all      # (phase 3) trains per-airport models
make eval-all       # (phase 3) evaluates per-airport models
```

## Project layout

```
better-wind/
├── pyproject.toml              # deps, ruff/mypy/pytest config, `wind-forecast` script
├── Makefile                    # ingest-all / train-all / eval-all / lint / test
├── CLAUDE.md                   # full spec + phased build plan
├── config/airports/            # one YAML per airport — the only place coords live
│   ├── KMAN.yaml
│   └── KBOI.yaml
├── src/wind_forecast/
│   ├── config.py               # Airport pydantic model + loader
│   ├── winds.py                # (u, v) <-> (direction-from, speed)
│   ├── logging_setup.py        # console + file logging, used by every CLI command
│   ├── progress.py             # rich ribbon panels for ingest + eval (shared)
│   ├── cli.py                  # `wind-forecast` click entry points
│   ├── ingest/
│   │   ├── metar.py            # Iowa Mesonet bulk downloader (chunked + parallel)
│   │   └── hrrr.py             # herbie wrapper, 5×5 grid extraction (parallel leads)
│   └── eval/
│       ├── io.py               # load HRRR + METAR, pair on valid_utc / cycle_utc
│       ├── metrics.py          # RMSE / MAE / circular dir error / CRPS
│       ├── baselines.py        # persistence, raw HRRR, climatological bias
│       └── harness.py          # chronological split + table formatter
├── data/                       # gitignored; all data outputs land here
│   └── raw/
│       ├── metar/{ICAO}/{STATION}.parquet
│       └── hrrr/{ICAO}/{YYYY}/{YYYYMMDD_HHZ}.parquet
├── logs/                       # gitignored; one log file per CLI invocation
├── notebooks/
│   └── 01_data_eda.ipynb       # wind rose, diurnal cycle, HRRR bias stub
└── tests/                      # 67 tests covering config, schema, winds, chunks, logging, eval
```

## Developer loop

```bash
uv run pytest                   # run tests
uv run ruff check src tests     # lint
uv run mypy --strict src        # type-check
```

All three must pass before commits.

## What's next

Phase 3 (LightGBM residual model) is the next milestone. See
[CLAUDE.md](CLAUDE.md) for the full roadmap — the target is to beat raw HRRR
by ≥15% RMSE on scalar wind speed at forecast hour 6 for KMAN, then confirm
the same pipeline beats raw HRRR at KBOI.
