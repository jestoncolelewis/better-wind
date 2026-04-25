# Site-Specific Wind Forecasting

A project spec for building a probabilistic wind forecasting system that outperforms stock HRRR/TAF guidance by learning site-specific bias patterns. Designed to be **airport-swappable**: one codebase, per-airport configs, separate trained models.

**Primary airport:** KMAN (Nampa Municipal, ID).
**Also target:** KBOI, KSUN, KJAC, and any other airport with a YAML config dropped into `config/airports/`.

## Goals

1. Produce probabilistic forecasts of wind speed, gust, and direction for a configurable airport at forecast hours +1 through +18.
1. Beat raw HRRR and nearby TAF wind forecasts on RMSE and CRPS at the target airport.
1. Output operationally useful quantities: `P(crosswind > threshold on runway X)`, `P(gust > 25kt)`, 10th/50th/90th percentile wind.
1. Keep the whole pipeline reproducible and runnable on a single workstation (no cloud GPU required).
1. Adding a new airport should require **only a new YAML file**, not code changes.

## Non-goals (for v1)

- Nowcasting below +1h (needs radar/satellite pipeline).
- Convective/thunderstorm-specific outflow prediction.
- **Cross-airport generalization**: a single model that predicts well at airports it was never trained on. Deferred — requires terrain features, elevation embeddings, much larger training set, and careful cross-site validation.

## Architecture

```
                 ┌─ config/airports/KMAN.yaml
                 ├─ config/airports/KBOI.yaml
                 └─ config/airports/*.yaml
                            │
                            ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Data ingestion │ →  │  Feature builder │ →  │  Model training │
│  (METAR + HRRR) │    │  (residual target│    │  (LightGBM +    │
│   per airport   │    │   + features)    │    │   quantile)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │   Inference &   │
                                              │   evaluation    │
                                              │   per airport   │
                                              └─────────────────┘
```

Per-airport models. Same code. Config-driven.

## Tech stack

- **Language:** Python 3.11
- **Environment:** `uv` for dependency management
- **Data:** `herbie` (HRRR/RAP retrieval), `xarray` + `cfgrib` (GRIB2 handling), `pandas`/`polars`, `requests` (Iowa Mesonet METAR bulk download)
- **Modeling:** `lightgbm` for gradient boosting, `scikit-learn` for splits/metrics, `ngboost` optional for full probabilistic output
- **Evaluation:** `properscoring` (CRPS), custom runway crosswind utilities
- **Config:** `pyyaml` + `pydantic` for typed airport configs
- **Storage:** Parquet for features/labels, joblib for models
- **Orchestration:** Plain Python scripts + Makefile. No Airflow/Prefect for v1.

## Project structure

```
wind-forecast/
├── pyproject.toml
├── Makefile
├── README.md
├── config/
│   └── airports/
│       ├── KMAN.yaml
│       ├── KBOI.yaml
│       └── KSUN.yaml
├── src/
│   ├── config.py            # Airport dataclass + loader
│   ├── ingest/
│   │   ├── metar.py         # Iowa Mesonet bulk downloader
│   │   └── hrrr.py          # herbie-based HRRR retrieval
│   ├── features/
│   │   ├── build.py         # pair forecasts to obs, compute residuals
│   │   ├── temporal.py      # hour, DOY, solar elevation
│   │   └── upstream.py      # tendencies at neighbor stations
│   ├── models/
│   │   ├── train.py         # LightGBM quantile regression
│   │   └── predict.py
│   ├── eval/
│   │   ├── metrics.py       # RMSE, MAE, CRPS, reliability
│   │   └── runway.py        # crosswind/headwind decomposition
│   └── cli.py               # entry points: ingest / build / train / eval / predict
├── data/                    # gitignored, structured by airport
│   ├── raw/
│   │   ├── metar/{airport}/
│   │   └── hrrr/{airport}/
│   ├── features/{airport}/
│   └── models/{airport}/
├── notebooks/               # EDA only, nothing production
└── tests/
    ├── test_config.py
    ├── test_features.py
    └── test_runway.py
```

## Airport config schema

Every airport is one YAML file under `config/airports/`. Example for KMAN:

```yaml
# config/airports/KMAN.yaml
icao: KMAN
name: Nampa Municipal Airport
latitude: 43.5817
longitude: -116.5225
elevation_ft: 2537
timezone: America/Boise

runways:
  - id: "11/29"
    heading_deg_true: 110   # heading of the 11 end
  # add more runways here for airports with multiple

# Nearby ASOS/AWOS stations used as upstream/neighbor features.
# Choose based on prevailing flow and terrain channeling.
neighbor_stations:
  - KBOI   # Boise, ~20nm NE, same valley
  - KEUL   # Caldwell, ~10nm W, same valley
  - KMUO   # Mountain Home AFB, ~45nm SE
  - KONO   # Ontario OR, ~45nm NW

# Optional: how far back to pull METAR history. Defaults to "max available".
history_start: "2015-01-01"
```

And the loader:

```python
# src/config.py
from pathlib import Path
from pydantic import BaseModel, Field
import yaml

class Runway(BaseModel):
    id: str
    heading_deg_true: int

class Airport(BaseModel):
    icao: str
    name: str
    latitude: float
    longitude: float
    elevation_ft: int
    timezone: str
    runways: list[Runway]
    neighbor_stations: list[str] = Field(default_factory=list)
    history_start: str | None = None

    @classmethod
    def load(cls, icao: str, config_dir: Path = Path("config/airports")) -> "Airport":
        path = config_dir / f"{icao.upper()}.yaml"
        with path.open() as f:
            return cls(**yaml.safe_load(f))

    def data_dir(self, root: Path = Path("data")) -> Path:
        return root / "features" / self.icao

    def model_dir(self, root: Path = Path("data")) -> Path:
        return root / "models" / self.icao
```

Every CLI command takes `--airport ICAO` and loads the config. No hardcoded coordinates anywhere in the pipeline.

## CLI surface

```bash
# Data pipeline
wind-forecast ingest-metar --airport KMAN
wind-forecast ingest-hrrr  --airport KMAN --start 2020-01-01 --end 2024-12-31
wind-forecast build-features --airport KMAN

# Training
wind-forecast train --airport KMAN
wind-forecast train --airport KBOI

# Evaluation
wind-forecast eval --airport KMAN
wind-forecast eval --airport KMAN --baseline hrrr

# Inference
wind-forecast predict --airport KMAN --cycle 2026-04-24T12Z --lead 6
```

A convenience target: `make train-all` iterates every YAML under `config/airports/` and trains a model per airport.

## Data sources

### METAR observations (ground truth)

- Iowa State Mesonet bulk download: `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`
- Stations: target airport + everything listed in `neighbor_stations`
- Fields: `drct`, `sknt`, `gust`, `tmpf`, `dwpf`, `alti`, `mslp`, `vsby`, `metar`
- Range: from `history_start` (or max available) to present

### HRRR forecasts (predictors + baseline)

- NOAA Open Data on AWS, retrieved via `herbie`
- Cycles: every hour, forecast hours 0–18
- Grid extraction: nearest 5×5 points around the airport’s lat/lon
- Variables needed:
  - 10m u/v wind, 10m gust
  - 925mb and 850mb u/v wind, temperature
  - 2m temperature, 2m dewpoint
  - Surface pressure, MSLP
  - PBL height, surface CAPE, surface CIN
  - Low-level (0–3km) lapse rate (derived)

### RAP (optional, v1.5)

- Same pipeline as HRRR, coarser grid, longer history. Useful for training data volume at airports with short HRRR archives.

## Target variable

**Do not predict raw wind.** Predict the HRRR residual:

```
residual_u = observed_u - hrrr_forecast_u
residual_v = observed_v - hrrr_forecast_v
residual_gust = observed_gust - hrrr_forecast_gust
```

At inference time, add the predicted residual back to the HRRR forecast. Train separate models per forecast hour (or use forecast hour as a feature — evaluate both).

Decompose direction as `(sin(θ), cos(θ))` on both observed and forecast before computing residuals. Never regress on degrees directly.

## Features

### From HRRR at the airport grid point

- Raw 10m u, v, gust (also the baseline)
- 925mb / 850mb u, v, vector shear (925–10m, 850–925)
- PBL height, CAPE, CIN
- 2m T, Td, T–Td spread (proxy for mixing)
- Surface pressure tendency (cycle-over-cycle)

### From HRRR at surrounding grid (spatial context)

- Mean and gradient of 10m wind across the 5×5 box
- Approximate convergence/divergence (∂u/∂x + ∂v/∂y)

### From recent METAR observations (persistence + trends)

- Most recent obs at the airport: u, v, gust
- 1h, 3h, 6h tendencies at the airport: Δu, Δv, Δpressure, Δtemperature
- Same tendencies at each `neighbor_stations` entry (upstream signal)
- Time since last observation (for gap handling)

### Temporal

- `hour_of_day_sin`, `hour_of_day_cos`
- `day_of_year_sin`, `day_of_year_cos`
- Solar elevation angle at the airport’s lat/lon (critical for diurnal valley winds)
- Forecast lead time (hour 1 vs hour 18)

### Regime indicators

- Crude synoptic regime from 500mb flow if accessible, else derived from surface pressure pattern
- Static stability proxy: (T_850 − T_2m) / height

### Explicitly NOT included in v1

- Elevation, terrain roughness, channeling angle. These would be required for a cross-airport model. Since we train per-airport, the model learns terrain effects implicitly through residual patterns.

## Modeling approach

### Baselines (build first, beat later)

- **Persistence**: last observed wind carried forward
- **Raw HRRR**: unchanged 10m wind from HRRR
- **HRRR + climatological bias correction**: subtract mean HRRR–obs error by hour-of-day and month

### Main model: LightGBM quantile regression (per airport)

- Train separate models for the 10th, 50th, 90th percentiles of each target
- Targets: `residual_u`, `residual_v`, `residual_gust`
- At inference, reconstruct:
  - Predicted wind speed = `sqrt((hrrr_u + Δu)^2 + (hrrr_v + Δv)^2)`
  - Predicted direction = `atan2(-(hrrr_u + Δu), -(hrrr_v + Δv))` (meteorological convention)
- Use `pinball_loss` for training; early stopping on validation fold
- Model artifacts saved to `data/models/{icao}/`

### Gust as a separate model

- Target: `residual_gust`, trained only on samples where observed gust exists
- Add predicted mean wind and stability features explicitly as inputs (gust factor is nonlinear in these)

### Probabilistic calibration

- Hold out last year as test; check that observed frequency matches nominal for 10/50/90 quantiles
- If miscalibrated, apply isotonic regression on validation quantiles — calibration fit saved per airport

## Evaluation

### Splits

- Chronological only. No random shuffling. Applied independently per airport.
- Train: oldest 70%
- Validation: next 15% (hyperparameters, early stopping, calibration)
- Test: most recent 15%, untouched until final evaluation

### Metrics (computed per airport)

- **RMSE and MAE** on u, v, gust, and scalar speed
- **Mean absolute direction error** (circular; use `min(|θ_pred − θ_obs|, 360 − |…|)`), masked for observed wind < 3kt (direction meaningless)
- **CRPS** from the quantile output
- **Reliability diagram** for predicted probability of crosswind exceeding thresholds
- **Skill score vs HRRR baseline**: `1 − RMSE_model / RMSE_hrrr` broken out by forecast hour and hour-of-day

### Operational check

- Compute `P(crosswind > 15kt on each runway)` over the test set
- Brier score vs observed crosswind exceedances

### Cross-airport report

- `make eval-all` runs evaluation for every configured airport and emits a summary table. Useful for spotting airports where the approach struggles (e.g., complex terrain, sparse data).

## Phased build plan

### Phase 1: Data pipeline + config plumbing (target: 1 week)

- [ ] `src/config.py`: `Airport` pydantic model + `load()` helper
- [ ] `config/airports/KMAN.yaml` + at least one other (KBOI) to prove multi-airport works
- [ ] `src/ingest/metar.py`: bulk download, parameterized by airport, write to `data/raw/metar/{icao}/`
- [ ] `src/ingest/hrrr.py`: `herbie` wrapper, parameterized by lat/lon, cache to `data/raw/hrrr/{icao}/`
- [ ] Sanity plots in `notebooks/01_data_eda.ipynb`: wind rose, diurnal cycle, HRRR raw bias — run for KMAN and KBOI, compare
- [ ] Tests: config loads correctly; feature builder produces no leakage (no future obs in features for a given forecast hour)

**Acceptance:** `wind-forecast ingest-metar --airport KMAN` and `--airport KBOI` both produce tidy Parquet outputs with identical schemas.

### Phase 2: Baselines (target: 2 days)

- [ ] Persistence, raw HRRR, and climatological-bias-correction baselines
- [ ] Evaluation harness producing RMSE/MAE/CRPS per forecast hour, keyed by airport

**Acceptance:** `make eval-baselines` prints a table of baseline performance for every configured airport.

### Phase 3: LightGBM residual model (target: 1 week)

- [ ] Quantile models for `residual_u`, `residual_v`, `residual_gust`
- [ ] Calibration step on validation set, saved per airport
- [ ] Full evaluation vs baselines

**Acceptance:** Beat raw HRRR by ≥15% RMSE on scalar wind speed at forecast hour 6 for KMAN. Train a second airport (KBOI) and confirm it also beats its raw HRRR baseline, confirming the pipeline generalizes.

### Phase 4: Operational outputs (target: 2 days)

- [ ] Runway crosswind/headwind decomposition utility (reads `runways` from airport config)
- [ ] `predict.py` produces: median wind, 10/90 band, P(crosswind > 10, 15, 20 kt) per runway, per forecast hour
- [ ] CLI: `wind-forecast predict --airport KMAN --cycle ... --lead 6`

### Phase 5 (stretch): Probabilistic upgrade

- [ ] Replace quantile ensemble with NGBoost or a small temporal transformer
- [ ] Ensemble 5 seeds, average predictions, use spread as additional uncertainty signal

### Phase 6 (deferred): Cross-airport generalization

Explicitly out of scope for v1. Revisit only after per-airport models are proven. Would require: elevation, terrain roughness, channeling indices, airport-ID embeddings, and careful leave-one-airport-out validation.

## Environment setup

```bash
uv init wind-forecast
cd wind-forecast
uv add lightgbm scikit-learn pandas polars xarray cfgrib herbie-data \
       properscoring pyyaml pydantic requests click joblib ngboost
uv add --dev pytest ruff mypy ipykernel matplotlib
```

## Conventions

- All times in UTC internally. Display in local time (using the airport’s configured timezone) only at the UI/CLI edge.
- Wind components stored as u (eastward) / v (northward). Convert from METAR’s direction-from convention at ingest.
- Airport ICAO code is the partition key for every dataset and artifact. No airport should ever be hardcoded in `src/`.
- No pandas `SettingWithCopyWarning` suppression. Fix the root cause.
- Every function that produces features must have a test that verifies no future leakage for a given `(cycle_time, lead_hour)` pair.
- `ruff` and `mypy --strict` must pass before commits.

## Open questions to resolve early

1. How many years of HRRR archive are realistically retrievable at reasonable speed? (HRRR only goes back to 2014; quality improved significantly after 2018.)
1. For each new airport, which neighbor stations best capture the upstream flow? This is a judgment call per site — document reasoning in the YAML as a comment.
1. For the gust model: how to handle METAR cycles where gust isn’t reported — treat as “no gust” or as missing?
1. Should `neighbor_stations` be auto-suggested (e.g., all ASOS stations within 50nm)? Probably not for v1 — human judgment is better for terrain-channeled airports.

## References

- Iowa Mesonet ASOS download: https://mesonet.agron.iastate.edu/request/download.phtml
- herbie docs: https://herbie.readthedocs.io
- HRRR variable reference: https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
- LightGBM quantile regression: https://lightgbm.readthedocs.io/en/latest/Parameters.html (objective=quantile)
- CRPS intuition: Gneiting & Raftery 2007, “Strictly Proper Scoring Rules”
- LAMP (the NWS system this mimics): https://vlab.noaa.gov/web/mdl/lamp
