"""Top-level evaluation: chronological split, run baselines, format a table.

`make eval-baselines` calls into `evaluate_airport(...)` once per configured
airport. The harness writes nothing but the printed/returned summary; nothing
here mutates `data/`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from wind_forecast.config import DEFAULT_DATA_ROOT, Airport
from wind_forecast.eval import baselines as bl
from wind_forecast.eval import io as eio
from wind_forecast.eval import metrics as em

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15

OVERALL_LEAD = "all"


def chronological_split(
    paired: pd.DataFrame,
    *,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by unique `cycle_utc` so all leads from a cycle stay together.

    The remaining fraction (`1 − train_frac − val_frac`) becomes the test set
    — most recent cycles, untouched until final scoring.
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(
            f"bad split fractions: train={train_frac}, val={val_frac}"
        )
    if paired.empty:
        empty = paired.iloc[0:0]
        return empty, empty, empty

    cycles = np.sort(paired["cycle_utc"].unique())
    n = len(cycles)
    n_train = max(1, int(n * train_frac))
    n_val = int(n * val_frac)
    n_train = min(n_train, n - 1)  # leave at least one cycle for test
    n_val = min(n_val, n - n_train - 1) if n - n_train > 1 else 0

    train_cut = cycles[n_train] if n_train < n else cycles[-1] + np.timedelta64(1, "ns")
    val_cut = (
        cycles[n_train + n_val]
        if n_train + n_val < n
        else cycles[-1] + np.timedelta64(1, "ns")
    )

    train = paired[paired["cycle_utc"] < train_cut]
    val = paired[(paired["cycle_utc"] >= train_cut) & (paired["cycle_utc"] < val_cut)]
    test = paired[paired["cycle_utc"] >= val_cut]
    logger.info(
        "split %d cycles: train=%d val=%d test=%d (rows: train=%d val=%d test=%d)",
        n, n_train, n_val, n - n_train - n_val, len(train), len(val), len(test),
    )
    return train.copy(), val.copy(), test.copy()


def _metrics_for(
    name: str,
    test: pd.DataFrame,
    pred: pd.DataFrame,
) -> list[em.MetricRow]:
    """One row per lead hour, plus an `all` row across every test sample."""
    rows: list[em.MetricRow] = []
    for lead, sub in test.groupby("lead_hour", sort=True):
        psub = pred.loc[sub.index]
        row = em.metric_row(
            pred_u=psub["pred_u"], pred_v=psub["pred_v"], pred_gust=psub["pred_gust"],
            obs_u=sub["obs_u"], obs_v=sub["obs_v"], obs_gust=sub["obs_gust"],
        )
        row["baseline"] = name
        row["lead_hour"] = int(lead)
        rows.append(row)

    overall = em.metric_row(
        pred_u=pred["pred_u"], pred_v=pred["pred_v"], pred_gust=pred["pred_gust"],
        obs_u=test["obs_u"], obs_v=test["obs_v"], obs_gust=test["obs_gust"],
    )
    overall["baseline"] = name
    overall["lead_hour"] = OVERALL_LEAD
    rows.append(overall)
    return rows


def evaluate_airport(
    airport: Airport,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    baselines: Iterable[str] = bl.ALL_BASELINES,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    paired: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the requested baselines for one airport, return a tidy metrics frame.

    `paired` is exposed for tests; production code passes `None` and we load
    from `data_root`.
    """
    if paired is None:
        paired = eio.load_and_pair(airport, data_root=data_root)
    if paired.empty:
        raise RuntimeError(f"no paired (HRRR, obs) rows for {airport.icao}")

    train, _val, test = chronological_split(paired, train_frac=train_frac, val_frac=val_frac)
    if test.empty:
        raise RuntimeError(f"empty test split for {airport.icao}")

    rows: list[em.MetricRow] = []
    for name in baselines:
        pred = bl.predict(name, train, test)
        rows.extend(_metrics_for(name, test, pred))

    df = pd.DataFrame(rows)
    df.insert(0, "airport", airport.icao)
    return df


METRIC_COLUMNS_ORDER: tuple[str, ...] = (
    "airport", "baseline", "lead_hour", "n",
    "rmse_u", "rmse_v", "rmse_speed",
    "mae_u", "mae_v", "mae_speed", "mae_dir_deg",
    "rmse_gust", "mae_gust", "crps_speed",
)


def format_table(df: pd.DataFrame, *, by_lead: bool = False) -> str:
    """Render a metrics frame as a fixed-width text table.

    `by_lead=False` (default) shows the `all`-leads summary only — the
    `make eval-baselines` acceptance test wants a small, readable table.
    `by_lead=True` includes every lead hour for deeper inspection.
    """
    if df.empty:
        return "(no metrics — empty test set)"
    work = df.copy()
    if not by_lead:
        work = work[work["lead_hour"] == OVERALL_LEAD]
    cols = [c for c in METRIC_COLUMNS_ORDER if c in work.columns]
    work = work[cols].copy()
    for c in work.columns:
        if c in {"airport", "baseline", "lead_hour"}:
            continue
        if c == "n":
            work[c] = work[c].astype("Int64")
            continue
        work[c] = work[c].map(lambda x: "nan" if pd.isna(x) else f"{x:.3f}")
    return str(work.to_string(index=False))
