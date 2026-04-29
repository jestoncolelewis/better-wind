"""Deterministic baselines.

Each baseline takes a paired frame (`io.pair_obs_to_forecasts` output) and
returns a `pred_*` frame aligned to the same index. Splitting prediction
generation from metric computation keeps the harness simple.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRED_COLUMNS: tuple[str, ...] = ("pred_u", "pred_v", "pred_gust")

DEFAULT_CLIMATOLOGY_KEYS: tuple[str, ...] = ("lead_hour", "hour_of_day", "month")

ALL_BASELINES: tuple[str, ...] = ("persistence", "hrrr", "climatology")


def _empty_predictions(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {c: np.full(len(index), np.nan, dtype="float64") for c in PRED_COLUMNS},
        index=index,
    )


def raw_hrrr(test: pd.DataFrame) -> pd.DataFrame:
    """The unmodified HRRR 10 m wind + surface gust at the nearest grid point."""
    out = _empty_predictions(test.index)
    out["pred_u"] = test["u10"].to_numpy()
    out["pred_v"] = test["v10"].to_numpy()
    out["pred_gust"] = test["gust"].to_numpy()
    return out


def persistence(test: pd.DataFrame) -> pd.DataFrame:
    """Carry the obs at cycle init forward for every lead hour."""
    out = _empty_predictions(test.index)
    out["pred_u"] = test["init_u"].to_numpy()
    out["pred_v"] = test["init_v"].to_numpy()
    out["pred_gust"] = test["init_gust"].to_numpy()
    return out


def _add_temporal_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour-of-day and month columns derived from `valid_utc`."""
    valid = pd.to_datetime(df["valid_utc"], utc=True)
    return df.assign(
        hour_of_day=valid.dt.hour.astype("int64"),
        month=valid.dt.month.astype("int64"),
    )


def fit_climatology_bias(
    train: pd.DataFrame,
    *,
    keys: Iterable[str] = DEFAULT_CLIMATOLOGY_KEYS,
) -> pd.DataFrame:
    """Mean (HRRR − obs) bias per key bucket — fit on training data only.

    Returns a frame with one row per key combination and the columns
    `bias_u/bias_v/bias_gust`. Empty bins are filled by the all-train mean at
    apply time.
    """
    keys = list(keys)
    if "hour_of_day" in keys or "month" in keys:
        train = _add_temporal_keys(train)
    bias = pd.DataFrame(
        {
            "bias_u": train["u10"] - train["obs_u"],
            "bias_v": train["v10"] - train["obs_v"],
            "bias_gust": train["gust"] - train["obs_gust"],
        }
    )
    keyed = pd.concat([train[keys].reset_index(drop=True), bias.reset_index(drop=True)], axis=1)
    grouped = (
        keyed.groupby(keys, dropna=False)[["bias_u", "bias_v", "bias_gust"]]
        .mean()
        .reset_index()
    )
    return grouped


def climatology(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    keys: Iterable[str] = DEFAULT_CLIMATOLOGY_KEYS,
) -> pd.DataFrame:
    """HRRR forecast minus the per-bucket mean bias learned from `train`.

    Bias is computed by `(lead_hour, hour_of_day, month)` so the correction
    captures both diurnal valley-wind effects and seasonal terrain channeling
    while staying linear and trivially auditable.
    """
    keys = list(keys)
    if train.empty:
        logger.warning("climatology baseline: empty training frame, falling back to raw HRRR")
        return raw_hrrr(test)

    bias = fit_climatology_bias(train, keys=keys)
    test_keyed = _add_temporal_keys(test) if {"hour_of_day", "month"} & set(keys) else test.copy()
    test_keyed = test_keyed.assign(_row=np.arange(len(test_keyed)))

    merged = test_keyed.merge(bias, on=keys, how="left").sort_values("_row").reset_index(drop=True)

    def _safe_mean(s: pd.Series) -> float:
        # nanmean on an all-NaN slice warns; treat that as "no info, no shift".
        arr = s.to_numpy(dtype="float64")
        return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else 0.0

    fill_u = _safe_mean(bias["bias_u"]) if len(bias) else 0.0
    fill_v = _safe_mean(bias["bias_v"]) if len(bias) else 0.0
    fill_gust = _safe_mean(bias["bias_gust"]) if len(bias) else 0.0
    merged["bias_u"] = merged["bias_u"].fillna(fill_u)
    merged["bias_v"] = merged["bias_v"].fillna(fill_v)
    merged["bias_gust"] = merged["bias_gust"].fillna(fill_gust)

    out = _empty_predictions(test.index)
    out["pred_u"] = (test["u10"].to_numpy() - merged["bias_u"].to_numpy())
    out["pred_v"] = (test["v10"].to_numpy() - merged["bias_v"].to_numpy())
    out["pred_gust"] = (test["gust"].to_numpy() - merged["bias_gust"].to_numpy())
    return out


def predict(name: str, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Dispatch by baseline name. Train frame is unused for non-fit baselines."""
    if name == "hrrr":
        return raw_hrrr(test)
    if name == "persistence":
        return persistence(test)
    if name == "climatology":
        return climatology(train, test)
    raise ValueError(f"unknown baseline: {name!r} (expected one of {ALL_BASELINES})")
