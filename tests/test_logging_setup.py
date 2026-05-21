"""Logging setup tests."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pytest

from wind_forecast.logging_setup import default_log_path, setup_logging


def test_default_log_path_uses_logs_dir(tmp_path: Path) -> None:
    p = default_log_path(log_dir=tmp_path)
    assert p.parent == tmp_path
    assert p.name.startswith("wind-forecast-")
    assert p.suffix == ".log"


def test_setup_logging_creates_file(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    resolved = setup_logging(verbose=0, log_file=log_file)
    try:
        logging.getLogger("wind_forecast.test").info("hello")
        # Force handlers to flush
        for h in logging.getLogger().handlers:
            h.flush()
        assert resolved == log_file
        assert log_file.exists()
        assert "hello" in log_file.read_text()
    finally:
        for h in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(h)
            h.close()


def test_console_level_changes_with_verbose(tmp_path: Path) -> None:
    setup_logging(verbose=0, log_file=tmp_path / "a.log")
    console = next(
        h for h in logging.getLogger().handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )
    assert console.level == logging.WARNING

    setup_logging(verbose=1, log_file=tmp_path / "b.log")
    console = next(
        h for h in logging.getLogger().handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )
    assert console.level == logging.INFO

    setup_logging(verbose=2, log_file=tmp_path / "c.log")
    console = next(
        h for h in logging.getLogger().handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )
    assert console.level == logging.DEBUG

    # Tear down so other tests aren't affected
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
        h.close()


def test_noisy_libraries_pinned_to_warning_below_vvv(tmp_path: Path) -> None:
    """At -vv we still want DEBUG from our own code but NOT from urllib3 etc.

    Library levels are set on the logger itself (not as a handler filter)
    because `tqdm.contrib.logging.logging_redirect_tqdm` replaces our console
    handler and silently drops its filters and level. Setting the level on
    the library logger survives that swap.
    """
    setup_logging(verbose=2, log_file=tmp_path / "d.log")
    assert logging.getLogger("urllib3").level == logging.WARNING
    assert logging.getLogger("cfgrib").level == logging.WARNING
    assert logging.getLogger("herbie").level == logging.WARNING

    # Sub-loggers like urllib3.connectionpool inherit from urllib3.
    assert logging.getLogger("urllib3.connectionpool").isEnabledFor(logging.WARNING)
    assert not logging.getLogger("urllib3.connectionpool").isEnabledFor(logging.DEBUG)

    # Our own logger keeps full DEBUG.
    assert logging.getLogger("wind_forecast.ingest.hrrr").isEnabledFor(logging.DEBUG)

    # -vvv lets the libraries through
    setup_logging(verbose=3, log_file=tmp_path / "e.log")
    assert logging.getLogger("urllib3").level == logging.DEBUG

    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
        h.close()


def test_warnings_routed_to_file_not_console(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`warnings.warn(...)` should land in the log file but not on stderr.

    Library warnings (e.g. herbie's "Will not remove GRIB file...") use
    Python's `warnings` module, which bypasses logging by default. With
    `logging.captureWarnings(True)` they go through `py.warnings`; a console
    filter then keeps them off stderr while the file handler records them.
    """
    log_file = tmp_path / "w.log"
    setup_logging(verbose=0, log_file=log_file)
    try:
        # Make sure this specific warning isn't suppressed by Python's
        # once-per-location filter (some other test may have triggered it).
        warnings.simplefilter("always")
        warnings.warn("herbie-like grib warning xyz123", UserWarning, stacklevel=1)

        for h in logging.getLogger().handlers:
            h.flush()

        captured = capsys.readouterr()
        assert "herbie-like grib warning xyz123" not in captured.err
        assert "herbie-like grib warning xyz123" in log_file.read_text()
    finally:
        for h in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(h)
            h.close()
        warnings.resetwarnings()
