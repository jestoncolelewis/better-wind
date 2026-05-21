"""Console + file logging for the CLI.

Console is intentionally quiet by default — the user sees a ribbon progress
panel (see `wind_forecast.progress`) and warnings/errors, nothing more. The
log file (always on) captures the full DEBUG stream for our own code
(`wind_forecast.*`) plus any `warnings.warn(...)` that escapes from
third-party libraries (herbie, cfgrib, etc.) — those would otherwise bypass
logging and clog stderr.
"""

from __future__ import annotations

import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG_DIR = Path("logs")
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# These libraries spam DEBUG/INFO on every HTTP request or GRIB decode and
# drown out our own progress output. We pin their level to WARNING so the
# messages never propagate to ANY handler (console *or* file). Pass -vvv to
# let them through.
NOISY_LIBRARIES: tuple[str, ...] = (
    "urllib3",
    "requests",
    "botocore",
    "boto3",
    "s3transfer",
    "s3fs",
    "fsspec",
    "aiobotocore",
    "asyncio",
    "cfgrib",
    "herbie",
    "matplotlib",
    "findlibs",
    "gribapi",
    "eccodes",
)


def default_log_path(now: datetime | None = None, log_dir: Path = DEFAULT_LOG_DIR) -> Path:
    now = now or datetime.now(tz=timezone.utc)
    return log_dir / f"wind-forecast-{now:%Y%m%dT%H%M%SZ}.log"


def _console_level(verbose: int) -> int:
    if verbose >= 2:
        return logging.DEBUG
    if verbose >= 1:
        return logging.INFO
    return logging.WARNING


def _noisy_library_level(verbose: int) -> int:
    """Below -vvv, noisy libraries are pinned to WARNING."""
    return logging.DEBUG if verbose >= 3 else logging.WARNING


def _drop_console_warnings(record: logging.LogRecord) -> bool:
    """Console filter: keep `py.warnings` records out of stderr (file only)."""
    return not record.name.startswith("py.warnings")


def _showwarning_to_log(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: Any = None,
    line: str | None = None,
) -> None:
    """Route `warnings.warn(...)` through the `py.warnings` logger.

    Installed directly instead of via `logging.captureWarnings` because that
    function caches the prior `showwarning` in a module global and becomes
    a silent no-op on the second call (e.g. across pytest tests).
    """
    del file  # always send through logging, ignore user-requested stream
    formatted = warnings.formatwarning(message, category, filename, lineno, line)
    logging.getLogger("py.warnings").warning("%s", formatted.rstrip())


def setup_logging(*, verbose: int = 0, log_file: Path | None = None) -> Path:
    """Configure the root logger and return the resolved log file path.

    Console handler emits WARNING+ by default (so the ribbon panel can own
    the terminal). `-v` raises it to INFO, `-vv` to DEBUG for our own code while
    keeping noisy third-party libraries silent, and `-vvv` lets the libraries
    through too. The file handler is always DEBUG.

    `warnings.warn(...)` is also routed through the logging system so
    library warnings — e.g. herbie's "Will not remove GRIB file..." — land
    in the log file instead of bare stderr. A console-side filter keeps
    them off the terminal at every verbosity (tail the log to watch them).
    """
    warnings.showwarning = _showwarning_to_log

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(_console_level(verbose))
    console.setFormatter(formatter)
    console.addFilter(_drop_console_warnings)
    root.addHandler(console)

    log_path = log_file or default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_h = logging.FileHandler(log_path, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(formatter)
    root.addHandler(file_h)

    lib_level = _noisy_library_level(verbose)
    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(lib_level)

    # Records from captured warnings land on this logger; WARNING level lets
    # them through to both handlers (console filter then drops them).
    logging.getLogger("py.warnings").setLevel(logging.WARNING)

    logging.getLogger("wind_forecast").debug("logging initialized -> %s", log_path)
    return log_path
