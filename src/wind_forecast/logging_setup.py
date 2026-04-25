"""Console + file logging for the CLI.

Console is intentionally quiet by default — the user sees a tqdm progress bar
and warnings/errors, nothing more. The log file (always on) captures the full
DEBUG stream for post-hoc debugging.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_DIR = Path("logs")
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def default_log_path(now: datetime | None = None, log_dir: Path = DEFAULT_LOG_DIR) -> Path:
    now = now or datetime.now(tz=timezone.utc)
    return log_dir / f"wind-forecast-{now:%Y%m%dT%H%M%SZ}.log"


def _console_level(verbose: int) -> int:
    if verbose >= 2:
        return logging.DEBUG
    if verbose >= 1:
        return logging.INFO
    return logging.WARNING


def setup_logging(*, verbose: int = 0, log_file: Path | None = None) -> Path:
    """Configure the root logger and return the resolved log file path.

    Console handler emits WARNING+ by default (so a tqdm bar can own the
    terminal). `-v` raises it to INFO, `-vv` to DEBUG. The file handler is
    always DEBUG so the log file is the source of truth.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(_console_level(verbose))
    console.setFormatter(formatter)
    root.addHandler(console)

    log_path = log_file or default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_h = logging.FileHandler(log_path, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(formatter)
    root.addHandler(file_h)

    logging.getLogger("wind_forecast").debug("logging initialized -> %s", log_path)
    return log_path
