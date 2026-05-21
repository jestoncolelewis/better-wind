"""Console + file logging for the CLI.

Console is intentionally quiet by default — the user sees a tqdm progress bar
and warnings/errors, nothing more. The log file (always on) captures the full
DEBUG stream for our own code (`wind_forecast.*`).
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_DIR = Path("logs")
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# These libraries spam DEBUG/INFO on every HTTP request or GRIB decode and
# drown out our own progress output. We pin their level to WARNING so the
# messages never propagate to ANY handler (console *or* file) regardless of
# what tqdm's `logging_redirect_tqdm` does to handler-level filters. Pass
# -vvv to let them through.
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


def setup_logging(*, verbose: int = 0, log_file: Path | None = None) -> Path:
    """Configure the root logger and return the resolved log file path.

    Console handler emits WARNING+ by default (so a tqdm bar can own the
    terminal). `-v` raises it to INFO, `-vv` to DEBUG for our own code while
    keeping noisy third-party libraries silent, and `-vvv` lets the libraries
    through too. The file handler is always DEBUG.
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

    lib_level = _noisy_library_level(verbose)
    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(lib_level)

    logging.getLogger("wind_forecast").debug("logging initialized -> %s", log_path)
    return log_path


@contextlib.contextmanager
def progress_logging() -> Iterator[None]:
    """Wrap `tqdm.contrib.logging.logging_redirect_tqdm` and keep our level.

    `logging_redirect_tqdm` swaps out the console handler for a tqdm-aware
    one but copies only the formatter and stream — it silently drops the
    handler's level and filters. Without this wrapper, every `logger.debug`
    in our code would land next to the progress bar even at default
    verbosity. We snapshot the original level + filters and re-apply them
    to the replacement handler.
    """
    from tqdm.contrib.logging import _TqdmLoggingHandler, logging_redirect_tqdm

    root = logging.getLogger()
    orig: tuple[int, list[logging.Filter]] | None = None
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            orig = (h.level, list(h.filters))
            break

    with logging_redirect_tqdm():
        if orig is not None:
            level, filters = orig
            for h in root.handlers:
                if isinstance(h, _TqdmLoggingHandler):
                    h.setLevel(level)
                    for f in filters:
                        h.addFilter(f)
                    break
        yield
