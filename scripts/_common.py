"""helper file for scripts 01-05, shared among scripts so that it does not need to be repeated multiple times throughout the workkflow"""
# allows arguments to be passed in terminal
import argparse
# allows messages with information to appear in terminal (and errors)
import logging
import sys
from pathlib import Path

# root directory is two paths up from common.py
ROOT = Path(__file__).resolve().parents[1]
# add src to root path so files in src/shishaldin are easy to find
sys.path.insert(0, str(ROOT / "src"))  # lets scripts run from a checkout without `pip install -e .`

# noqa: E402 allows for imports not at the beginning of the script and prevents warning  messages reltaed to that
import matplotlib  # noqa: E402

# create and save figures without them popping up in window
matplotlib.use("Agg")  

from shishaldin.config import load_config
from shishaldin.store import PIPELINES
from obspy import UTCDateTime

SECONDS_PER_WEEK = 7 * 24 * 60 * 60


def iter_time_windows(start, stop):
    """
    Yield consecutive time windows no longer than seven days.
    The final window may be shorter.
    """
    start = UTCDateTime(start)
    stop = UTCDateTime(stop)

    if stop <= start:
        raise ValueError("Stop time must be after start time.")

    current = start

    while current < stop:
        next_time = min(current + SECONDS_PER_WEEK, stop)
        yield current, next_time
        current = next_time

def get_cfg(description, extra=None):
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--start", help="override data.start_time")
    ap.add_argument("--stop", help="override data.stop_time")
    ap.add_argument("--out-dir", help="override output.dir")
    if extra:
        extra(ap)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, **{
        "data.start_time": args.start,
        "data.stop_time": args.stop,
        "output.dir": args.out_dir,
    })
    return cfg, args


def add_refresh(ap):
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-download")


def add_pipeline(ap):
    ap.add_argument("--pipeline", choices=[*PIPELINES, "both"], default="both",
                    help="which dataset(s) to process (default: %(default)s)")


def pipelines_from_args(args):
    return list(PIPELINES) if args.pipeline == "both" else [args.pipeline]


STAGE1 = {"filter_then_denoise": "01_filter_then_denoise.py",
          "denoise_then_filter": "02_denoise_then_filter.py"}


def require(cfg, pipes, check, what, fix):
    """Fail fast (before doing any work) if an earlier stage hasn't been run.
    check(cfg, pipe) -> bool; fix(pipe) -> the command to run."""
    missing = [p for p in pipes if not check(cfg, p)]
    if missing:
        lines = "\n".join(f"  {fix(p)}" for p in missing)
        sys.exit(f"Missing {what} for {', '.join(missing)} (window {cfg.run_id}).\n"
                 f"Run first (same --start/--stop):\n{lines}\n"
                 f"or process only one dataset with --pipeline.")
