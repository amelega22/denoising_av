"""Shared argument parsing / setup for the stage scripts."""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # lets scripts run from a checkout without `pip install -e .`

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: figures are saved, not shown

from shishaldin.config import load_config
from shishaldin.store import PIPELINES

def get_cfg(description, extra=None):
    """Parse the common CLI flags and return (cfg, args). `extra(parser)` adds script-specific flags."""
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
