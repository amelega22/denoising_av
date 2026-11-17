"""Parse and semantically validate a workflow YAML file without running a pipeline."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shishaldin.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=ROOT / "config.yaml", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Valid configuration: {args.config.resolve()}")
    print(f"Run: {cfg.run_id}")
    print(f"Stations: {', '.join(cfg.data.stations)} ({cfg.coincidence_sum} required)")
    print(f"Window: {cfg.start} to {cfg.stop}")


if __name__ == "__main__":
    main()
