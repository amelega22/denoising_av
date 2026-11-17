"""
Run the whole workflow with one command:

    python scripts/run_all.py

Steps run in order and the run stops at the first one that fails. Options:

    --refresh        passed to the Stage 1 script only (re-download data)
    --stitch-full    passed to Stage 1 (save duplicate full-period traces)
    --save-cft       passed to Stage 2 (save large STA/LTA arrays)
    --figures        passed to the catalog script only (save catalog figures)
    --start-at N     start from step N (1-4), e.g. to resume after a failure

Any other arguments (for example config options that every script accepts
through get_cfg) are passed unchanged to all four scripts.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Edit the file names to match your scripts. Each step is
# (description, file name, name of the optional flag that step accepts).
STEPS = [
    ("Stage 1: load, filter and denoise", "1_process_data.py", "refresh"),
    ("Stage 2: detect events", "2_sta_lta.py", None),
    ("Stage 3: build catalogs", "3_build_catalog.py", "figures"),
    ("Stage 4: plot comparisons", "4_plots.py", None),
]


def format_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--stitch-full", action="store_true")
    parser.add_argument("--save-cft", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument(
        "--start-at", type=int, default=1, choices=range(1, len(STEPS) + 1),
        metavar=f"1-{len(STEPS)}",
    )
    args, passthrough = parser.parse_known_args()

    script_dir = Path(__file__).resolve().parent
    missing = [name for _, name, _ in STEPS if not (script_dir / name).exists()]

    if missing:
        sys.exit(
            f"Script(s) not found in {script_dir}: {', '.join(missing)}\n"
            "Edit the STEPS list at the top of run_all.py."
        )

    flags = {
        "refresh": ["--refresh"] if args.refresh else [],
        "stitch_full": ["--stitch-full"] if args.stitch_full else [],
        "save_cft": ["--save-cft"] if args.save_cft else [],
        "figures": ["--figures"] if args.figures else [],
    }

    run_start = time.time()

    for number, (description, name, flag) in enumerate(STEPS, start=1):
        if number < args.start_at:
            print(f"Skipping step {number}: {description}")
            continue

        command = [
            sys.executable,
            str(script_dir / name),
            *passthrough,
            *(flags[flag] if flag else []),
            *(flags["stitch_full"] if number == 1 else []),
            *(flags["save_cft"] if number == 2 else []),
        ]

        print(f"\n{'=' * 70}")
        print(f"Step {number}/{len(STEPS)}: {description}")
        print(f"  {' '.join(command)}")
        print("=" * 70, flush=True)

        step_start = time.time()
        result = subprocess.run(command)

        if result.returncode != 0:
            print(
                f"\nStep {number} ({name}) failed with exit code "
                f"{result.returncode}. Stopping.\n"
                f"Fix the problem, then resume with:  "
                f"python {Path(__file__).name} --start-at {number}",
                file=sys.stderr,
            )
            sys.exit(result.returncode)

        print(f"\nStep {number} finished in "
              f"{format_duration(time.time() - step_start)}")

    print(f"\nAll steps finished in {format_duration(time.time() - run_start)}")


if __name__ == "__main__":
    main()
