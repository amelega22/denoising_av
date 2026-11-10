
"""
run STA/LTA network-coincidence detections separately for each weekly batch produced by Stage 1A or Stage 1B

Run with either:
python scripts/03_detect_and_plot.py --pipeline filter_then_denoise
python scripts/03_detect_and_plot.py --pipeline denoise_then_filter --explainer
"""

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from _common import STAGE1, add_pipeline, get_cfg, pipelines_from_args, require
from shishaldin import plotting as P
from shishaldin import store
from shishaldin.detection import analyze_station_triggers, detect

logger = logging.getLogger("stage3")

def load_weekly_traces(week_dir, kind):
    traces_dir = week_dir / "traces"

    if not traces_dir.exists():
        raise FileNotFoundError(f"Missing weekly traces directory: {traces_dir}")

    traces = {}

    for path in sorted(traces_dir.glob(f"{kind}.*.mseed")):
        station = path.name[len(kind) + 1:-len(".mseed")]
        stream = read(str(path))

        if len(stream) != 1:
            raise RuntimeError(f"Expected one trace in {path}, found {len(stream)}")

        traces[station] = stream[0]

    return traces


def save_weekly_detections(week_dir, cft_f, cft_d, trig_f, trig_d, summary):
    np.savez_compressed(week_dir / "detections.npz", cft_filtered=cft_f, cft_denoised=cft_d, triggers_filtered=np.asarray(trig_f, dtype=float),triggers_denoised=np.asarray(trig_d, dtype=float),)

    with open(week_dir / "detections_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def detect_one_week(cfg, pipe, week_dir, week_number):
    logger.info("Processing %s", week_dir.name)

    filtered = load_weekly_traces(week_dir, "filtered")
    denoised = load_weekly_traces(week_dir, "denoised")

    if not filtered:
        raise RuntimeError(f"No filtered traces found in {week_dir}")

    if set(filtered) != set(denoised):
        raise RuntimeError("Filtered and denoised traces contain different stations: "f"{sorted(filtered)} vs {sorted(denoised)}")

    stations = sorted(filtered)

    if len(stations) < cfg.coincidence_sum:
        raise RuntimeError(
            f"{week_dir.name}: need at least {cfg.coincidence_sum} stations, "f"but only have {stations}")

    primary = (cfg.data.primary_station if cfg.data.primary_station in filtered else stations[0])

    # Network-coincidence detections for this week only.
    cft_f, trig_f = detect(filtered, cfg)
    cft_d, trig_d = detect(denoised, cfg)

    print(f"\n{week_dir.name}: stations = {stations}")
    print(
        f"Network coincidence requires {cfg.coincidence_sum} "
        f"of {len(stations)} available stations"
    )
    print(
        f"Filtered events: {len(trig_f)} "
        f"(any NaN in cft: {bool(np.isnan(cft_f).any())})"
    )
    print(
        f"Denoised events: {len(trig_d)} "
        f"(any NaN in cft: {bool(np.isnan(cft_d).any())})"
    )

    # Independent station triggers, classified by network coincidence.
    f_res, f_coinc = analyze_station_triggers(filtered, cfg)
    d_res, d_coinc = analyze_station_triggers(denoised, cfg)

    for name, results, coincidences in (("Filtered", f_res, f_coinc),("Denoised", d_res, d_coinc)):
        print(f"\n{name}: {len(coincidences)} network coincidence events")

        for station, result in results.items():
            n = len(result["coincident_flags"])
            k = sum(result["coincident_flags"])
            print(f"  {station}: {n} independent triggers; "f"{k} coincidence-associated; {n - k} station-only")

    # Save this week's detection results.
    summary = {
        "pipeline": pipe,
        "week": week_number,
        "stations": stations,
        "primary_station": primary,
        "coincidence_sum": cfg.coincidence_sum,
        "n_events_filtered": int(len(trig_f)),
        "n_events_denoised": int(len(trig_d)),
        "trigger_params": store.trigger_params(cfg),
        **store.provenance(),
    }

    save_weekly_detections(week_dir, cft_f, cft_d, trig_f, trig_d, summary)

    # # Make this week's figures.
    # figs = {
    #     "triggers_thresholds": P.plot_triggers(
    #         filtered[primary], denoised[primary],
    #         cft_f, cft_d, trig_f, trig_d, cfg,
    #         mode="lines",
    #     ),
    #     "triggers_overlap": P.plot_triggers(
    #         filtered[primary], denoised[primary],
    #         cft_f, cft_d, trig_f, trig_d, cfg,
    #         mode="overlap",
    #     ),
    #     "station_triggers": P.plot_station_triggers(
    #         filtered, denoised, f_res, d_res, cfg
    #     ),
    # }
    #
    # figures_dir = week_dir / "figures"
    # figures_dir.mkdir(parents=True, exist_ok=True)
    #
    # for name, fig in figs.items():
    #     fig.savefig(figures_dir / f"{name}.png", dpi=150)
    #     plt.close(fig)

    print(f"Saved weekly detections")

def detect_one(cfg, pipe):
    weekly_root = store.stage_dir(cfg, pipe) / "weekly"

    if not weekly_root.exists():
        raise FileNotFoundError(f"No weekly output directory found: {weekly_root}\n" "Run the corresponding Stage 1 script first.")

    week_dirs = sorted(path for path in weekly_root.glob("week_*") if path.is_dir())

    if not week_dirs:
        raise FileNotFoundError(f"No week folders found in {weekly_root}")

    logger.info(
        "Processing %d weekly batch(es) for %s",
        len(week_dirs), pipe
    )

    for week_number, week_dir in enumerate(week_dirs, start=1):
        logger.info("Week %02d/%02d: %s",week_number, len(week_dirs), week_dir.name)
        detect_one_week(cfg, pipe, week_dir, week_number)


def main():
    def extra(ap):
        add_pipeline(ap)
    cfg, args = get_cfg(__doc__, extra=extra)
    pipes = pipelines_from_args(args)
    require(cfg, pipes, store.has_traces, "stage-1 traces",lambda p: f"python scripts/{STAGE1[p]}")

    for pipe in pipes:
        detect_one(cfg, pipe)


if __name__ == "__main__":
    main()
