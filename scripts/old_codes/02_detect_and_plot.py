
"""
Run independent STA/LTA network-coincidence detections on the filtered
and denoised waveform branches for each weekly batch produced by Stage 1.

The two detection methods are:
1. Filtered: raw waveform bandpass filtered at 0.9-8 Hz.
2. Denoised: denoised waveform highpass filtered at 0.1 Hz.

Run with:
python scripts/03_detect_and_plot.py --pipeline denoise_then_filter
"""

import json
import logging

import numpy as np
from obspy import read

from _common import STAGE1, add_pipeline, get_cfg, pipelines_from_args, require
from shishaldin import store
from shishaldin.detection import analyze_station_triggers, detect


logger = logging.getLogger("stage3")


def load_weekly_traces(week_dir, kind):
    traces_dir = week_dir / "traces"

    if not traces_dir.exists():
        raise FileNotFoundError(
            f"Missing weekly traces directory: {traces_dir}"
        )

    traces = {}

    for path in sorted(traces_dir.glob(f"{kind}.*.mseed")):
        station = path.name[len(kind) + 1:-len(".mseed")]
        stream = read(str(path))

        if len(stream) != 1:
            raise RuntimeError(
                f"Expected one trace in {path}, found {len(stream)}"
            )

        traces[station] = stream[0]

    return traces


def interval_to_strings(intervals):
    return [
        {
            "start": str(start),
            "end": str(end),
        }
        for start, end in intervals
    ]


def prepare_station_results(results):
    """
    Convert station-level detection results into JSON-compatible data.
    """

    summary = {}

    for station, result in results.items():
        summary[station] = {
            "n_independent_triggers": len(result["triggers"]),
            "n_coincidence_associated": int(
                sum(result["coincident_flags"])
            ),
            "n_station_only": int(
                len(result["coincident_flags"])
                - sum(result["coincident_flags"])
            ),
            "triggers": [
                {
                    "on_sample": int(on),
                    "off_sample": int(off),
                    "coincident": bool(flag),
                }
                for (on, off), flag in zip(
                    result["triggers"],
                    result["coincident_flags"],
                )
            ],
            "intervals": interval_to_strings(result["intervals"]),
        }

    return summary


def save_weekly_detections(
    week_dir,
    cft_f,
    cft_d,
    trig_f,
    trig_d,
    filtered_results,
    denoised_results,
    filtered_coincidences,
    denoised_coincidences,
    summary,
):
    """
    Save primary-station network detections and all station-level results.
    """

    arrays = {
        "cft_filtered_primary": cft_f,
        "cft_denoised_primary": cft_d,
        "triggers_filtered_primary": np.asarray(
            trig_f,
            dtype=int,
        ).reshape(-1, 2),
        "triggers_denoised_primary": np.asarray(
            trig_d,
            dtype=int,
        ).reshape(-1, 2),
    }

    # Save the characteristic functions and independent triggers for
    # every station and for both waveform branches.
    for station, result in filtered_results.items():
        arrays[f"cft_filtered_{station}"] = np.asarray(
            result["cft"]
        )
        arrays[f"triggers_filtered_{station}"] = np.asarray(
            result["triggers"],
            dtype=int,
        ).reshape(-1, 2)
        arrays[f"coincident_flags_filtered_{station}"] = np.asarray(
            result["coincident_flags"],
            dtype=bool,
        )

    for station, result in denoised_results.items():
        arrays[f"cft_denoised_{station}"] = np.asarray(
            result["cft"]
        )
        arrays[f"triggers_denoised_{station}"] = np.asarray(
            result["triggers"],
            dtype=int,
        ).reshape(-1, 2)
        arrays[f"coincident_flags_denoised_{station}"] = np.asarray(
            result["coincident_flags"],
            dtype=bool,
        )

    np.savez_compressed(
        week_dir / "detections.npz",
        **arrays,
    )

    # Save UTC intervals and readable station-level summaries.
    summary["filtered_station_results"] = prepare_station_results(
        filtered_results
    )
    summary["denoised_station_results"] = prepare_station_results(
        denoised_results
    )

    summary["filtered_network_coincidences"] = interval_to_strings(
        filtered_coincidences
    )
    summary["denoised_network_coincidences"] = interval_to_strings(
        denoised_coincidences
    )

    with open(week_dir / "detections_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def detect_one_week(cfg, pipe, week_dir, week_number):
    logger.info("Processing %s", week_dir.name)

    # Load the two Stage 1 waveform branches.
    filtered = load_weekly_traces(week_dir, "filtered")
    denoised = load_weekly_traces(week_dir, "denoised")

    if not filtered:
        raise RuntimeError(
            f"No filtered traces found in {week_dir}"
        )

    if not denoised:
        raise RuntimeError(
            f"No denoised traces found in {week_dir}"
        )

    if set(filtered) != set(denoised):
        raise RuntimeError(
            "Filtered and denoised traces contain different stations: "
            f"{sorted(filtered)} vs {sorted(denoised)}"
        )

    stations = sorted(filtered)

    if len(stations) < cfg.coincidence_sum:
        raise RuntimeError(
            f"{week_dir.name}: need at least {cfg.coincidence_sum} "
            f"stations, but only have {stations}"
        )

    primary = (
        cfg.data.primary_station
        if cfg.data.primary_station in filtered
        else stations[0]
    )

    # Run independent network-coincidence detections.
    # Each method uses only its own waveform branch.
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

    # Analyze independent station triggers for each branch.
    filtered_results, filtered_coincidences = (
        analyze_station_triggers(filtered, cfg)
    )

    denoised_results, denoised_coincidences = (
        analyze_station_triggers(denoised, cfg)
    )

    # Print station-level results.
    for name, results, coincidences in (
        (
            "Filtered",
            filtered_results,
            filtered_coincidences,
        ),
        (
            "Denoised",
            denoised_results,
            denoised_coincidences,
        ),
    ):
        print(
            f"\n{name}: "
            f"{len(coincidences)} network coincidence events"
        )

        for station, result in results.items():
            n = len(result["coincident_flags"])
            k = sum(result["coincident_flags"])

            print(
                f"  {station}: {n} independent triggers; "
                f"{k} coincidence-associated; "
                f"{n - k} station-only"
            )

    # Prepare the weekly summary.
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

    # Save network and station-level detection results.
    save_weekly_detections(
        week_dir,
        cft_f,
        cft_d,
        trig_f,
        trig_d,
        filtered_results,
        denoised_results,
        filtered_coincidences,
        denoised_coincidences,
        summary,
    )

    print("Saved weekly detections")


def detect_one(cfg, pipe):
    weekly_root = store.stage_dir(cfg, pipe) / "weekly"

    if not weekly_root.exists():
        raise FileNotFoundError(
            f"No weekly output directory found: {weekly_root}\n"
            "Run the corresponding Stage 1 script first."
        )

    week_dirs = sorted(
        path
        for path in weekly_root.glob("week_*")
        if path.is_dir()
    )

    if not week_dirs:
        raise FileNotFoundError(
            f"No week folders found in {weekly_root}"
        )

    logger.info(
        "Processing %d weekly batch(es) for %s",
        len(week_dirs),
        pipe,
    )

    for week_number, week_dir in enumerate(week_dirs, start=1):
        logger.info(
            "Week %02d/%02d: %s",
            week_number,
            len(week_dirs),
            week_dir.name,
        )

        detect_one_week(
            cfg,
            pipe,
            week_dir,
            week_number,
        )


def main():
    def extra(ap):
        add_pipeline(ap)

    cfg, args = get_cfg(__doc__, extra=extra)

    pipes = pipelines_from_args(args)

    require(
        cfg,
        pipes,
        store.has_traces,
        "stage-1 traces",
        lambda p: f"python scripts/{STAGE1[p]}",
    )

    for pipe in pipes:
        if pipe != "denoise_then_filter":
            raise ValueError(
                "Stage 3 now expects the combined Stage 1 output. "
                "Run with --pipeline denoise_then_filter."
            )

        detect_one(cfg, pipe)


if __name__ == "__main__":
    main()
