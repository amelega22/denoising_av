"""
Run independent STA/LTA network-coincidence detections on the filtered and denoised waveform branches for each weekly batch produced by 1_process_data.py

Runs for both Stage 1 pipelines, using the shared filtered traces and each
pipeline's own denoised traces:
    denoise_then_filter   raw -> denoise -> bandpass
    filter_then_denoise   raw -> bandpass -> denoise

2 methods:
1. Filtered: raw waveform bandpass filtered at 0.9-8 Hz (identical in both).
2. Denoised: the pipeline's denoised waveform.
"""

import json
import logging

import numpy as np
from obspy import UTCDateTime

from _common import get_cfg
from shishaldin import store
from shishaldin.detection import (analyze_station_triggers,detect,intervals_overlap,triggers_to_intervals)


logger = logging.getLogger("stage3")

PIPELINES = ("denoise_then_filter", "filter_then_denoise")
GAP_BUFFER_SECONDS = 60


def load_gaps(cfg, pipe, week_number, week_dir):
    path = store.weekly_gaps_path(cfg, pipe, week_number)
    if not path.exists():
        logger.warning(
            "%s has no gaps.json; triggers near data gaps will not be removed. "
            "Rerun Stage 1 to record gaps.",
            week_dir.name,
        )
        return {}

    with open(path, "r") as f:
        raw = json.load(f)

    return {station: [(UTCDateTime(gap["start"]) - GAP_BUFFER_SECONDS, UTCDateTime(gap["end"]) + GAP_BUFFER_SECONDS) for gap in gaps] for station, gaps in raw.items()}

def overlaps_gap(interval, gaps):
    return any(intervals_overlap(interval[0], interval[1], gap_start, gap_end) for gap_start, gap_end in gaps)

def mask_triggers(trace, triggers, gaps):
    triggers = np.asarray(triggers, dtype=int).reshape(-1, 2)

    if len(triggers) == 0 or not gaps:
        return triggers, np.ones(len(triggers), dtype=bool)

    intervals = triggers_to_intervals(trace, triggers)

    if len(intervals) != len(triggers):
        raise RuntimeError(f"Got {len(intervals)} intervals for {len(triggers)} triggers.")

    keep = np.array([not overlaps_gap(interval, gaps) for interval in intervals],dtype=bool)

    return triggers[keep], keep


def mask_station_results(results, traces, gaps_by_station):
    masked = {}
    removed = {}

    for station, result in results.items():
        original = np.asarray(result["triggers"], dtype=int).reshape(-1, 2)
        flags = np.asarray(result["coincident_flags"], dtype=bool)
        intervals = list(result["intervals"])

        if not (len(original) == len(flags) == len(intervals)):
            raise RuntimeError(
                f"{station}: triggers, coincidence flags and intervals have "
                f"different lengths ({len(original)}, {len(flags)}, "
                f"{len(intervals)})."
            )

        kept, keep = mask_triggers(traces[station], original, gaps_by_station.get(station, []))

        new_result = dict(result)
        new_result["triggers"] = kept
        new_result["coincident_flags"] = flags[keep]
        new_result["intervals"] = [interval for interval, k in zip(intervals, keep) if k]
        masked[station] = new_result
        removed[station] = int((~keep).sum())

    return masked, removed

def mask_network_coincidences(coincidences, primary_gaps):
    if not primary_gaps:
        return list(coincidences), 0

    kept = [interval for interval in coincidences if not overlaps_gap(interval, primary_gaps)]

    return kept, len(coincidences) - len(kept)

def interval_to_strings(intervals):
    return [{"start": str(start),"end": str(end)} for start, end in intervals]

def prepare_station_results(results):
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


def save_weekly_detections(week_dir, cft_f, cft_d, trig_f, trig_d, filtered_results, denoised_results, filtered_coincidences, denoised_coincidences, summary, save_cft=False):
    arrays = {"triggers_filtered_primary": np.asarray(trig_f, dtype=int).reshape(-1, 2), "triggers_denoised_primary": np.asarray(trig_d, dtype=int).reshape(-1, 2)}

    if save_cft:
        arrays["cft_filtered_primary"] = np.asarray(cft_f)
        arrays["cft_denoised_primary"] = np.asarray(cft_d)

    # Independent triggers are required downstream; full characteristic
    # functions are diagnostic-only and therefore opt-in.
    for station, result in filtered_results.items():
        arrays[f"triggers_filtered_{station}"] = np.asarray(result["triggers"],dtype=int).reshape(-1, 2)
        arrays[f"coincident_flags_filtered_{station}"] = np.asarray(result["coincident_flags"],dtype=bool)
        if save_cft:
            arrays[f"cft_filtered_{station}"] = np.asarray(result["cft"])

    for station, result in denoised_results.items():
        arrays[f"triggers_denoised_{station}"] = np.asarray(result["triggers"], dtype=int).reshape(-1, 2)
        arrays[f"coincident_flags_denoised_{station}"] = np.asarray(result["coincident_flags"],dtype=bool)
        if save_cft:
            arrays[f"cft_denoised_{station}"] = np.asarray(result["cft"])

    np.savez_compressed(week_dir / "detections.npz", **arrays)

    summary["filtered_station_results"] = prepare_station_results(filtered_results)
    summary["denoised_station_results"] = prepare_station_results(denoised_results)
    summary["filtered_network_coincidences"] = interval_to_strings(filtered_coincidences)
    summary["denoised_network_coincidences"] = interval_to_strings(denoised_coincidences)

    with open(week_dir / "detections_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

def detect_one_week(cfg, pipe, week_dir, week_number, save_cft=False):
    logger.info("Processing %s", week_dir.name)
    filtered = store.load_weekly_traces(
        cfg, pipe, week_number, "filtered"
    )
    denoised = store.load_weekly_traces(
        cfg, pipe, week_number, "denoised"
    )

    if not filtered:
        raise RuntimeError(f"No filtered traces found in {week_dir}")
    if not denoised:
        raise RuntimeError(f"No denoised traces found in {week_dir}")
    if set(filtered) != set(denoised):
        raise RuntimeError("Filtered and denoised traces contain different stations: " f"{sorted(filtered)} vs {sorted(denoised)}")

    stations = sorted(filtered)

    if len(stations) < cfg.coincidence_sum:
        raise RuntimeError(f"{week_dir.name}: need at least {cfg.coincidence_sum} " f"stations, but only have {stations}")

    primary = (cfg.data.primary_station if cfg.data.primary_station in filtered else stations[0])

    gaps_by_station = load_gaps(cfg, pipe, week_number, week_dir)
    primary_gaps = gaps_by_station.get(primary, [])

    # Run independent network-coincidence detections
    cft_f, trig_f = detect(filtered, cfg)
    cft_d, trig_d = detect(denoised, cfg)

    filtered_results, filtered_coincidences = (analyze_station_triggers(filtered, cfg))
    denoised_results, denoised_coincidences = (analyze_station_triggers(denoised, cfg))

    # Remove triggers that overlap a data gap + buffer
    n_network_f_before = len(trig_f)
    n_network_d_before = len(trig_d)

    trig_f, _ = mask_triggers(filtered[primary], trig_f, primary_gaps)
    trig_d, _ = mask_triggers(denoised[primary], trig_d, primary_gaps)

    filtered_results, filtered_removed = mask_station_results(filtered_results, filtered, gaps_by_station)
    denoised_results, denoised_removed = mask_station_results(denoised_results, denoised, gaps_by_station)
    filtered_coincidences, n_filtered_coincidences_removed = (mask_network_coincidences(filtered_coincidences, primary_gaps))
    denoised_coincidences, n_denoised_coincidences_removed = (mask_network_coincidences(denoised_coincidences, primary_gaps))

    print(f"\n[{pipe}] {week_dir.name}: stations = {stations}")

    print(
        f"Network coincidence requires {cfg.coincidence_sum} "
        f"of {len(stations)} available stations"
    )

    print(
        f"Triggers within {GAP_BUFFER_SECONDS} s of a data gap are removed "
        f"({'gaps.json found' if gaps_by_station else 'no gap information'})"
    )

    print(
        f"Filtered events: {len(trig_f)} "
        f"({n_network_f_before - len(trig_f)} removed near gaps; "
        f"any NaN in cft: {bool(np.isnan(cft_f).any())})"
    )

    print(
        f"Denoised events: {len(trig_d)} "
        f"({n_network_d_before - len(trig_d)} removed near gaps; "
        f"any NaN in cft: {bool(np.isnan(cft_d).any())})"
    )

    # Print station-level results.
    for name, results, coincidences, removed in (("Filtered",filtered_results,filtered_coincidences,filtered_removed),("Denoised",denoised_results,denoised_coincidences,denoised_removed)):
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
                f"{n - k} station-only; "
                f"{removed[station]} removed near gaps"
            )

    # weekly summary
    summary = {
        "pipeline": pipe,
        "week": week_number,
        "stations": stations,
        "primary_station": primary,
        "coincidence_sum": cfg.coincidence_sum,
        "n_events_filtered": int(len(trig_f)),
        "n_events_denoised": int(len(trig_d)),
        "gap_buffer_seconds": GAP_BUFFER_SECONDS,
        "characteristic_functions_saved": bool(save_cft),
        "removed_near_gaps": {
            "filtered_network_triggers": int(n_network_f_before - len(trig_f)),"denoised_network_triggers": int(n_network_d_before - len(trig_d)),"filtered_network_coincidences": int(n_filtered_coincidences_removed),
            "denoised_network_coincidences": int(n_denoised_coincidences_removed),
            "filtered_station_triggers": filtered_removed,
            "denoised_station_triggers": denoised_removed,
        },
        "trigger_params": store.trigger_params(cfg),
        **store.provenance(),
    }
    save_weekly_detections(week_dir,cft_f,cft_d,trig_f,trig_d,filtered_results,denoised_results,filtered_coincidences,denoised_coincidences,summary,save_cft=save_cft)
    print("Saved weekly detections")


def detect_one(cfg, pipe, save_cft=False):
    weekly_root = store.stage_dir(cfg, pipe) / "weekly"

    if not weekly_root.exists():
        raise FileNotFoundError(
            f"No weekly output directory found for {pipe}: {weekly_root}\n"
            "Run the Stage 1 script first."
        )

    week_dirs = sorted(path for path in weekly_root.glob("week_*") if path.is_dir())
    if not week_dirs:
        raise FileNotFoundError(f"No week folders found in {weekly_root}")

    logger.info(
        "Processing %d weekly batch(es) for %s",
        len(week_dirs),
        pipe,
    )

    for position, week_dir in enumerate(week_dirs, start=1):
        week_number = int(week_dir.name.removeprefix("week_"))
        logger.info("Week %02d/%02d: %s",position,len(week_dirs),week_dir.name,)
        detect_one_week(cfg,pipe,week_dir,week_number,save_cft=save_cft)


def main():
    def extra(parser):
        parser.add_argument(
            "--save-cft", action="store_true",
            help="save full STA/LTA characteristic arrays (large)",
        )

    cfg, args = get_cfg(__doc__, extra=extra)

    for pipe in PIPELINES:
        detect_one(cfg, pipe, save_cft=args.save_cft)


if __name__ == "__main__":
    main()
