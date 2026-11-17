"""
build weekly catalog from stage one detections

call with:
python scripts/04_build_catalog.py --pipeline filter_then_denoise
python scripts/04_build_catalog.py --pipeline denoise_then_filter --figures
"""

import sys
import json
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from obspy import read
from _common import add_pipeline, get_cfg, pipelines_from_args
from shishaldin import catalog as cat
from shishaldin import plotting as P
from shishaldin import store
from shishaldin.metrics import whole_period_snr

logger = logging.getLogger("stage4")

def load_weekly_traces(week_dir, kind):
    # load one week trace
    traces_dir = week_dir / "traces"
    traces = {}
    for path in sorted(traces_dir.glob(f"{kind}.*.mseed")):
        station = path.name[len(kind) + 1:-len(".mseed")]
        stream = read(str(path))
        if len(stream) != 1:
            raise RuntimeError(f"Expected one trace in {path}, found {len(stream)}")
        traces[station] = stream[0]
    return traces

def load_weekly_detections(week_dir):
    path = week_dir / "detections.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing detections file: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def build_one(cfg, pipe, figures):
    weekly_root = store.stage_dir(cfg, pipe) / "weekly"
    if not weekly_root.exists():
        raise FileNotFoundError(f"Missing weekly directory: {weekly_root}")

    week_dirs = sorted(path for path in weekly_root.glob("week_*") if path.is_dir())

    if not week_dirs:
        raise RuntimeError(f"No weekly batches found in {weekly_root}")

    all_catalogs = []
    all_filtered_only = []

    total_filtered = 0
    total_denoised = 0
    total_matched = 0
    total_filtered_only = 0
    total_denoised_only = 0

    filtered_signal_sq = 0.0
    filtered_noise_sq = 0.0
    denoised_signal_sq = 0.0
    denoised_noise_sq = 0.0
    filtered_metric_count = 0
    denoised_metric_count = 0

    weekly_summaries = []

    for week_number, week_dir in enumerate(week_dirs, start=1):
        print(f"\nProcessing {week_dir.name}")
        filtered_traces = load_weekly_traces(week_dir, "filtered")
        denoised_traces = load_weekly_traces(week_dir, "denoised")

        if not filtered_traces:
            raise RuntimeError(f"No filtered traces found in {week_dir}")

        if set(filtered_traces) != set(denoised_traces):
            raise RuntimeError(f"{week_dir.name}: filtered and denoised stations differ")

        stations = sorted(filtered_traces)

        if len(stations) < cfg.coincidence_sum:
            raise RuntimeError(f"{week_dir.name}: need at least {cfg.coincidence_sum} "f"stations, found {stations}")

        primary = (cfg.data.primary_station if cfg.data.primary_station in filtered_traces else stations[0])
        filtered = filtered_traces[primary]
        denoised = denoised_traces[primary]

        det = load_weekly_detections(week_dir)
        trig_f = det["triggers_filtered"]
        trig_d = det["triggers_denoised"]

        with open(week_dir / "detections_summary.json") as f:
            saved_summary = json.load(f)
        saved_params = saved_summary["trigger_params"]
        if saved_params != store.trigger_params(cfg):
            sys.exit(f"Trigger settings differ in {week_dir.name}.\n"
                f"Saved:   {saved_params}\n"
                f"Current: {store.trigger_params(cfg)}\n"
                "Rerun Stage 3 or restore the original settings."
            )

        # build the week's catalog
        weekly_catalog, summary = cat.build_catalog(filtered, denoised, trig_f, trig_d, noise_seconds=cfg.trigger.lta_seconds)
        weekly_catalog["week"] = week_number
        all_catalogs.append(weekly_catalog)

        # filtered only events
        weekly_fo = cat.filtered_only_report(filtered, denoised, trig_f, trig_d, det["cft_denoised"])
        weekly_fo["week"] = week_number
        all_filtered_only.append(weekly_fo)

        total_filtered += summary["filtered_detections"]
        total_denoised += summary["denoised_detections"]
        total_matched += summary["matched"]
        total_filtered_only += summary["filtered_only"]
        total_denoised_only += summary["denoised_only"]

        # Whole-week SNR components.
        for name, trace, triggers in (("filtered", filtered, trig_f), ("denoised", denoised, trig_d)):
            signal_rms, noise_rms, snr_db = whole_period_snr(trace, triggers)

            if name == "filtered":
                filtered_signal_sq += signal_rms ** 2
                filtered_noise_sq += noise_rms ** 2
                filtered_metric_count += 1
            else:
                denoised_signal_sq += signal_rms ** 2
                denoised_noise_sq += noise_rms ** 2
                denoised_metric_count += 1

            print(f"{name.capitalize()} weekly SNR: {snr_db:.2f} dB")

        weekly_summaries.append({"week": week_number,"week_folder": week_dir.name,"stations": stations,"catalog_summary": summary})

    # combine weekly catalogs
    catalog = pd.concat(all_catalogs, ignore_index=True)
    filtered_only = pd.concat(all_filtered_only, ignore_index=True)

    if not catalog.empty:
        catalog = catalog.sort_values("start_time").reset_index(drop=True)
        catalog["event_number"] = np.arange(1, len(catalog) + 1)
    snr = {}
    for name, signal_sq, noise_sq, count in (("filtered", filtered_signal_sq, filtered_noise_sq, filtered_metric_count),
        ("denoised",denoised_signal_sq, denoised_noise_sq, denoised_metric_count)):
        if count:
            signal_rms = np.sqrt(signal_sq / count)
            noise_rms = np.sqrt(noise_sq / count)
            snr_db = (20 * np.log10(signal_rms / noise_rms) if noise_rms > 0 and signal_rms > 0 else float("nan"))
        else:
            signal_rms = noise_rms = snr_db = float("nan")

        snr[name] = {"signal_rms": float(signal_rms), "noise_rms": float(noise_rms), "snr_db": float(snr_db)}

    out = store.stage_dir(cfg, pipe)
    out.mkdir(parents=True, exist_ok=True)
    cat.save_catalog(catalog, out / "catalog.csv", compact=cfg.output.compact_csv_floats)
    filtered_only.to_csv(out / "filtered_only_events.csv", index=False)

    print("\nRUN-WIDE CATALOG SUMMARY")
    print(f"Filtered detections: {total_filtered}")
    print(f"Denoised detections: {total_denoised}")
    print(f"Matched: {total_matched}")
    print(f"Filtered only: {total_filtered_only}")
    print(f"Denoised only: {total_denoised_only}")
    print(f"Total catalog rows: {len(catalog)}")

    print("\nRUN-WIDE SNR")
    for name, values in snr.items():
        print(
            f"{name.capitalize()}: "
            f"signal RMS={values['signal_rms']:.6e}, "
            f"noise RMS={values['noise_rms']:.6e}, "
            f"SNR={values['snr_db']:.2f} dB"
        )

    # Save manifest.
    manifest = {
        "pipeline": pipe,
        "number_of_weeks": len(week_dirs),
        "weekly_summaries": weekly_summaries,
        "catalog_summary": {
            "filtered_detections": total_filtered,
            "denoised_detections": total_denoised,
            "matched": total_matched,
            "filtered_only": total_filtered_only,
            "denoised_only": total_denoised_only,
            "rows": int(len(catalog)),
        },
        "whole_period_snr": snr,
        **store.provenance(),
    }

    store.save_json(
        cfg,
        pipe,
        "catalog_manifest.json",
        manifest,
    )

    print("\nDURATIONS (s)")
    if not catalog.empty:
        print(catalog["duration_seconds"].describe())

    if figures and not catalog.empty:
        fig = P.plot_duration_hist(catalog)
        fig.savefig(
            store.figure_dir(cfg, pipe) / "duration_hist.png",
            dpi=150,
        )
        plt.close(fig)

    print(f"\nSaved run-wide catalog to {out / 'catalog.csv'}")


def main():
    def extra(ap):
        add_pipeline(ap)
        ap.add_argument(
            "--figures",
            action="store_true",
            help="also save the duration histogram",
        )

    cfg, args = get_cfg(__doc__, extra=extra)
    pipes = pipelines_from_args(args)

    for pipe in pipes:
        build_one(cfg, pipe, args.figures)

    if len(pipes) == 2:
        print(
            "\nNext: python scripts/05_snr_plots.py "
            "(check whether it needs weekly input changes too)"
        )


if __name__ == "__main__":
    main()
