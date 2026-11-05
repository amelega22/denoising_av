"""Stage 3: metrics + catalog from the stage-2 detections.

    python scripts/04_build_catalog.py --pipeline filter_then_denoise

Writes to outputs/<run_id>/<pipeline>/:
  catalog.csv                 one row per detection (filtered / denoised), as in the original notebook
  filtered_only_events.csv    filtered detections with no denoised counterpart + the denoised peak STA/LTA there
  catalog_manifest.json       counts, overlap summary, whole-period SNR, package versions
  figures/duration_hist.png   (--figures)
"""
import sys

import matplotlib.pyplot as plt

from _common import add_pipeline, get_cfg, pipelines_from_args, require
from shishaldin import catalog as cat
from shishaldin import plotting as P
from shishaldin import store
from shishaldin.metrics import whole_period_snr


def build_one(cfg, pipe, figures):
    p = cfg.data.primary_station
    print(f"\n{'=' * 70}\nCATALOG: {pipe}\n{'=' * 70}")

    filtered = store.load_traces(cfg, pipe, "filtered")[p]
    denoised = store.load_traces(cfg, pipe, "denoised")[p]
    det = store.load_detections(cfg, pipe)
    trig_f, trig_d = det["triggers_filtered"], det["triggers_denoised"]

    # refuse to build a catalog from detections made with different settings
    saved = store.load_json(cfg, pipe, "detections_summary.json")["trigger_params"]
    if saved != store.trigger_params(cfg):
        sys.exit("Trigger settings in the config differ from the ones used for the saved detections.\n"
                 f"  saved:   {saved}\n  current: {store.trigger_params(cfg)}\n"
                 "Rerun 03_detect_and_plot.py (or restore the old settings).")

    # ---- whole-period SNR
    snr = {}
    for name, tr, trig in (("filtered", filtered, trig_f), ("denoised", denoised, trig_d)):
        sig, noise, db = whole_period_snr(tr, trig)
        snr[name] = {"signal_rms": sig, "noise_rms": noise, "snr_db": db}
    print("WHOLE-PERIOD SNR")
    for name, s in snr.items():
        print(f"{name.capitalize()}:\n  Signal RMS: {s['signal_rms']:.6e}\n"
              f"  Noise RMS:  {s['noise_rms']:.6e}\n  SNR:        {s['snr_db']:.2f} dB\n")

    # ---- per-event metrics + catalog
    catalog, summary = cat.build_catalog(filtered, denoised, trig_f, trig_d,
                                         noise_seconds=cfg.trigger.lta_seconds)
    out = store.stage_dir(cfg, pipe)
    cat.save_catalog(catalog, out / "catalog.csv", compact=cfg.output.compact_csv_floats)

    print(f"filtered detections: {summary['filtered_detections']}")
    print(f"denoised detections: {summary['denoised_detections']}")
    print(f"matched (same event_number in both): {summary['matched']}")
    print(f"filtered-only events: {summary['filtered_only']}")
    print(f"denoised-only events: {summary['denoised_only']}")
    print(f"total rows saved: {summary['rows']}")

    # ---- overlap + filtered-only diagnostics
    overlap = cat.overlap_summary(filtered, denoised, trig_f, trig_d)
    print("\nEVENT OVERLAP SUMMARY (any overlap, not one-to-one)")
    for k, v in overlap.items():
        print(f"  {k}: {v}")

    fo = cat.filtered_only_report(filtered, denoised, trig_f, trig_d, det["cft_denoised"])
    fo.to_csv(out / "filtered_only_events.csv", index=False)
    print(f"\n{len(fo)} filtered-only events (peak denoised STA/LTA vs threshold {cfg.trigger.thr_on}) "
          f"-> filtered_only_events.csv")

    print("\nDURATIONS (s)")
    print(catalog["duration_seconds"].describe())

    store.save_json(cfg, pipe, "catalog_manifest.json", {
        "pipeline": pipe, "catalog_summary": summary, "overlap_summary": overlap,
        "whole_period_snr": snr, **store.provenance(),
    })

    if figures:
        fig = P.plot_duration_hist(catalog)
        fig.savefig(store.figure_dir(cfg, pipe) / "duration_hist.png", dpi=150)
        plt.close(fig)
    print(f"\nsaved catalog to {out / 'catalog.csv'}")


def main():
    def extra(ap):
        add_pipeline(ap)
        ap.add_argument("--figures", action="store_true", help="also save the duration histogram")

    cfg, args = get_cfg(__doc__, extra=extra)
    pipes = pipelines_from_args(args)
    require(cfg, pipes, store.has_detections, "detections",
            lambda p: f"python scripts/03_detect_and_plot.py --pipeline {p}")

    for pipe in pipes:
        build_one(cfg, pipe, args.figures)

    if len(pipes) == 2:
        print("\nNext: python scripts/05_snr_plots.py   (compare SNR between the two catalogs)")


if __name__ == "__main__":
    main()
