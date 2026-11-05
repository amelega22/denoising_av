"""Stage 2: STA/LTA network-coincidence detections on the filtered and denoised
traces from a stage-1 run, plus the trigger plots.

    python scripts/03_detect_and_plot.py --pipeline filter_then_denoise
    python scripts/03_detect_and_plot.py --pipeline denoise_then_filter --explainer

Figures (outputs/<run_id>/<pipeline>/figures/):
  triggers_thresholds.png  filtered vs denoised waveform + STA/LTA with threshold lines, trigger on/off marked
  triggers_overlap.png     same, events shaded: green = in both, orange = filtered only, purple = denoised only
  station_triggers.png     each station's own triggers: green = part of a network coincidence, orange = one station only
  sta_lta_explainer.png    (--explainer) one event, step by step

Also writes detections.npz (cft + trigger indices) and detections_summary.json for stage 3.
"""
import matplotlib.pyplot as plt
import numpy as np

from _common import STAGE1, add_pipeline, get_cfg, pipelines_from_args, require

from shishaldin import plotting as P
from shishaldin import store
from shishaldin.detection import analyze_station_triggers, detect


def detect_one(cfg, pipe, explainer):
    p = cfg.data.primary_station
    print(f"\n{'=' * 70}\nDETECTION: {pipe}\n{'=' * 70}")

    filtered = store.load_traces(cfg, pipe, "filtered")
    denoised = store.load_traces(cfg, pipe, "denoised")

    # network coincidence trigger: only events seen at ALL required stations are kept
    cft_f, trig_f = detect(filtered, cfg)
    cft_d, trig_d = detect(denoised, cfg)

    print(f"stations: {cfg.data.stations} (primary: {p}), must trigger at "
          f"{cfg.coincidence_sum} of {len(cfg.data.stations)}")
    print(f"events in filtered data: {len(trig_f)}  (any NaN in cft: {bool(np.isnan(cft_f).any())})")
    print(f"events in denoised data: {len(trig_d)}  (any NaN in cft: {bool(np.isnan(cft_d).any())})")

    store.save_detections(cfg, pipe, cft_filtered=cft_f, cft_denoised=cft_d,
                          triggers_filtered=trig_f, triggers_denoised=trig_d)
    store.save_json(cfg, pipe, "detections_summary.json", {
        "pipeline": pipe,
        "n_events_filtered": int(len(trig_f)),
        "n_events_denoised": int(len(trig_d)),
        "trigger_params": store.trigger_params(cfg),
        **store.provenance(),
    })

    figs = {
        "triggers_thresholds": P.plot_triggers(filtered[p], denoised[p], cft_f, cft_d,
                                               trig_f, trig_d, cfg, mode="lines"),
        "triggers_overlap": P.plot_triggers(filtered[p], denoised[p], cft_f, cft_d,
                                            trig_f, trig_d, cfg, mode="overlap"),
    }

    # per-station independent triggers, classified by network coincidence
    f_res, f_coinc = analyze_station_triggers(filtered, cfg)
    d_res, d_coinc = analyze_station_triggers(denoised, cfg)
    figs["station_triggers"] = P.plot_station_triggers(filtered, denoised, f_res, d_res, cfg)

    for name, res, coinc in (("Filtered", f_res, f_coinc), ("Denoised", d_res, d_coinc)):
        print(f"\n{name}: {len(coinc)} network coincidence events")
        for stn, r in res.items():
            n, k = len(r["coincident_flags"]), sum(r["coincident_flags"])
            print(f"  {stn}: {n} independent triggers; {k} coincidence-associated; {n - k} station-only")

    if explainer:
        if len(trig_f):
            figs["sta_lta_explainer"] = P.plot_sta_lta_explainer(filtered[p], trig_f, cfg)
        else:
            print("no filtered events -> skipping explainer")

    out = store.figure_dir(cfg, pipe)
    for name, fig in figs.items():
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)
    print(f"\nsaved detections + {len(figs)} figures to {store.stage_dir(cfg, pipe)}")


def main():
    def extra(ap):
        add_pipeline(ap)
        ap.add_argument("--explainer", action="store_true",
                        help="also make the single-event STA/LTA explainer figure")

    cfg, args = get_cfg(__doc__, extra=extra)
    pipes = pipelines_from_args(args)
    require(cfg, pipes, store.has_traces, "stage-1 traces", lambda p: f"python scripts/{STAGE1[p]}")

    for pipe in pipes:
        detect_one(cfg, pipe, args.explainer)


if __name__ == "__main__":
    main()
