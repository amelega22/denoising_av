"""Stage 1b: load -> remove response -> DeepDenoiser -> HIGHPASS filter.
Plots raw / filtered / denoised waveforms + spectrograms.

"filtered" here is raw -> detrend -> highpass only (the no-denoiser reference);
"denoised" is raw -> detrend -> DeepDenoiser -> highpass. Denoising and filtering
both run on the PADDED trace; the padding is trimmed off last.

    python scripts/02_denoise_then_filter.py

Writes outputs/<run_id>/denoise_then_filter/{traces/, figures/, config_used.yaml}
"""
import logging

import matplotlib.pyplot as plt

from _common import add_refresh, get_cfg

from shishaldin import store
from shishaldin.data import get_inventory, get_raw_stream
from shishaldin.plotting import plot_waveforms_spectrograms
from shishaldin.processing import (denoise_trace, detrend_trace, filter_trace,
                                   load_denoiser, prepare_station)

PIPELINE = "denoise_then_filter"
FILTER_TYPE = "highpass"          # uses filter.highpass from config.yaml

log = logging.getLogger("stage1b")


def main():
    cfg, args = get_cfg(__doc__, extra=add_refresh)

    inv = get_inventory(cfg, args.refresh)
    model = load_denoiser(cfg)
    raw, filtered, denoised = {}, {}, {}

    for stn in cfg.data.stations:
        padded, raw[stn] = prepare_station(get_raw_stream(cfg, stn, args.refresh), inv, cfg, stn)

        # reference: no denoiser
        filtered[stn] = filter_trace(padded, cfg, FILTER_TYPE)

        # detrend -> denoise (padded) -> highpass -> trim padding
        den = denoise_trace(model, detrend_trace(padded))
        log.info("denoised %s (check sampling rate): %s", stn, den)
        denoised[stn] = filter_trace(den, cfg, FILTER_TYPE, detrend=False)

    for kind, traces in (("raw", raw), ("filtered", filtered), ("denoised", denoised)):
        store.save_traces(cfg, PIPELINE, kind, traces)
    cfg.save(store.stage_dir(cfg, PIPELINE) / "config_used.yaml")

    p = cfg.data.primary_station
    fig = plot_waveforms_spectrograms(
        raw[p], filtered[p], denoised[p], cfg,
        labels=("Raw Data", f"Filtered Data ({FILTER_TYPE})", f"Denoised, then {FILTER_TYPE}"))
    fig.savefig(store.figure_dir(cfg, PIPELINE) / "waveforms_spectrograms.png", dpi=150)
    plt.close(fig)
    print(f"saved traces + figure to {store.stage_dir(cfg, PIPELINE)}")


if __name__ == "__main__":
    main()
