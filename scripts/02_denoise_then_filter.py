"""load -> remove response -> DeepDenoiser -> highpass filter
plots are raw / filtered / denoised waveforms + spectrograms

filtered here is raw -> detrend -> highpass only (reference);
denoised is raw -> detrend -> DD -> highpass, 

run with: python scripts/02_denoise_then_filter.py

output: outputs/<run_id>/denoise_then_filter/(traces/, figures/, config_used.yaml)
"""
import logging

import matplotlib.pyplot as plt

from _common import add_refresh, get_cfg

from shishaldin import store
from shishaldin.data import get_inventory, get_raw_stream
from shishaldin.plotting import plot_waveforms_spectrograms
from shishaldin.processing import (NoStationDataError, denoise_trace, detrend_trace, filter_trace,load_denoiser, prepare_station)
from obspy.clients.fdsn.header import FDSNNoDataException

PIPELINE = "denoise_then_filter"

# uses filter.highpass from config file
FILTER_TYPE = "highpass"         

log = logging.getLogger("stage1b")

def main():
    cfg, args = get_cfg(__doc__, extra=add_refresh)

    inv = get_inventory(cfg, args.refresh)
    model = load_denoiser(cfg)
    raw, filtered, denoised = {}, {}, {}

    for stn in cfg.data.stations:
        try:
            stream = get_raw_stream(cfg, stn, args.refresh)

            padded, raw_trace = prepare_station(stream, inv, cfg, stn)

        except (FDSNNoDataException, NoStationDataError) as exc:
            log.warning("skipping station %s: %s", stn, exc)
            continue

        raw[stn] = raw_trace

        # reference: no denoiser
        filtered[stn] = filter_trace(padded, cfg, FILTER_TYPE)

        # detrend -> denoise -> highpass filter
        den = denoise_trace(model, detrend_trace(padded))
        denoised[stn] = filter_trace(den, cfg, FILTER_TYPE, detrend=False)

    if len(filtered) < cfg.coincidence_sum:
    raise RuntimeError(f"Need at least {cfg.coincidence_sum} stations with usable data, "f"but only found {list(filtered)}")

    for kind, traces in (("raw", raw), ("filtered", filtered), ("denoised", denoised)):
        store.save_traces(cfg, PIPELINE, kind, traces)
    cfg.save(store.stage_dir(cfg, PIPELINE) / "config_used.yaml")

    p = (cfg.data.primary_station if cfg.data.primary_station in raw else sorted(raw)[0])
    fig = plot_waveforms_spectrograms(
        raw[p], filtered[p], denoised[p], cfg,
        labels=("Raw Data", f"Filtered Data ({FILTER_TYPE})", f"Denoised, then {FILTER_TYPE}"))
    fig.savefig(store.figure_dir(cfg, PIPELINE) / "waveforms_spectrograms.png", dpi=150)
    plt.close(fig)
    print(f"saved traces + figure to {store.stage_dir(cfg, PIPELINE)}")


if __name__ == "__main__":
    main()
