"""load -> remove response -> BANDPASS filter -> DeepDenoiser.
plots are raw / filtered / denoised waveforms + spectrograms.

output is in outputs/<run_id>/filter_then_denoise/(traces/, figures/, config_used.yaml)
config file can be updated and whatever config file was used will be saved so that it is reproducable 
"""
import logging

import matplotlib.pyplot as plt

# added this to be able to refresh the data being used, when refresh is True it should start fresh
# get_cfg lets me load in the config file
from _common import add_refresh, get_cfg

# imports from files in src/shishaldin
from shishaldin import store
from shishaldin.data import get_inventory, get_raw_stream
from shishaldin.plotting import plot_waveforms_spectrograms
from shishaldin.processing import denoise_trace, filter_trace, load_denoiser, prepare_station

# established what steps are being taken, pipeline has two options in config (filter then denoise or denoise then filter) 
PIPELINE = "filter_then_denoise"

# this is because previously the highpass was just using the highpass value from the bandpass filter params, now they are two separate variables
FILTER_TYPE = "bandpass"          # uses filter.bandpass from config.yaml

log = logging.getLogger("stage1a")

def main():
    cfg, args = get_cfg(__doc__, extra=add_refresh)

    inv = get_inventory(cfg, args.refresh)
    raw, filtered, denoised = {}, {}, {}

    for stn in cfg.data.stations:
        padded, raw[stn] = prepare_station(get_raw_stream(cfg, stn, args.refresh), inv, cfg, stn)
        filtered[stn] = filter_trace(padded, cfg, FILTER_TYPE)
        log.info("%s", filtered[stn])

    model = load_denoiser(cfg)
    for stn in cfg.data.stations:
        # DeepDenoiser runs once on the full continuous filtered trace
        denoised[stn] = denoise_trace(model, filtered[stn])
        log.info("denoised %s (check sampling rate): %s", stn, denoised[stn])

    for kind, traces in (("raw", raw), ("filtered", filtered), ("denoised", denoised)):
        store.save_traces(cfg, PIPELINE, kind, traces)
    cfg.save(store.stage_dir(cfg, PIPELINE) / "config_used.yaml")

    p = cfg.data.primary_station
    fig = plot_waveforms_spectrograms(
        raw[p], filtered[p], denoised[p], cfg,
        labels=("Raw Data", f"Filtered Data ({FILTER_TYPE})", "Denoised Data"))
    out = store.figure_dir(cfg, PIPELINE) / "waveforms_spectrograms.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved traces + figure to {store.stage_dir(cfg, PIPELINE)}")


if __name__ == "__main__":
    main()
