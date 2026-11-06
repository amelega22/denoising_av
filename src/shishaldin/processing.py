"""prepare seismic waveforms and apply instrument response removal, detrending, filtering, and deepdenoiser processing"""from __future__ import annotations

import logging
from typing import Tuple
from obspy import Inventory, Stream, Trace
from .config import Config

log = logging.getLogger(__name__)

# remove instrument response and get raw data 
def prepare_station(raw_stream: Stream, inv: Inventory, cfg: Config, station: str) -> Tuple[Trace, Trace]:
    st = raw_stream.select(channel=cfg.data.component).copy()
    if len(st) == 0:
        raise ValueError(f"{station}: no {cfg.data.component} data returned")

    st.remove_response(inventory=inv, output="VEL")
    # gaps filled with 0
    if len(st) > 1:
        log.warning("%s has %d %s pieces (data gap) -- merging, gaps filled with 0", station, len(st), cfg.data.component)
        st.merge(method=1, fill_value=0)
    padded = st[0]
    raw = padded.copy().trim(starttime=cfg.start, endtime=cfg.stop)
    return padded, raw

def detrend_trace(tr: Trace) -> Trace:
    tr = tr.copy()
    tr.detrend("demean")
    tr.detrend("linear")
    return tr

# filter based on type of filter (bandpass or highpass)
def filter_trace(tr: Trace, cfg: Config, kind: str, detrend: bool = True) -> Trace:
    tr = detrend_trace(tr) if detrend else tr.copy()

    if kind == "bandpass":
        f = cfg.filter.bandpass
        tr.filter("bandpass", freqmin=f.freqmin, freqmax=f.freqmax, corners=f.corners, zerophase=f.zerophase)
    elif kind == "highpass":
        f = cfg.filter.highpass
        tr.filter("highpass", freq=f.freq, corners=f.corners, zerophase=f.zerophase)
    else:
        raise ValueError("kind must be 'bandpass' or 'highpass'")

    tr.trim(starttime=cfg.start, endtime=cfg.stop)
    return tr


def load_denoiser(cfg: Config):
    import seisbench.models as sbm  # heavy import, only needed here
    return sbm.DeepDenoiser.from_pretrained(cfg.denoise.pretrained)


def denoise_trace(model, trace: Trace) -> Trace:
    out = model.annotate(Stream([trace]))[0].copy()
    out.stats.channel = trace.stats.channel
    return out
