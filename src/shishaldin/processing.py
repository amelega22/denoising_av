"""Waveform processing: response removal, detrend, filter, DeepDenoiser."""
from __future__ import annotations

import logging
from typing import Tuple

from obspy import Inventory, Stream, Trace

from .config import Config

log = logging.getLogger(__name__)


def prepare_station(raw_stream: Stream, inv: Inventory, cfg: Config, station: str) -> Tuple[Trace, Trace]:
    """Remove instrument response (-> velocity) and pick the working component.

    Returns
    -------
    padded : the full padded trace (input to detrend / filter / denoise)
    raw    : the same data trimmed to the analysis window, response removed
             only (no filtering) -- used for the raw-vs-filtered-vs-denoised plot
    """
    st = raw_stream.select(channel=cfg.data.component).copy()
    if len(st) == 0:
        raise ValueError(f"{station}: no {cfg.data.component} data returned")

    st.remove_response(inventory=inv, output="VEL")

    # if the request came back in pieces (data gap), join them; gaps filled with 0
    if len(st) > 1:
        log.warning("%s has %d %s pieces (data gap) -- merging, gaps filled with 0",
                    station, len(st), cfg.data.component)
        st.merge(method=1, fill_value=0)

    padded = st[0]
    raw = padded.copy().trim(starttime=cfg.start, endtime=cfg.stop)
    return padded, raw


def detrend_trace(tr: Trace) -> Trace:
    tr = tr.copy()
    tr.detrend("demean")
    tr.detrend("linear")
    return tr


def filter_trace(tr: Trace, cfg: Config, kind: str, detrend: bool = True) -> Trace:
    """(Detrend,) filter, then trim the edge padding off.
    kind: "bandpass" (cfg.filter.bandpass) or "highpass" (cfg.filter.highpass)."""
    tr = detrend_trace(tr) if detrend else tr.copy()

    if kind == "bandpass":
        f = cfg.filter.bandpass
        tr.filter("bandpass", freqmin=f.freqmin, freqmax=f.freqmax,
                  corners=f.corners, zerophase=f.zerophase)
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
    """Run DeepDenoiser once over the full continuous trace.

    seisbench names the output channel "DeepDenoiser_<chan>", which is too long
    for miniSEED, so the output keeps the input's channel code.
    """
    out = model.annotate(Stream([trace]))[0].copy()
    out.stats.channel = trace.stats.channel
    return out
