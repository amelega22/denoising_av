"""Prepare seismic waveforms, correct instrument gain, filter, and denoise."""
from __future__ import annotations

import logging
from typing import Tuple
from obspy import Inventory, Stream, Trace
from .config import Config

log = logging.getLogger(__name__)

class NoStationDataError(ValueError):
    """Raised when a station has no usable waveform data"""

def correct_instrument(tr: Trace, inv: Inventory, method: str = "remove_sensitivity") -> Trace:
    """Correct a trace in place using the configured instrument correction."""
    if method == "remove_sensitivity":
        sensitivity = inv.get_response(tr.id, tr.stats.starttime).instrument_sensitivity
        if sensitivity is None or (sensitivity.input_units or "").upper() != "M/S":
            raise ValueError(f"{tr.id}: sensitivity input units must be m/s for velocity plots")
        tr.remove_sensitivity(inventory=inv)
    elif method == "remove_response":
        tr.remove_response(inventory=inv, output="VEL")
    else:
        raise ValueError(f"Unknown instrument correction: {method}")
    return tr


# Correct instrument gain and get raw velocity data.
def prepare_station(raw_stream: Stream, inv: Inventory, cfg: Config, station: str, start=None, stop=None,) -> Tuple[Trace, Trace]:
    st = raw_stream.select(channel=cfg.data.component).copy()
    if len(st) == 0:
        raise NoStationDataError(f"{station}: no {cfg.data.component} data returned")

    for tr in st:
        correct_instrument(tr, inv, cfg.data.instrument_correction)

    if len(st) > 1:
        log.warning("%s has %d %s pieces (data gap) -- merging, gaps filled with 0", station,len(st), cfg.data.component)
        st.merge(method=1, fill_value=0)
    padded = st[0]

    week_start = cfg.start if start is None else start
    week_stop = cfg.stop if stop is None else stop

    raw = padded.copy().trim(starttime=week_start,endtime=week_stop )

    return padded, raw


def detrend_trace(tr: Trace) -> Trace:
    tr = tr.copy()
    tr.detrend("demean")
    tr.detrend("linear")
    return tr

# filter based on type of filter (bandpass or highpass)

def filter_trace(tr: Trace, cfg: Config, kind: str, detrend: bool = True, start=None, stop=None, trim: bool = True,
) -> Trace:
    tr = detrend_trace(tr) if detrend else tr.copy()

    if kind == "bandpass":
        f = cfg.filter.bandpass
        tr.filter(
            "bandpass",
            freqmin=f.freqmin,
            freqmax=f.freqmax,
            corners=f.corners,
            zerophase=f.zerophase,
        )

    elif kind == "highpass":
        f = cfg.filter.highpass
        tr.filter("highpass",freq=f.freq, corners=f.corners, zerophase=f.zerophase )

    else:
        raise ValueError("kind must be 'bandpass' or 'highpass'")

    if trim:
        trim_start = cfg.start if start is None else start
        trim_stop = cfg.stop if stop is None else stop
        tr.trim(starttime=trim_start, endtime=trim_stop)

    return tr



def load_denoiser(cfg: Config):
    import seisbench.models as sbm  # heavy import, only needed here
    return sbm.DeepDenoiser.from_pretrained(cfg.denoise.pretrained)


def denoise_trace(model, trace: Trace) -> Trace:
    out = model.annotate(Stream([trace]))[0].copy()
    out.stats.channel = trace.stats.channel
    return out
