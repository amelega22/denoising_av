"""retrieve seismic station metadata and raw waveform data with an on disk cache and padded time windows
--> on disk cache means that once the data are downloaded they are saved to the cache directory and can be used again 
rather than needing to be reloaded
"""
from __future__ import annotations

import logging
from pathlib import Path
from obspy import Stream, UTCDateTime, read, read_inventory
from .config import Config

log = logging.getLogger(__name__)

_STAMP = "%Y%m%dT%H%M%S%f"

def _client(cfg: Config):
    # imported lazily so fully-cached runs never touch the network stack
    from obspy.clients.fdsn import Client

    return Client(cfg.data.source)

def padded_window(cfg: Config):
    pad = cfg.data.edge_trim_seconds
    return cfg.start - pad, cfg.stop + pad

def _cache_dir(cfg: Config) -> Path:
    d = cfg.resolve(cfg.data.cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_inventory(cfg: Config, refresh: bool = False):
    d = cfg.data
    path = _cache_dir(cfg) / f"inventory_{d.network}_{'-'.join(d.stations)}.xml"
    if path.exists() and not refresh:
        log.info("inventory: cache hit (%s)", path.name)
        return read_inventory(str(path))

    log.info("inventory: downloading from %s", d.source)
    inv = _client(cfg).get_stations(network=d.network, station=",".join(d.stations), level="response")
    inv.write(str(path), format="STATIONXML")
    return inv

def get_raw_stream(cfg: Config, station: str, refresh: bool = False) -> Stream:
    d = cfg.data
    t0, t1 = padded_window(cfg)
    chan = d.channel.replace("*", "x")
    name = (f"{d.network}.{station}.{d.location or '--'}.{chan}."
            f"{t0.strftime(_STAMP)}_{t1.strftime(_STAMP)}.mseed")
    path = _cache_dir(cfg) / name
    if path.exists() and not refresh:
        log.info("%s: cache hit (%s)", station, path.name)
        return read(str(path))
    log.info("%s: downloading %s - %s", station, t0, t1)
    st = _client(cfg).get_waveforms(d.network, station, d.location, d.channel, t0, t1)
    if len(st) == 0:
        return st
st.write(str(path), format="MSEED")
return st
