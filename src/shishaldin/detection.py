"""perform sta lta event detection, identify network coincidence events, and analyze individual station triggers"""from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.trigger import classic_sta_lta, coincidence_trigger, trigger_onset

from .config import Config, TriggerConfig

Interval = Tuple[UTCDateTime, UTCDateTime]


# helper functions (turn triggers to intervals, get interval overlaps)
def triggers_to_intervals(trace: Trace, triggers) -> List[Interval]:
    """(on, off) sample indices of `trace` -> (start, end) UTCDateTime pairs."""
    fs = trace.stats.sampling_rate
    t0 = trace.stats.starttime
    return [(t0 + on / fs, t0 + off / fs) for on, off in triggers]

def intervals_overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end

def overlap_flags(intervals: Sequence[Interval], others: Sequence[Interval]) -> List[bool]:
    """For each interval, is there ANY overlapping interval in `others`?"""
    return [any(intervals_overlap(s, e, os_, oe) for os_, oe in others) for s, e in intervals]


# network coincidence triggering
def run_sta_lta(trace: Trace, others: Sequence[Trace], tcfg: TriggerConfig, coinc_sum: int):
    fs = trace.stats.sampling_rate
    t0 = trace.stats.starttime
    nsta = int(tcfg.sta_seconds * fs)
    nlta = int(tcfg.lta_seconds * fs)

    # characteristic function of the primary trace, for plotting and diagnostics
    cft = classic_sta_lta(trace.data, nsta, nlta)

    traces = [trace, *others]
    coinc_sum = min(coinc_sum, len(traces))
    st_trig = Stream([tr.copy() for tr in traces])
    trig = coincidence_trigger(
        trigger_type="classicstalta", thr_on=tcfg.thr_on, thr_off=tcfg.thr_off,
        stream=st_trig, thr_coincidence_sum=coinc_sum,
        sta=tcfg.sta_seconds, lta=tcfg.lta_seconds, details=False,
    )

    triggers = []
    for t in trig:
        # safeguard: require `coinc_sum` DISTINCT stations in the event
        stations_hit = {tid.split(".")[1] for tid in t["trace_ids"]}
        if len(stations_hit) < coinc_sum:
            continue

        # no events during the startup period
        if (t["time"] - t0) < tcfg.startup_seconds:
            continue

        # UTC onset + duration -> sample indices of the primary trace event starts at the earliest onset of the
        # stations, which can be fractionally before the primary trace's own crossing, so clamp at 0
        on = max(int(round((t["time"] - t0) * fs)), 0)
        off = min(on + int(round(t["duration"] * fs)), len(trace.data) - 1)
        triggers.append((on, off))

    return cft, np.array(triggers, dtype=int).reshape(-1, 2)

def detect(traces_by_station: Dict[str, Trace], cfg: Config):
    # Run run_sta_lta with the configured primary station and the rest as partners
    primary = cfg.data.primary_station
    others = [traces_by_station[s] for s in cfg.data.stations if s != primary]
    return run_sta_lta(traces_by_station[primary], others, cfg.trigger, cfg.coincidence_sum)


# compare station by station and network
def analyze_station_triggers(traces_by_station: Dict[str, Trace], cfg: Config):
    """
    1. run sta/lta independently on every station
    2. run a coincidence trigger on the same traces
    3. classify each independent station trigger as coincident or single-station-only based on the network coincidence intervals

    return station_results, coincidence_intervals
    """
    t = cfg.trigger
    stations = cfg.data.stations
    station_results = {}
    coincidence_stream = Stream()

    for stn in stations:
        tr = traces_by_station[stn]
        fs = tr.stats.sampling_rate

        cft = classic_sta_lta(tr.data, int(t.sta_seconds * fs), int(t.lta_seconds * fs))
        # each station evaluated on its own, without requiring another to trigger
        triggers = trigger_onset(cft, t.thr_on, t.thr_off)
        # exclude startup triggers, as in the network detection
        triggers = [(int(on), int(off)) for on, off in triggers
                    if (on / fs) >= t.startup_seconds]
        station_results[stn] = {"cft": cft, "triggers": triggers, "intervals": triggers_to_intervals(tr, triggers),}
        coincidence_stream.append(tr.copy())

    coincidence_events = coincidence_trigger(
        trigger_type="classicstalta", thr_on=t.thr_on, thr_off=t.thr_off,
        stream=coincidence_stream, thr_coincidence_sum=cfg.coincidence_sum,
        sta=t.sta_seconds, lta=t.lta_seconds, details=True,
    )

    t0 = traces_by_station[cfg.data.primary_station].stats.starttime
    coincidence_intervals = []
    for event in coincidence_events:
        stations_hit = {tid.split(".")[1] for tid in event.get("trace_ids", [])}
        if len(stations_hit) < cfg.coincidence_sum:
            continue
        if (event["time"] - t0) < t.startup_seconds:
            continue
        start = event["time"]
        coincidence_intervals.append((start, start + event["duration"]))

    for stn in stations:
        station_results[stn]["coincident_flags"] = overlap_flags(station_results[stn]["intervals"], coincidence_intervals)

    return station_results, coincidence_intervals


# # explainer-plot helper
# def sta_lta_components(trace: Trace, tcfg: TriggerConfig):
#     """STA and LTA kept separate (trailing boxcar means of squared velocity) plus
#     their ratio, with the same warm-up / startup muting as run_sta_lta."""
#     fs = trace.stats.sampling_rate
#     nsta = int(tcfg.sta_seconds * fs)
#     nlta = int(tcfg.lta_seconds * fs)

#     sq = trace.data.astype(float) ** 2
#     sta = np.convolve(sq, np.ones(nsta) / nsta, mode="full")[:len(sq)]
#     lta = np.convolve(sq, np.ones(nlta) / nlta, mode="full")[:len(sq)]
#     ratio = np.where(lta > 0, sta / lta, 0.0)
#     ratio[:nlta] = 0                              # LTA window not yet full
#     ratio[:int(tcfg.startup_seconds * fs)] = 0    # same startup mute as run_sta_lta
#     return sta, lta, ratio
