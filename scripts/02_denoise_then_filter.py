"""
Load, remove response, DeepDenoiser, highpass filter

processes time period in weekly batches, saves weekly and intermediate outputs, stitches the weekly traces, and saves final
raw / filtered / denoised traces and waveform + spectrogram figure
"""

import logging

import matplotlib.pyplot as plt
from obspy import Stream, UTCDateTime

from _common import add_refresh, get_cfg
from shishaldin import store
from shishaldin.data import get_inventory, get_raw_stream
from shishaldin.plotting import plot_waveforms_spectrograms
from shishaldin.processing import (NoStationDataError, denoise_trace, detrend_trace, filter_trace, load_denoiser, prepare_station)
from obspy.clients.fdsn.header import FDSNNoDataException


PIPELINE = "denoise_then_filter"
FILTER_TYPE = "highpass"
SECONDS_PER_WEEK = 7 * 24 * 60 * 60
log = logging.getLogger("stage1b")


def iter_time_windows(start, stop):
    current = UTCDateTime(start)
    stop = UTCDateTime(stop)
    if stop <= current:
        raise ValueError("Stop time must be after start time.")

    while current < stop:
        next_time = min(current + SECONDS_PER_WEEK, stop)
        yield current, next_time
        current = next_time


def trim_week(trace, start, stop, is_last_week):
    tr = trace.copy()
    end = stop if is_last_week else stop - tr.stats.delta
    tr.trim(starttime=start, endtime=end, nearest_sample=True)
    return tr

def stitch_weekly_traces(weekly_traces, station, kind):
    if not weekly_traces:
        raise ValueError(f"No weekly {kind} traces available for {station}.")

    combined = Stream(traces=[tr.copy() for tr in weekly_traces])
    combined.sort()
    combined.merge(method=1, fill_value=0)

    if len(combined) != 1:
        raise RuntimeError(f"Could not stitch {kind} traces for {station}: "f"got {len(combined)} separate traces.")
    return combined[0]


def save_weekly_traces(cfg, week_number, kind, traces):
    week_dir = store.weekly_dir(cfg,PIPELINE,week_number)
    traces_dir = week_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    for station, trace in traces.items():
        Stream([trace]).write(str(traces_dir / f"{kind}.{station}.mseed"), format="MSEED")

def main():
    cfg, args = get_cfg(__doc__, extra=add_refresh)
    inv = get_inventory(cfg, args.refresh)
    model = load_denoiser(cfg)

    weekly_outputs = {"raw": {}, "filtered": {}, "denoised": {}}
    windows = list(iter_time_windows(cfg.start, cfg.stop))

    log.info("processing %d weekly batch(es)", len(windows))

    for week_number, (week_start, week_stop) in enumerate(windows,start=1):
        is_last_week = week_number == len(windows)
        log.info("week %02d/%02d: %s to %s", week_number, len(windows), week_start, week_stop)

        week_raw = {}
        week_filtered = {}
        week_denoised = {}

        for station in cfg.data.stations:
            log.info("week %02d: attempting station %s", week_number, station)
            try:
                stream = get_raw_stream(cfg,station,args.refresh,start=week_start,stop=week_stop)
                if len(stream) == 0:
                    raise NoStationDataError(f"{station}: no waveform data returned")

                padded, raw_trace = prepare_station(stream,inv,cfg,station,start=week_start,stop=week_stop)
                filtered_padded = filter_trace(padded, cfg, FILTER_TYPE, trim=False)
                detrended = detrend_trace(padded)
                denoised_before_filter = denoise_trace(model,detrended,)
                denoised_padded = filter_trace(denoised_before_filter, cfg, FILTER_TYPE, detrend=False, trim=False)

                week_raw[station] = trim_week(raw_trace, week_start, week_stop, is_last_week)

                week_filtered[station] = trim_week(filtered_padded, week_start, week_stop, is_last_week)

                week_denoised[station] = trim_week(denoised_padded, week_start, week_stop, is_last_week)

                log.info("week %02d: successfully processed %s", week_number, station)

            except (FDSNNoDataException, NoStationDataError) as exc:
                log.warning("week %02d: skipping station %s: %s",week_number,station,exc)

        # Require enough stations for this week's processing.
        if len(week_filtered) < cfg.coincidence_sum:
            raise RuntimeError(
                f"Week {week_number} has only "
                f"{len(week_filtered)} usable station(s): "
                f"{list(week_filtered)}. "
                f"At least {cfg.coincidence_sum} are required."
            )

        # Save weekly intermediate outputs and retain them for stitching.
        for kind, traces in (
            ("raw", week_raw),
            ("filtered", week_filtered),
            ("denoised", week_denoised),
        ):
            save_weekly_traces(cfg,week_number,kind,traces)

            for station, trace in traces.items():
                weekly_outputs[kind].setdefault(station,[]).append(trace)

        log.info("finished week %02d", week_number)

    # Only include stations that were successfully processed in every week.
    expected_weeks = len(windows)

    complete_stations = sorted(
        station
        for station, traces in weekly_outputs["filtered"].items()
        if len(traces) == expected_weeks
    )

    if len(complete_stations) < cfg.coincidence_sum:
        raise RuntimeError(
            f"Only {len(complete_stations)} station(s) have data "
            f"for every week: {complete_stations}. "
            f"At least {cfg.coincidence_sum} complete stations "
            "are required."
        )

    incomplete_stations = sorted(
        set(weekly_outputs["filtered"]) - set(complete_stations)
    )

    if incomplete_stations:
        log.warning(
            "excluding stations without data in every week: %s",
            incomplete_stations,
        )

    # Stitch each station's weekly traces into one full-period trace.
    raw = {}
    filtered = {}
    denoised = {}

    for station in complete_stations:
        raw[station] = stitch_weekly_traces(weekly_outputs["raw"][station],station,"raw",)
        filtered[station] = stitch_weekly_traces(weekly_outputs["filtered"][station],station,"filtered")
        denoised[station] = stitch_weekly_traces(weekly_outputs["denoised"][station],station,"denoised")

    for kind, traces in (("raw", raw),("filtered", filtered),("denoised", denoised)):
        store.save_traces(cfg,PIPELINE,kind,traces)

    cfg.save(store.stage_dir(cfg, PIPELINE) / "config_used.yaml")
    primary = (cfg.data.primary_station if cfg.data.primary_station in raw else complete_stations[0])

    fig = plot_waveforms_spectrograms(raw[primary], filtered[primary], denoised[primary], cfg, labels=("Raw Data",f"Filtered Data ({FILTER_TYPE})",f"Denoised, then {FILTER_TYPE}"))
    output_figure = (store.figure_dir(cfg, PIPELINE) / "waveforms_spectrograms.png")
    fig.savefig(output_figure, dpi=150)
    plt.close(fig)

    print(f"saved weekly and stitched traces to "f"{store.stage_dir(cfg, PIPELINE)}")
    print(f"saved figure to {output_figure}")
    print(f"run folder: {cfg.run_dir}")


if __name__ == "__main__":
    main()
