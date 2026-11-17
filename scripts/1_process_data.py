"""
Load, correct instrument gain, bandpass filter raw data, and denoise the data both ways
in a single run:

    denoise_then_filter:  raw -> denoise -> bandpass
    filter_then_denoise:  raw -> bandpass -> denoise

Each pipeline is saved in its own output folder, named after the pipeline. The
raw and filtered (bandpass) traces are identical for both, so they are computed
once and saved in both folders, which keeps each folder complete on its own.

Processes the time period in weekly batches,  batch is downloaded with padding
on both sides, split at data gaps, and every continuous segment is processed
separately (response removal, taper, filter, denoise) so that gaps never pass
through a filter or the denoiser. The processed segments are rejoined (gaps
filled with zeros only at this last step), trimmed to the batch window, and saved
together with a record of where the gaps are. Daily traces are then stitched and
the final raw, filtered, and denoised traces are saved with waveform and
spectrogram figures.
"""

import json
import logging

import matplotlib.pyplot as plt
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn.header import FDSNNoDataException
import numpy as np
from _common import add_refresh, get_cfg
from shishaldin import store
from shishaldin.diagnostics import check_trace, save_denoising_diagnostics
from shishaldin.data import get_inventory, get_raw_stream
from shishaldin.plotting import plot_waveforms_spectrograms
from shishaldin.processing import (NoStationDataError, correct_instrument, denoise_trace, detrend_trace, filter_trace, load_denoiser)

PIPELINES = ("denoise_then_filter", "filter_then_denoise")
FILTERED_TYPE = "bandpass"
DENOISED_TYPE = "highpass"
SECONDS_PER_day = 7 * 24 * 60 * 60
log = logging.getLogger("stage1")

# padding downloaded and processed with the window and trimmed off afterward so filter and denoiser edge effects stay outside
PAD_SECONDS = 15 * 60
MIN_SEGMENT_SECONDS = 120
TAPER_SECONDS = 30

def iter_time_windows(start, stop):
    current = UTCDateTime(start)
    stop = UTCDateTime(stop)

    if stop <= current:
        raise ValueError("Stop time must be after start time.")

    while current < stop:
        next_time = min(current + SECONDS_PER_day, stop)
        yield current, next_time
        current = next_time

def trim_week(trace, start, stop, is_last_week):
    # pad=True keeps every daily trace the same length even when data is missing
    tr = trace.copy()
    end = stop if is_last_week else stop - tr.stats.delta
    tr.trim(starttime=start, endtime=end, pad=True, fill_value=0, nearest_sample=True)
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

def save_weekly_traces(cfg, pipeline, week_number, kind, traces):
    week_dir = store.weekly_dir(cfg, pipeline, week_number)
    traces_dir = week_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    for station, trace in traces.items():
        Stream([trace]).write(str(traces_dir / f"{kind}.{station}.mseed"),format="MSEED")


def save_weekly_gaps(cfg, pipeline, week_number, gaps_by_station):
    week_dir = store.weekly_dir(cfg, pipeline, week_number)
    week_dir.mkdir(parents=True, exist_ok=True)

    payload = {station: [{"start": str(start), "end": str(end)} for start, end in gaps] for station, gaps in gaps_by_station.items()}

    with open(week_dir / "gaps.json", "w") as f:
        json.dump(payload, f, indent=2)

def prepare_segments(stream, inv, station, instrument_correction="remove_sensitivity"):
    vertical = stream.select(component="Z").copy()

    if len(vertical) == 0:
        raise NoStationDataError(f"{station}: no vertical-component data")

    # Join overlapping or touching pieces, leave real gaps masked, then split
    # the masked trace into one ordinary trace per continuous segment.
    vertical.merge(method=1, fill_value=None)
    pieces = vertical.split()

    segments = []
    n_dropped = 0

    for piece in pieces:
        duration = piece.stats.endtime - piece.stats.starttime

        if duration < MIN_SEGMENT_SECONDS:
            n_dropped += 1
            continue

        correct_instrument(piece, inv, instrument_correction)
        segments.append(piece)

    return segments, n_dropped


def denoise_segment(model, cfg, pipeline, tapered, filtered, station, index, diagnostics_dir=None):
    if pipeline == "denoise_then_filter":
        detrended = detrend_trace(tapered.copy())
    else:
        detrended = detrend_trace(filtered.copy())
    check_trace(detrended)
    preprocessed = None
    if diagnostics_dir is not None:
        # Mirror SeisBench preprocessing on a copy; leave the actual pipeline intact.
        preprocessed = model.annotate_stream_pre(
            Stream([detrended.copy()]), dict(model.default_args)
        )[0]
        check_trace(preprocessed)
    denoised_raw = denoise_trace(model, detrended)
    check_trace(denoised_raw)

    log.debug(
        "%s %s segment %d: sampling rates input=%.2f Hz, model=%.2f Hz, "
        "output=%.2f Hz",
        station,
        pipeline,
        index,
        detrended.stats.sampling_rate,
        model.sampling_rate,
        denoised_raw.stats.sampling_rate,
    )

    for name, trace in (("input", detrended),("model output", denoised_raw)):
        data = np.asarray(trace.data, dtype=float)
        log.debug(
            "%s %s segment %d %s: npts=%d, min=%.4e, max=%.4e, RMS=%.4e",
            station, pipeline, index, name, len(data),
            np.min(data), np.max(data),
            np.sqrt(np.mean(data ** 2)),
        )

    if pipeline == "denoise_then_filter":
        result = filter_trace(denoised_raw, cfg, DENOISED_TYPE, detrend=False, trim=False)
    else:
        result = denoised_raw

    check_trace(result)
    if diagnostics_dir is not None:
        save_denoising_diagnostics(
            diagnostics_dir / pipeline / f"{station}.segment_{index:03d}",
            {"input": detrended, "model_preprocessed": preprocessed,
             "model_output": denoised_raw, "final_output": result},
            model, cfg,
        )
    return result


def process_segments(segments, model, cfg, station, diagnostics_dir=None):
    raw_pieces = []
    filtered_pieces = []
    denoised_pieces = {pipeline: [] for pipeline in PIPELINES}

    for index, segment in enumerate(segments):
        # Instrument-corrected, untapered copy for the raw output
        raw_pieces.append(segment.copy())

        # Taper the segment ends so gap edges do not ring through the filters
        tapered = segment.copy()
        tapered.taper(max_percentage=0.05, type="hann", max_length=TAPER_SECONDS)

        filtered = filter_trace(tapered.copy(),cfg,FILTERED_TYPE,trim=False)
        filtered_pieces.append(filtered)
        for pipeline in PIPELINES:
            denoised_pieces[pipeline].append(
                denoise_segment(model, cfg, pipeline, tapered, filtered, station, index, diagnostics_dir))

    return raw_pieces, filtered_pieces, denoised_pieces


def find_gaps(pieces, start, stop):
    """
    Return the (start, end) intervals inside [start, stop] that are not covered
    by any processed segment. These are the intervals that get zero-filled.
    """
    if not pieces:
        return [(start, stop)]

    tolerance = 1.5 * pieces[0].stats.delta
    intervals = sorted((piece.stats.starttime, piece.stats.endtime) for piece in pieces)
    gaps = []
    cursor = start

    for piece_start, piece_end in intervals:
        if piece_start >= stop:
            break

        if piece_start - cursor > tolerance:
            gaps.append((cursor, min(piece_start, stop)))

        cursor = max(cursor, piece_end)

    if stop - cursor > tolerance:
        gaps.append((cursor, stop))

    return gaps


def rebuild_trace(pieces, start, stop, is_last_week, station, kind):
    """
    Rejoin processed segments into one continuous trace, filling gaps with zeros
    only now, then trim to the batch window (removing the padding).
    """
    stream = Stream(traces=[piece.copy() for piece in pieces])
    stream.sort()
    stream.merge(method=1, fill_value=0)

    if len(stream) != 1:
        raise RuntimeError(
            f"{station}: could not rejoin {kind} segments into one trace "
            f"(got {len(stream)})."
        )

    return trim_week(stream[0], start, stop, is_last_week)


def add_options(parser):
    add_refresh(parser)
    parser.add_argument("--diagnostics", action="store_true",
                        help="save intermediate waveforms, spectral plots and checks per segment")


def main():
    cfg, args = get_cfg(__doc__, extra=add_options)
    inv = get_inventory(cfg, args.refresh)
    model = load_denoiser(cfg)

    # raw and filtered are shared by both pipelines, denoised is per pipeline
    weekly_raw = {}
    weekly_filtered = {}
    weekly_denoised = {pipeline: {} for pipeline in PIPELINES}

    windows = list(iter_time_windows(cfg.start, cfg.stop))

    log.info("pipelines: %s", ", ".join(PIPELINES))
    log.info("processing %d daily batch(es)", len(windows))

    for week_number, (week_start, week_stop) in enumerate(windows, start=1):
        is_last_week = week_number == len(windows)

        log.info(
            "week %02d/%02d: %s to %s",
            week_number,
            len(windows),
            week_start,
            week_stop,
        )

        week_raw = {}
        week_filtered = {}
        week_denoised = {pipeline: {} for pipeline in PIPELINES}
        week_gaps = {}

        for station in cfg.data.stations:
            log.info("week %02d: attempting station %s", week_number, station)

            try:
                # Download the batch with padding on both sides
                stream = get_raw_stream(cfg,station,args.refresh,start=week_start - PAD_SECONDS,stop=week_stop + PAD_SECONDS)

                if len(stream) == 0:
                    raise NoStationDataError(f"{station}: no waveform data returned")

                segments, n_dropped = prepare_segments(
                    stream, inv, station, cfg.data.instrument_correction)

                if not segments:
                    raise NoStationDataError(f"{station}: no continuous segment of at least ", f"{MIN_SEGMENT_SECONDS} s")

                # filter once, denoise once per pipeline, segment by segment
                diagnostics_dir = (cfg.run_dir / "diagnostics" / f"week_{week_number:02d}"
                                   if args.diagnostics else None)
                raw_pieces, filtered_pieces, denoised_pieces = process_segments(
                    segments, model, cfg, station, diagnostics_dir)

                gaps = find_gaps(filtered_pieces, week_start, week_stop)
                gap_seconds = sum(float(end - start) for start, end in gaps)

                log.info(
                    "week %02d: %s has %d segment(s), %d short piece(s) "
                    "dropped, %d gap(s) totalling %.0f s inside the window",
                    week_number,
                    station,
                    len(segments),
                    n_dropped,
                    len(gaps),
                    gap_seconds,
                )

                # rejoin and trim off the padding
                week_raw[station] = rebuild_trace(raw_pieces, week_start, week_stop,is_last_week, station, "raw")
                week_filtered[station] = rebuild_trace(filtered_pieces, week_start, week_stop,is_last_week, station, "filtered")

                for pipeline in PIPELINES:
                    week_denoised[pipeline][station] = rebuild_trace(denoised_pieces[pipeline], week_start, week_stop, is_last_week, station, f"denoised ({pipeline})")

                week_gaps[station] = gaps

                log.info(
                    "week %02d: successfully processed %s",
                    week_number,
                    station,
                )

            except (FDSNNoDataException, NoStationDataError) as exc:
                log.warning(
                    "week %02d: skipping station %s: %s",
                    week_number,
                    station,
                    exc,
                )

        # Require enough stations for this week's processing
        if len(week_filtered) < cfg.coincidence_sum:
            raise RuntimeError(
                f"Week {week_number} has only "
                f"{len(week_filtered)} usable station(s): "
                f"{list(week_filtered)}. "
                f"At least {cfg.coincidence_sum} are required."
            )

        # save weekly outputs into each pipeline's folder
        for pipeline in PIPELINES:
            for kind, traces in (("raw", week_raw), ("filtered", week_filtered), ("denoised", week_denoised[pipeline])):
                save_weekly_traces(cfg, pipeline, week_number, kind, traces)

            save_weekly_gaps(cfg, pipeline, week_number, week_gaps)

        for station, trace in week_raw.items():
            weekly_raw.setdefault(station, []).append(trace)

        for station, trace in week_filtered.items():
            weekly_filtered.setdefault(station, []).append(trace)

        for pipeline in PIPELINES:
            for station, trace in week_denoised[pipeline].items():
                weekly_denoised[pipeline].setdefault(station, []).append(trace)

        log.info("finished week %02d", week_number)

    # only include stations successfully processed in every week
    expected_weeks = len(windows)

    complete_stations = sorted(station for station, traces in weekly_filtered.items() if len(traces) == expected_weeks)

    if len(complete_stations) < cfg.coincidence_sum:
        raise RuntimeError(
            f"Only {len(complete_stations)} station(s) have data "
            f"for every week: {complete_stations}. "
            f"At least {cfg.coincidence_sum} complete stations "
            "are required."
        )

    incomplete_stations = sorted(set(weekly_filtered) - set(complete_stations))

    if incomplete_stations:
        log.warning(
            "excluding stations without data in every week: %s",
            incomplete_stations,
        )

    # Stitch each station's weekly traces into one full-period trace.
    # Raw and filtered are stitched once and reused for both pipelines.
    raw = {}
    filtered = {}
    denoised = {pipeline: {} for pipeline in PIPELINES}

    for station in complete_stations:
        raw[station] = stitch_weekly_traces(weekly_raw[station], station,"raw")
        filtered[station] = stitch_weekly_traces(weekly_filtered[station], station, "filtered")

        for pipeline in PIPELINES:
            denoised[pipeline][station] = stitch_weekly_traces(weekly_denoised[pipeline][station],station,f"denoised ({pipeline})")

    primary = (cfg.data.primary_station if cfg.data.primary_station in raw else complete_stations[0])

    for pipeline in PIPELINES:
        for kind, traces in (("raw", raw), ("filtered", filtered), ("denoised", denoised[pipeline])):
            store.save_traces(cfg, pipeline, kind, traces)

        cfg.save(store.stage_dir(cfg, pipeline) / "config_used.yaml")

        if pipeline == "denoise_then_filter":
            denoised_label = f"Denoised Raw + {DENOISED_TYPE}"
        else:
            denoised_label = f"{FILTERED_TYPE.capitalize()} + Denoised"

        fig = plot_waveforms_spectrograms(raw[primary], filtered[primary], denoised[pipeline][primary],cfg, labels=("Raw Data", f"Raw Bandpass Data ({FILTERED_TYPE})",denoised_label))

        output_figure = (store.figure_dir(cfg, pipeline) / "waveforms_spectrograms.png")

        fig.savefig(output_figure, dpi=150)
        plt.close(fig)

        print(f"[{pipeline}] saved traces to {store.stage_dir(cfg, pipeline)}")
        print(f"[{pipeline}] saved figure to {output_figure}")

    print(f"run folder: {cfg.run_dir}")


if __name__ == "__main__":
    main()
