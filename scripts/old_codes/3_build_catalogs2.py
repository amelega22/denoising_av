# %% Imports
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read
from scipy import signal as scipy_signal

from _common import get_cfg
from shishaldin import catalog as cat
from shishaldin import store
from shishaldin.detection import triggers_to_intervals, intervals_overlap
from shishaldin.metrics import window_metrics


# %% Settings
LOGGER = logging.getLogger(__name__)
PIPELINES = ("denoise_then_filter", "filter_then_denoise")
STATIONS = ["SSLS", "SSLN", "SSBA"]

COMPARISON_FREQMIN = 0.9
COMPARISON_FREQMAX = 8.0
COMPARISON_FILTER_CORNERS = 4
COMPARISON_MAX_LAG_SECONDS = 1.0
INTERPOLATION_METHOD = "lanczos"
INTERPOLATION_A = 20

# Used only if a day's detections_summary.json does not record the gap buffer
# that the detection script applied.
DEFAULT_GAP_BUFFER_SECONDS = 60


# %% Loading functions
def load_weekly_traces(week_dir, kind):
    traces = {}
    for station in STATIONS:
        path = week_dir / "traces" / f"{kind}.{station}.mseed"
        if path.exists():
            traces[station] = read(str(path))[0]
    return traces


def load_weekly_detections(week_dir):
    path = week_dir / "detections.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing detections file: {path}")

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def validate_trigger_params(week_dir, cfg):
    """Check the saved trigger parameters and return the detection summary."""
    path = week_dir / "detections_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing detection summary: {path}")

    with open(path, "r") as f:
        summary = json.load(f)

    saved = summary.get("trigger_params")
    current = store.trigger_params(cfg)

    if saved != current:
        raise ValueError(
            f"Trigger parameters do not match in {week_dir}. "
            f"Saved: {saved}; current: {current}"
        )

    return summary


def load_gaps(week_dir, buffer_seconds):
    """
    Load the gap intervals recorded by Stage 1, widened by buffer_seconds on
    each side. Returns {station: [(start, end), ...]}.
    """
    path = week_dir / "gaps.json"

    if not path.exists():
        LOGGER.warning(
            "%s has no gaps.json; metrics near data gaps will not be flagged.",
            week_dir.name
        )
        return {}

    with open(path, "r") as f:
        raw = json.load(f)

    return {
        station: [
            (
                UTCDateTime(gap["start"]) - buffer_seconds,
                UTCDateTime(gap["end"]) + buffer_seconds
            )
            for gap in gaps
        ]
        for station, gaps in raw.items()
    }


def overlaps_gap(interval, gaps):
    return any(
        intervals_overlap(interval[0], interval[1], gap_start, gap_end)
        for gap_start, gap_end in gaps
    )


# %% Waveform preparation
def make_denoised_comparison_traces(filtered_traces, denoised_traces):
    # Bandpass the denoised traces in memory (the saved denoised traces are not
    # modified), then interpolate onto the filtered reference's sample grid.
    comparison_traces = {}

    for station, denoised_trace in denoised_traces.items():
        filtered_trace = filtered_traces.get(station)

        if filtered_trace is None:
            comparison_traces[station] = None
            continue

        target_rate = float(filtered_trace.stats.sampling_rate)
        source_rate = float(denoised_trace.stats.sampling_rate)

        if target_rate / 2 <= COMPARISON_FREQMAX:
            LOGGER.warning(
                "%s reference sampling rate is too low for the comparison bandpass.",
                station
            )
            comparison_traces[station] = None
            continue

        comparison_trace = denoised_trace.copy()

        try:
            comparison_trace.filter(
                "bandpass",
                freqmin=COMPARISON_FREQMIN,
                freqmax=COMPARISON_FREQMAX,
                corners=COMPARISON_FILTER_CORNERS,
                zerophase=True
            )

            # Find the time interval covered by both waveforms.
            common_start = max(
                filtered_trace.stats.starttime,
                comparison_trace.stats.starttime
            )
            common_end = min(
                filtered_trace.stats.endtime,
                comparison_trace.stats.endtime
            )

            if common_end <= common_start:
                LOGGER.warning("%s has no overlapping waveform interval.", station)
                comparison_traces[station] = None
                continue

            # Snap the common interval to the filtered reference sample grid.
            reference_start = filtered_trace.stats.starttime
            reference_npts = int(filtered_trace.stats.npts)

            first_index = max(
                0,
                int(np.ceil(
                    float(common_start - reference_start) * target_rate - 1e-8
                ))
            )
            last_index = min(
                reference_npts - 1,
                int(np.floor(
                    float(common_end - reference_start) * target_rate + 1e-8
                ))
            )

            if last_index < first_index:
                comparison_traces[station] = None
                continue

            aligned_start = reference_start + first_index / target_rate
            aligned_npts = last_index - first_index + 1

            # Interpolate the denoised waveform onto the reference time grid.
            comparison_trace.interpolate(
                sampling_rate=target_rate,
                starttime=aligned_start,
                npts=aligned_npts,
                method=INTERPOLATION_METHOD,
                a=INTERPOLATION_A
            )

            comparison_traces[station] = comparison_trace

            LOGGER.info(
                "%s comparison waveform: %.2f Hz -> %.2f Hz; "
                "common interval %s to %s",
                station, source_rate, target_rate,
                comparison_trace.stats.starttime,
                comparison_trace.stats.endtime
            )

        except Exception:
            LOGGER.exception(
                "Could not prepare aligned denoised comparison trace for %s",
                station
            )
            comparison_traces[station] = None

    return comparison_traces


def waveform_covers(trace, start, end):
    if trace is None:
        return False
    return trace.stats.starttime <= start and trace.stats.endtime >= end


# %% SNR metrics
def unavailable_metrics():
    return {
        "waveform_available": False,
        "signal_rms": np.nan,
        "noise_rms": np.nan,
        "snr_db": np.nan,
        "max_amplitude": np.nan
    }


def unavailable_comparison():
    return {
        "sdr_db": np.nan,
        "max_correlation": np.nan,
        "cc_lag_seconds": np.nan
    }


def calculate_metrics(trace, start, end, noise_seconds):
    if not waveform_covers(trace, start, end):
        return unavailable_metrics()

    values = window_metrics(trace, start, end, noise_seconds)
    return {
        "waveform_available": True,
        "signal_rms": values[0],
        "noise_rms": values[1],
        "snr_db": values[2],
        "max_amplitude": values[3]
    }


# %% SDR and cross-correlation
def signal_distortion_ratio(reference, estimate):
    # SDR (dB) = 10*log10(||reference||^2 / ||reference-estimate||^2).
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    if reference.shape != estimate.shape:
        raise ValueError("Reference and estimate must have the same length.")

    distortion_power = np.sum((estimate - reference) ** 2)
    reference_power = np.sum(reference ** 2)

    if distortion_power == 0:
        return float("inf")
    if reference_power == 0:
        return float("-inf")

    return 10 * np.log10(reference_power / distortion_power)


def best_lag_and_correlation(
    reference, estimate, sampling_rate,
    max_lag_seconds=COMPARISON_MAX_LAG_SECONDS
):
    # Find the lag with the largest absolute normalized cross-correlation.
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    if len(reference) == 0 or len(estimate) == 0:
        return 0, np.nan

    correlation = scipy_signal.correlate(
        estimate, reference, mode="full"
    )
    lags = scipy_signal.correlation_lags(
        len(estimate), len(reference), mode="full"
    )

    max_lag_samples = int(round(max_lag_seconds * sampling_rate))
    window = np.abs(lags) <= max_lag_samples
    correlation = correlation[window]
    lags = lags[window]

    norm = np.sqrt(
        np.sum(reference ** 2) * np.sum(estimate ** 2)
    )

    normalized = correlation / norm if norm > 0 else np.zeros_like(correlation)

    best_idx = int(np.argmax(np.abs(normalized)))

    return int(lags[best_idx]), float(normalized[best_idx])


def align_and_score(
    reference, estimate, sampling_rate,
    max_lag_seconds=COMPARISON_MAX_LAG_SECONDS
):
    # Find the lag with the strongest correlation, regardless of polarity.
    best_lag_samples, signed_correlation = best_lag_and_correlation(
        reference, estimate, sampling_rate, max_lag_seconds
    )

    if best_lag_samples > 0:
        aligned_estimate = estimate[best_lag_samples:]
        aligned_reference = reference[:len(aligned_estimate)]
    elif best_lag_samples < 0:
        aligned_reference = reference[-best_lag_samples:]
        aligned_estimate = estimate[:len(aligned_reference)]
    else:
        aligned_reference = reference
        aligned_estimate = estimate

    n = min(len(aligned_reference), len(aligned_estimate))

    if n == 0:
        return {
            "cc_lag_seconds": best_lag_samples / sampling_rate,
            "max_correlation": np.nan,
            "sdr_db": np.nan
        }

    aligned_reference = np.asarray(aligned_reference[:n], dtype=float)
    aligned_estimate = np.asarray(aligned_estimate[:n], dtype=float)

    # Correct polarity for shape-only comparison.
    if np.isfinite(signed_correlation) and signed_correlation < 0:
        aligned_estimate = -aligned_estimate

    absolute_correlation = abs(signed_correlation)

    sdr_db = signal_distortion_ratio(
        aligned_reference, aligned_estimate
    )

    return {
        "cc_lag_seconds": best_lag_samples / sampling_rate,
        "max_correlation": absolute_correlation,
        "sdr_db": sdr_db
    }


def calculate_comparison_metrics(
    filtered_trace, denoised_bandpass_trace, start, end
):
    if not waveform_covers(filtered_trace, start, end):
        return unavailable_comparison()

    if not waveform_covers(denoised_bandpass_trace, start, end):
        return unavailable_comparison()

    reference_trace = filtered_trace.copy()
    estimate_trace = denoised_bandpass_trace.copy()

    reference_trace.trim(starttime=start, endtime=end)
    estimate_trace.trim(starttime=start, endtime=end)

    if len(reference_trace.data) == 0 or len(estimate_trace.data) == 0:
        return unavailable_comparison()

    reference_rate = float(reference_trace.stats.sampling_rate)
    estimate_rate = float(estimate_trace.stats.sampling_rate)

    if not np.isclose(reference_rate, estimate_rate):
        LOGGER.warning(
            "Sampling rates still differ for comparison window: %.3f vs %.3f Hz",
            reference_rate, estimate_rate
        )
        return unavailable_comparison()

    if not np.isclose(
        float(reference_trace.stats.delta),
        float(estimate_trace.stats.delta)
    ):
        return unavailable_comparison()

    # Both traces now use the filtered reference's sampling grid.
    # Trim to the same number of samples before lag estimation.
    n = min(len(reference_trace.data), len(estimate_trace.data))

    if n == 0:
        return unavailable_comparison()

    reference = np.asarray(reference_trace.data[:n], dtype=float)
    estimate = np.asarray(estimate_trace.data[:n], dtype=float)

    return align_and_score(
        reference,
        estimate,
        reference_rate,
        COMPARISON_MAX_LAG_SECONDS
    )


# %% Coincidence filtering and event matching
def coincidence_filtered_primary_intervals(
    detections, kind, primary_intervals, station_traces, primary_station
):
    """
    Keep primary-station intervals that overlap a coincidence-flagged
    trigger at at least one other station.

    The saved coincidence flags belong to each secondary station's trigger
    array. Convert those flagged trigger sample intervals to UTC intervals
    using that station's waveform, then match them to primary intervals.
    """
    flagged_secondary_intervals = []

    for station in STATIONS:
        if station == primary_station:
            continue

        trigger_key = f"triggers_{kind}_{station}"
        flags_key = f"coincident_flags_{kind}_{station}"

        if trigger_key not in detections or flags_key not in detections:
            LOGGER.warning(
                "Skipping coincidence information for %s: missing %s or %s.",
                station, trigger_key, flags_key
            )
            continue

        station_trace = station_traces.get(station)
        if station_trace is None:
            LOGGER.warning(
                "Skipping coincidence information for %s: waveform is unavailable.",
                station
            )
            continue

        station_triggers = np.asarray(detections[trigger_key])
        coincidence_flags = np.asarray(detections[flags_key], dtype=bool)

        if len(station_triggers) != len(coincidence_flags):
            raise ValueError(
                f"{flags_key} has {len(coincidence_flags)} flags but "
                f"{trigger_key} has {len(station_triggers)} triggers."
            )

        flagged_triggers = station_triggers[coincidence_flags]
        if len(flagged_triggers):
            flagged_secondary_intervals.extend(
                triggers_to_intervals(station_trace, flagged_triggers)
            )

    if not flagged_secondary_intervals:
        LOGGER.warning(
            "No coincidence-flagged secondary triggers available for %s data.",
            kind
        )
        return []

    coincident_primary = []
    for primary_interval in primary_intervals:
        is_coincident = any(
            intervals_overlap(
                primary_interval[0], primary_interval[1],
                secondary_interval[0], secondary_interval[1]
            )
            for secondary_interval in flagged_secondary_intervals
        )
        if is_coincident:
            coincident_primary.append(primary_interval)

    return coincident_primary


def match_intervals(filtered_intervals, denoised_intervals):
    candidates = []

    for fi, filtered in enumerate(filtered_intervals):
        for di, denoised in enumerate(denoised_intervals):
            if intervals_overlap(filtered[0], filtered[1], denoised[0], denoised[1]):
                overlap = min(filtered[1], denoised[1]) - max(filtered[0], denoised[0])
                onset_difference = abs(float(filtered[0] - denoised[0]))
                candidates.append((-float(overlap), onset_difference, fi, di))

    candidates.sort()
    matched_filtered = {}
    matched_denoised = set()

    for _, _, fi, di in candidates:
        if fi not in matched_filtered and di not in matched_denoised:
            matched_filtered[fi] = di
            matched_denoised.add(di)

    return matched_filtered, matched_denoised


def build_event_records(filtered_intervals, denoised_intervals):
    """
    Match detections one-to-one between the two datasets.

    Returns the matched events (these go in the catalog) and the unmatched
    filtered-only and denoised-only intervals (counted, not catalogued).
    """
    matches, matched_denoised = match_intervals(
        filtered_intervals, denoised_intervals
    )
    events = []

    for fi, di in matches.items():
        filtered = filtered_intervals[fi]
        denoised = denoised_intervals[di]
        start = min(filtered[0], denoised[0])
        end = max(filtered[1], denoised[1])

        events.append({
            "filtered_interval": filtered,
            "denoised_interval": denoised,
            "start": start,
            "end": end,
            "status": "both"
        })

    events.sort(key=lambda event: event["start"])

    filtered_only = [
        interval for index, interval in enumerate(filtered_intervals)
        if index not in matches
    ]
    denoised_only = [
        interval for index, interval in enumerate(denoised_intervals)
        if index not in matched_denoised
    ]

    return events, filtered_only, denoised_only


# %% Catalog construction
def make_catalog_rows(
    events,
    filtered_traces,
    denoised_traces,
    denoised_comparison_traces,
    gaps_by_station,
    week_number,
    first_event_number,
    noise_seconds
):
    rows = []

    for event_index, event in enumerate(events):
        event_number = first_event_number + event_index
        start, end = event["start"], event["end"]
        filtered_interval = event["filtered_interval"]
        denoised_interval = event["denoised_interval"]

        for station in STATIONS:
            filtered_trace = filtered_traces.get(station)
            denoised_trace = denoised_traces.get(station)
            denoised_bandpass_trace = denoised_comparison_traces.get(station)

            # If the noise window or the event overlaps a data gap at this
            # station, its waveform there is zero-filled, so every metric for
            # that station is set to NaN and the row is flagged.
            near_gap = overlaps_gap(
                (start - noise_seconds, end),
                gaps_by_station.get(station, [])
            )

            if near_gap:
                filtered_metrics = unavailable_metrics()
                denoised_metrics = unavailable_metrics()
                comparison = unavailable_comparison()
            else:
                filtered_metrics = calculate_metrics(
                    filtered_trace, start, end, noise_seconds
                )
                denoised_metrics = calculate_metrics(
                    denoised_trace, start, end, noise_seconds
                )
                comparison = calculate_comparison_metrics(
                    filtered_trace, denoised_bandpass_trace, start, end
                )

            rows.append({
                "event_number": event_number,
                "week": week_number,
                "station": station,
                "event_start_time": str(start),
                "event_end_time": str(end),
                "duration_seconds": float(end - start),
                "detection_status": event["status"],
                "filtered_detected": filtered_interval is not None,
                "denoised_detected": denoised_interval is not None,
                "filtered_detection_start_time": str(filtered_interval[0]) if filtered_interval else None,
                "filtered_detection_end_time": str(filtered_interval[1]) if filtered_interval else None,
                "denoised_detection_start_time": str(denoised_interval[0]) if denoised_interval else None,
                "denoised_detection_end_time": str(denoised_interval[1]) if denoised_interval else None,
                "near_gap": near_gap,
                "filtered_waveform_available": filtered_metrics["waveform_available"],
                "denoised_waveform_available": denoised_metrics["waveform_available"],
                "filtered_signal_rms": filtered_metrics["signal_rms"],
                "filtered_noise_rms": filtered_metrics["noise_rms"],
                "filtered_snr_db": filtered_metrics["snr_db"],
                "filtered_max_amplitude": filtered_metrics["max_amplitude"],
                "denoised_signal_rms": denoised_metrics["signal_rms"],
                "denoised_noise_rms": denoised_metrics["noise_rms"],
                "denoised_snr_db": denoised_metrics["snr_db"],
                "denoised_max_amplitude": denoised_metrics["max_amplitude"],
                "sdr_db": comparison["sdr_db"],
                "max_correlation": comparison["max_correlation"],
                "cc_lag_seconds": comparison["cc_lag_seconds"]
            })

    return rows


# %% Plotting
def plot_duration_histogram(catalog, output_dir):
    if catalog.empty:
        LOGGER.warning("No catalog events available for duration plot.")
        return

    durations = catalog.drop_duplicates("event_number")["duration_seconds"].dropna()
    if durations.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.hist(durations, bins=30, edgecolor="black")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of events")
    plt.title("Event Duration Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "duration_histogram.png", dpi=200)
    plt.close()


# %% Event count summary
def print_event_counts(pipeline, totals):
    print(f"\n[{pipeline}] primary-station event counts")
    print(
        f"  all detections:                   "
        f"filtered {totals['filtered_detections_total']:>6}   "
        f"denoised {totals['denoised_detections_total']:>6}"
    )
    print(
        f"  coincidence-confirmed:            "
        f"filtered {totals['filtered_coincidence_detections']:>6}   "
        f"denoised {totals['denoised_coincidence_detections']:>6}"
    )
    print(
        f"  matched in both (in catalog):     "
        f"{totals['matched_events_in_both']:>6}"
    )
    print(
        f"  filtered only (not in catalog):   "
        f"{totals['filtered_only_events']:>6}"
    )
    print(
        f"  denoised only (not in catalog):   "
        f"{totals['denoised_only_events']:>6}"
    )


# %% Main catalog workflow
def build_one(cfg, pipeline, make_figures=False):
    if pipeline not in PIPELINES:
        raise ValueError(
            f"Unsupported pipeline '{pipeline}'. "
            f"Expected one of: {', '.join(PIPELINES)}."
        )

    stage_dir = store.stage_dir(cfg, pipeline)
    weekly_dirs = sorted((stage_dir / "weekly").glob("week_*"))

    if not weekly_dirs:
        raise FileNotFoundError(f"No weekly directories found in {stage_dir}")

    all_rows = []
    weekly_summaries = []
    next_event_number = 1

    for week_dir in weekly_dirs:
        LOGGER.info("[%s] Building catalog for %s", pipeline, week_dir.name)
        detection_summary = validate_trigger_params(week_dir, cfg)

        gap_buffer = detection_summary.get(
            "gap_buffer_seconds", DEFAULT_GAP_BUFFER_SECONDS
        )
        gaps_by_station = load_gaps(week_dir, gap_buffer)

        detections = load_weekly_detections(week_dir)
        filtered_traces = load_weekly_traces(week_dir, "filtered")
        denoised_traces = load_weekly_traces(week_dir, "denoised")

        if (
            "triggers_filtered_primary" not in detections
            or "triggers_denoised_primary" not in detections
        ):
            raise KeyError(f"Missing primary trigger arrays in {week_dir}")

        if not filtered_traces or not denoised_traces:
            LOGGER.warning(
                "Missing waveform data in %s; available stations may have incomplete metrics.",
                week_dir
            )

        filtered_primary = filtered_traces.get(cfg.data.primary_station)
        denoised_primary = denoised_traces.get(cfg.data.primary_station)

        if filtered_primary is None or denoised_primary is None:
            raise FileNotFoundError(
                f"Primary station {cfg.data.primary_station} is missing in {week_dir}"
            )

        filtered_intervals_all = triggers_to_intervals(
            filtered_primary, detections["triggers_filtered_primary"]
        )
        denoised_intervals_all = triggers_to_intervals(
            denoised_primary, detections["triggers_denoised_primary"]
        )

        # First require a coincidence-flagged detection at another station
        # independently in both the filtered and denoised datasets.
        filtered_intervals = coincidence_filtered_primary_intervals(
            detections,
            "filtered",
            filtered_intervals_all,
            filtered_traces,
            cfg.data.primary_station
        )
        denoised_intervals = coincidence_filtered_primary_intervals(
            detections,
            "denoised",
            denoised_intervals_all,
            denoised_traces,
            cfg.data.primary_station
        )

        # Then match detections between the two datasets. Matched events go in
        # the catalog; unmatched ones are only counted.
        events, filtered_only, denoised_only = build_event_records(
            filtered_intervals, denoised_intervals
        )
        matched_count = len(events)

        # Prepare and time-align the denoised comparison traces once per station.
        denoised_comparison_traces = make_denoised_comparison_traces(
            filtered_traces, denoised_traces
        )

        rows = make_catalog_rows(
            events,
            filtered_traces,
            denoised_traces,
            denoised_comparison_traces,
            gaps_by_station,
            week_dir.name,
            next_event_number,
            cfg.trigger.lta_seconds
        )
        all_rows.extend(rows)

        weekly_summaries.append({
            "week": week_dir.name,
            "filtered_detections_total": len(filtered_intervals_all),
            "denoised_detections_total": len(denoised_intervals_all),
            "filtered_coincidence_detections": len(filtered_intervals),
            "denoised_coincidence_detections": len(denoised_intervals),
            "filtered_noncoincident_excluded": (
                len(filtered_intervals_all) - len(filtered_intervals)
            ),
            "denoised_noncoincident_excluded": (
                len(denoised_intervals_all) - len(denoised_intervals)
            ),
            "matched_events_in_both": matched_count,
            "filtered_only_events": len(filtered_only),
            "denoised_only_events": len(denoised_only),
            "unique_events": len(events)
        })

        next_event_number += len(events)

    catalog = pd.DataFrame(all_rows)
    output_dir = store.comparison_dir(cfg) / pipeline
    output_dir.mkdir(parents=True, exist_ok=True)

    cat.save_catalog(catalog, output_dir / "catalog.csv")

    totals = {
        key: sum(item[key] for item in weekly_summaries)
        for key in (
            "filtered_detections_total",
            "denoised_detections_total",
            "filtered_coincidence_detections",
            "denoised_coincidence_detections",
            "filtered_noncoincident_excluded",
            "denoised_noncoincident_excluded",
            "matched_events_in_both",
            "filtered_only_events",
            "denoised_only_events",
            "unique_events"
        )
    }

    manifest = {
        "pipeline": pipeline,
        "stations": STATIONS,
        "primary_station": cfg.data.primary_station,
        "noise_seconds": cfg.trigger.lta_seconds,
        "matching_method": "one-to-one greedy matching by largest overlap duration after coincidence filtering",
        "coincidence_method": "primary detection retained only when it overlaps a coincidence-flagged trigger at another station; applied independently to filtered and denoised data",
        "catalog_inclusion_rule": "event must be coincidence-triggered in both filtered and denoised data and matched by interval overlap; filtered-only and denoised-only events are counted but not catalogued",
        "metric_window": "union of the matched filtered and denoised detection intervals",
        "gap_handling": "metrics are set to NaN and near_gap is True for a station when its noise window plus event window overlaps a data gap (buffer taken from the detection summary)",
        "comparison_method": {
            "reference": "filtered waveform, 0.9-8 Hz",
            "denoised": "denoised waveform bandpass-filtered in memory, 0.9-8 Hz",
            "bandpass_corners": COMPARISON_FILTER_CORNERS,
            "zerophase": True,
            "denoised_interpolation": INTERPOLATION_METHOD,
            "interpolation_a": INTERPOLATION_A,
            "sampling_grid": "denoised interpolated onto filtered reference sample grid",
            "alignment_method": "maximum absolute normalized cross-correlation",
            "maximum_cc_lag_seconds": COMPARISON_MAX_LAG_SECONDS,
            "sdr_calculation": "after lag alignment and polarity correction: 10*log10(sum(reference**2)/sum((reference-estimate)**2))",
            "cc_calculation": "absolute value of the maximum normalized cross-correlation (polarity corrected)",
            "cc_lag_seconds": "positive lag means denoised estimate is shifted forward to align"
        },
        "totals": totals,
        "weekly_summaries": weekly_summaries,
        "catalog_rows": len(catalog)
    }

    with open(output_dir / "catalog_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if make_figures:
        plot_duration_histogram(catalog, output_dir / "figures")

    LOGGER.info("[%s] Catalog saved to %s", pipeline, output_dir)
    LOGGER.info("[%s] Detection totals: %s", pipeline, totals)

    print_event_counts(pipeline, totals)

    return catalog, manifest


# %% Command line
def main():
    def extra(ap):
        ap.add_argument("--figures", action="store_true", help="Save catalog figures.")

    config, args = get_cfg(
        "Build filtered-versus-denoised event catalogs for both pipelines",
        extra=extra
    )

    for pipeline in PIPELINES:
        build_one(config, pipeline, make_figures=args.figures)


if __name__ == "__main__":
    main()