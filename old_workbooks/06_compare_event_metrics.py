
from fractions import Fraction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import correlate, correlation_lags, resample_poly
from obspy import read

from _common import get_cfg
from shishaldin import store
from shishaldin.detection import triggers_to_intervals
from shishaldin.metrics import window_metrics


MAX_LAG_SECONDS = 2.0

COLUMNS = [
    "week", "event_number", "station", "reference_type", "estimate_type",
    "start_time", "end_time", "duration_seconds",
    "reference_signal_rms", "reference_noise_rms", "reference_snr_db",
    "estimate_signal_rms", "estimate_noise_rms", "estimate_snr_db",
    "delta_snr_db", "sdr_db", "max_correlation", "best_lag_seconds",
    "reference_detection_start", "reference_detection_end",
    "estimate_detection_start", "estimate_detection_end"
]


def load_weekly_traces(week_dir, kind):
    traces = {}
    for path in sorted((week_dir / "traces").glob(f"{kind}.*.mseed")):
        station = path.stem.split(".", 1)[1]
        stream = read(str(path))
        if len(stream) != 1:
            raise RuntimeError(f"Expected one trace in {path}, found {len(stream)}.")
        traces[station] = stream[0]
    return traces


def load_weekly_detections(week_dir):
    path = week_dir / "detections.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing weekly detection file: {path}")
    with np.load(path) as data:
        return {
            "triggers_filtered": np.asarray(data["triggers_filtered"], dtype=int),
            "triggers_denoised": np.asarray(data["triggers_denoised"], dtype=int)
        }


def calculate_sdr(reference, estimate, sampling_rate, max_lag_seconds=MAX_LAG_SECONDS):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]

    if n == 0:
        raise ValueError("Cannot calculate SDR for an empty waveform.")

    ref_centered = reference - np.mean(reference)
    est_centered = estimate - np.mean(estimate)

    if np.std(ref_centered) == 0 or np.std(est_centered) == 0:
        raise ValueError("Cannot align a constant waveform.")

    corr = correlate(est_centered, ref_centered, mode="full", method="fft")
    lags = correlation_lags(n, n, mode="full")
    valid = np.abs(lags) <= int(round(max_lag_seconds * sampling_rate))
    corr, lags = corr[valid], lags[valid]

    normalized_corr = []
    for lag, value in zip(lags, corr):
        if lag > 0:
            ref_part, est_part = ref_centered[:-lag], est_centered[lag:]
        elif lag < 0:
            ref_part, est_part = ref_centered[-lag:], est_centered[:lag]
        else:
            ref_part, est_part = ref_centered, est_centered

        denominator = np.sqrt(np.sum(ref_part ** 2) * np.sum(est_part ** 2))
        normalized_corr.append(value / denominator if denominator else 0.0)

    normalized_corr = np.asarray(normalized_corr)
    best_index = int(np.argmax(np.abs(normalized_corr)))
    best_lag = int(lags[best_index])
    best_correlation = float(normalized_corr[best_index])

    if best_lag > 0:
        aligned_reference, aligned_estimate = reference[:-best_lag], estimate[best_lag:]
    elif best_lag < 0:
        aligned_reference, aligned_estimate = reference[-best_lag:], estimate[:best_lag]
    else:
        aligned_reference, aligned_estimate = reference, estimate

    distortion_power = np.sum((aligned_reference - aligned_estimate) ** 2)
    reference_power = np.sum(aligned_reference ** 2)

    if distortion_power == 0:
        sdr_db = float("inf")
    elif reference_power == 0:
        sdr_db = float("-inf")
    else:
        sdr_db = float(10 * np.log10(reference_power / distortion_power))

    return {
        "sdr_db": sdr_db,
        "max_correlation": best_correlation,
        "best_lag_seconds": best_lag / sampling_rate
    }


def calculate_event_sdr(reference_trace, estimate_trace, start, end):
    ref_rate = float(reference_trace.stats.sampling_rate)
    est_rate = float(estimate_trace.stats.sampling_rate)
    target_rate = min(ref_rate, est_rate)

    def resample_trace(trace):
        data = np.asarray(trace.data, dtype=float)
        original_rate = float(trace.stats.sampling_rate)

        if not np.isclose(original_rate, target_rate):
            ratio = Fraction(target_rate / original_rate).limit_denominator(10000)
            data = resample_poly(data, ratio.numerator, ratio.denominator)
            actual_rate = original_rate * ratio.numerator / ratio.denominator
        else:
            actual_rate = original_rate

        return data, actual_rate

    reference, ref_rate = resample_trace(reference_trace)
    estimate, est_rate = resample_trace(estimate_trace)

    ref_start = reference_trace.stats.starttime
    est_start = estimate_trace.stats.starttime
    ref_end = ref_start + (len(reference) - 1) / ref_rate
    est_end = est_start + (len(estimate) - 1) / est_rate

    common_start = max(start, ref_start, est_start)
    common_end = min(end, ref_end, est_end)

    if common_end <= common_start:
        raise ValueError("Event has no common waveform coverage.")

    n = int(np.floor(float(common_end - common_start) * target_rate)) + 1
    offsets = np.arange(n) / target_rate

    ref_indices = (float(common_start - ref_start) + offsets) * ref_rate
    est_indices = (float(common_start - est_start) + offsets) * est_rate

    reference = np.interp(ref_indices, np.arange(len(reference)), reference)
    estimate = np.interp(est_indices, np.arange(len(estimate)), estimate)

    if np.std(reference) == 0:
        raise ValueError("Reference waveform is constant.")
    if np.std(estimate) == 0:
        raise ValueError("Estimate waveform is constant.")

    return calculate_sdr(reference, estimate, target_rate)


def match_events(reference_intervals, estimate_intervals):
    reference_order = sorted(
        range(len(reference_intervals)),
        key=lambda i: reference_intervals[i][0]
    )
    estimate_order = sorted(
        range(len(estimate_intervals)),
        key=lambda i: estimate_intervals[i][0]
    )

    active, next_reference, matches = [], 0, []

    for estimate_index in estimate_order:
        e_start, e_end = estimate_intervals[estimate_index]

        while next_reference < len(reference_order):
            r_index = reference_order[next_reference]
            r_start, r_end = reference_intervals[r_index]
            if r_start >= e_end:
                break
            active.append((r_index, r_start, r_end))
            next_reference += 1

        active = [item for item in active if item[2] > e_start]
        candidates = [
            item for item in active
            if item[1] < e_end and item[2] > e_start
        ]

        if not candidates:
            continue

        r_index, r_start, r_end = min(candidates, key=lambda item: item[0])
        active = [item for item in active if item[0] != r_index]

        start, end = max(e_start, r_start), min(e_end, r_end)
        if end > start:
            matches.append((start, end, r_start, r_end, e_start, e_end))

    return matches


def calculate_event_snr(trace, start, end, noise_seconds):
    signal_rms, noise_rms, snr_db, max_amplitude = window_metrics(
        trace, start, end, noise_seconds
    )
    return {
        "signal_rms": float(signal_rms),
        "noise_rms": float(noise_rms),
        "snr_db": float(snr_db),
        "max_amplitude": float(max_amplitude)
    }


def build_comparison_catalog(
    cfg, reference_traces, estimate_traces, matches, week_name, first_event_number
):
    noise_seconds = float(cfg.trigger.lta_seconds)
    stations = sorted(set(reference_traces) & set(estimate_traces))

    if not stations:
        raise RuntimeError("No stations have both reference and estimate traces.")

    rows = []

    for match_index, match in enumerate(matches):
        event_number = first_event_number + match_index
        start, end, r_start, r_end, e_start, e_end = match

        for station in stations:
            reference_trace = reference_traces[station]
            estimate_trace = estimate_traces[station]

            reference_metrics = calculate_event_snr(
                reference_trace, start, end, noise_seconds
            )
            estimate_metrics = calculate_event_snr(
                estimate_trace, start, end, noise_seconds
            )

            try:
                waveform_metrics = calculate_event_sdr(
                    reference_trace, estimate_trace, start, end
                )
            except ValueError as exc:
                print(f"Event {event_number}, {station}: waveform comparison skipped: {exc}")
                waveform_metrics = {
                    "sdr_db": np.nan,
                    "max_correlation": np.nan,
                    "best_lag_seconds": np.nan
                }

            reference_snr = reference_metrics["snr_db"]
            estimate_snr = estimate_metrics["snr_db"]
            delta_snr = (
                estimate_snr - reference_snr
                if np.isfinite(estimate_snr) and np.isfinite(reference_snr)
                else np.nan
            )

            rows.append({
                "week": week_name,
                "event_number": event_number,
                "station": station,
                "reference_type": "bandpass_filtered",
                "estimate_type": "denoised_highpass_filtered",
                "start_time": start.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "end_time": end.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "duration_seconds": float(end - start),
                "reference_signal_rms": reference_metrics["signal_rms"],
                "reference_noise_rms": reference_metrics["noise_rms"],
                "reference_snr_db": reference_snr,
                "estimate_signal_rms": estimate_metrics["signal_rms"],
                "estimate_noise_rms": estimate_metrics["noise_rms"],
                "estimate_snr_db": estimate_snr,
                "delta_snr_db": delta_snr,
                "sdr_db": waveform_metrics["sdr_db"],
                "max_correlation": waveform_metrics["max_correlation"],
                "best_lag_seconds": waveform_metrics["best_lag_seconds"],
                "reference_detection_start": r_start.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "reference_detection_end": r_end.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "estimate_detection_start": e_start.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "estimate_detection_end": e_end.strftime("%Y-%m-%dT%H:%M:%S.%f")
            })

    return pd.DataFrame(rows, columns=COLUMNS)


def plot_histogram(values, title, x_label, output_path):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(values):
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ax.hist(values, bins=30, alpha=0.75, edgecolor="black")
        ax.axvspan(
            mean - std, mean + std, alpha=0.20,
            label=f"Mean ± 1 SD ({mean:.2f} ± {std:.2f})"
        )
        ax.axvline(mean, linestyle="-.", linewidth=2, label=f"Mean: {mean:.2f}")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center",
                transform=ax.transAxes)

    ax.axvline(0, linestyle="--", color="black", linewidth=1.2)
    ax.set(title=title, xlabel=x_label, ylabel="Event-station measurements")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_over_time(catalog, column, title, y_label, output_path):
    data = catalog[["start_time", column]].copy()
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[np.isfinite(data[column])]
    data["start_time"] = pd.to_datetime(data["start_time"])
    data = data.sort_values("start_time")

    fig, ax = plt.subplots(figsize=(12, 5))

    if len(data):
        ax.scatter(data["start_time"], data[column], s=8, alpha=0.45)
    else:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center",
                transform=ax.transAxes)

    ax.set(title=title, xlabel="Event time", ylabel=y_label)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    cfg, _args = get_cfg(__doc__)

    reference_root = store.stage_dir(cfg, "filter_then_denoise") / "weekly"
    estimate_root = store.stage_dir(cfg, "denoise_then_filter") / "weekly"

    reference_weeks = {
        path.name: path for path in reference_root.glob("week_*") if path.is_dir()
    }
    estimate_weeks = {
        path.name: path for path in estimate_root.glob("week_*") if path.is_dir()
    }
    common_weeks = sorted(set(reference_weeks) & set(estimate_weeks))

    if not common_weeks:
        raise RuntimeError("No matching weekly folders were found in both pipelines.")

    noise_seconds = float(cfg.trigger.lta_seconds)
    print(f"Weekly batches to compare: {len(common_weeks)}")
    print(f"Weeks included: {common_weeks}")
    print(f"Pre-event noise window: {noise_seconds} seconds")

    all_catalogs, next_event_number = [], 1

    for week_name in common_weeks:
        print(f"\nProcessing {week_name}")

        reference_week = reference_weeks[week_name]
        estimate_week = estimate_weeks[week_name]

        reference_traces = load_weekly_traces(reference_week, "filtered")
        estimate_traces = load_weekly_traces(estimate_week, "denoised")
        reference_detections = load_weekly_detections(reference_week)
        estimate_detections = load_weekly_detections(estimate_week)

        common_stations = set(reference_traces) & set(estimate_traces)
        if not common_stations:
            print(f"No common stations in {week_name}; skipping.")
            continue

        primary = (
            cfg.data.primary_station
            if cfg.data.primary_station in common_stations
            else sorted(common_stations)[0]
        )

        reference_intervals = triggers_to_intervals(
            reference_traces[primary],
            reference_detections["triggers_filtered"]
        )
        estimate_intervals = triggers_to_intervals(
            estimate_traces[primary],
            estimate_detections["triggers_denoised"]
        )

        matches = match_events(reference_intervals, estimate_intervals)

        print(f"Primary station: {primary}")
        print(f"Bandpass-filtered events: {len(reference_intervals)}")
        print(f"Denoised + highpass events: {len(estimate_intervals)}")
        print(f"Matched events: {len(matches)}")
        print(f"Stations compared: {sorted(common_stations)}")

        if not matches:
            print(f"No overlapping events in {week_name}; skipping.")
            continue

        week_catalog = build_comparison_catalog(
            cfg, reference_traces, estimate_traces,
            matches, week_name, next_event_number
        )
        all_catalogs.append(week_catalog)
        next_event_number += len(matches)

        print(f"Finished {week_name}: {len(week_catalog)} event-station measurements")

    if not all_catalogs:
        raise RuntimeError("No overlapping events were found in any weekly batch.")

    catalog = pd.concat(all_catalogs, ignore_index=True)

    output_dir = store.comparison_dir(cfg)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = output_dir / "catalog.csv"
    catalog.to_csv(
        catalog_path,
        index=False,
        float_format="%.5g" if bool(cfg.output.compact_csv_floats) else None
    )

    # Combined histograms for the entire set of weekly batches.
    histograms = [
        (
            "delta_snr_db",
            "SNR difference: denoised + highpass vs bandpass",
            "ΔSNR (estimate − reference, dB)",
            "snr_delta_histogram.png"
        ),
        (
            "sdr_db",
            "SDR: denoised + highpass relative to bandpass reference",
            "SDR (dB)",
            "sdr_histogram.png"
        ),
        (
            "max_correlation",
            "Peak CC: denoised + highpass vs bandpass reference",
            "Peak cross-correlation (−1 to +1)",
            "cc_histogram.png"
        )
    ]

    for column, title, label, filename in histograms:
        plot_histogram(catalog[column], title, label, figure_dir / filename)

    # Event-time plots across the full combined period.
    plot_metric_over_time(
        catalog, "sdr_db",
        "SDR through the full comparison period",
        "SDR (dB)",
        figure_dir / "sdr_over_time.png"
    )
    plot_metric_over_time(
        catalog, "max_correlation",
        "Peak CC through the full comparison period",
        "Peak cross-correlation (−1 to +1)",
        figure_dir / "cc_over_time.png"
    )

    print("\nComparison finished.")
    print(f"Weeks processed: {len(all_catalogs)}")
    print(f"Total matched events: {next_event_number - 1}")
    print(f"Total event-station measurements: {len(catalog)}")
    print(f"Saved comparison catalog: {catalog_path}")

    for metric, label, suffix in [
        ("delta_snr_db", "Median ΔSNR", " dB"),
        ("sdr_db", "Median SDR", " dB"),
        ("max_correlation", "Median peak CC", "")
    ]:
        values = catalog[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        print(f"{label}: {np.median(values):.3f}{suffix}" if len(values)
              else f"{label}: no finite values")
        print(f"  Finite measurements plotted: {len(values)}")

    print(f"\nFigures saved in: {figure_dir}")


if __name__ == "__main__":
    main()
