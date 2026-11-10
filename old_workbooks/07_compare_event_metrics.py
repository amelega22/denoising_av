"""Compare SNR and SDR for matched events using saved waveform traces.

Loads:
- Raw and denoised traces from denoise_then_filter
- Bandpass-filtered traces from filter_then_denoise
- Saved detections from both pipelines

Matches network detections by overlapping event intervals. For every match,
uses the denoised event interval as the common measurement window for raw,
bandpass-filtered, and denoised/highpass-filtered waveforms.

SNR uses the configured LTA duration (15 seconds in the supplied config)
as the pre-event noise window.

SDR is calculated twice per event and station:
- raw waveform = reference; bandpass-filtered waveform = estimate
- raw waveform = reference; denoised/highpass-filtered waveform = estimate
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags, resample_poly
from fractions import Fraction

from _common import get_cfg
from shishaldin import store
from shishaldin.detection import intervals_overlap, triggers_to_intervals
from shishaldin.metrics import window_metrics


# Maximum time shift considered when aligning waveforms for SDR.
MAX_LAG_SECONDS = 2.0
# Shared x-axis limits for SDR and CC histograms
HIST_X_MIN = -10
HIST_X_MAX = 10
HIST_X_TICKS = np.arange(-10, 11, 2)

def calculate_sdr(reference, estimate, sampling_rate,
                  max_lag_seconds=MAX_LAG_SECONDS):
    """Align estimate to reference, then return SDR, correlation, and lag.

    The sign of the correlation is retained. The lag is the shift applied
    to the estimate relative to the reference.
    """
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    # Use equal-length arrays before calculating cross-correlation.
    n = min(len(reference), len(estimate))
    reference = reference[:n]
    estimate = estimate[:n]

    if n == 0:
        raise ValueError("Cannot calculate SDR for an empty waveform.")

    # Remove each waveform's mean for the alignment calculation only.
    ref_centered = reference - np.mean(reference)
    est_centered = estimate - np.mean(estimate)

    if np.std(ref_centered) == 0 or np.std(est_centered) == 0:
        raise ValueError("Cannot align a constant waveform.")

    corr = correlate(
        est_centered,
        ref_centered,
        mode="full",
        method="fft",
    )
    lags = correlation_lags(
        len(est_centered),
        len(ref_centered),
        mode="full",
    )

    max_lag_samples = int(round(max_lag_seconds * sampling_rate))
    valid = np.abs(lags) <= max_lag_samples
    corr = corr[valid]
    lags = lags[valid]

    denominator = np.sqrt(
        np.sum(ref_centered ** 2) * np.sum(est_centered ** 2)
    )
    if denominator == 0:
        raise ValueError("Cannot normalize correlation for zero-energy data.")

    normalized_corr = corr / denominator
    best_index = int(np.argmax(np.abs(normalized_corr)))
    best_lag_samples = int(lags[best_index])
    best_correlation = float(normalized_corr[best_index])

    # Apply the best lag using the same convention as the original notebook.
    if best_lag_samples > 0:
        aligned_estimate = estimate[best_lag_samples:]
        aligned_reference = reference[:len(aligned_estimate)]
    elif best_lag_samples < 0:
        aligned_reference = reference[-best_lag_samples:]
        aligned_estimate = estimate[:len(aligned_reference)]
    else:
        aligned_reference = reference
        aligned_estimate = estimate

    n_aligned = min(len(aligned_reference), len(aligned_estimate))
    if n_aligned == 0:
        sdr_db = np.nan
    else:
        aligned_reference = aligned_reference[:n_aligned]
        aligned_estimate = aligned_estimate[:n_aligned]

        distortion_power = np.sum(
            (aligned_reference - aligned_estimate) ** 2
        )
        reference_power = np.sum(aligned_reference ** 2)

        if distortion_power == 0:
            sdr_db = float("inf")
        elif reference_power == 0:
            sdr_db = float("-inf")
        else:
            sdr_db = float(
                10 * np.log10(reference_power / distortion_power)
            )

    return {
        "sdr_db": sdr_db,
        "max_correlation": best_correlation,
        "best_lag_seconds": best_lag_samples / sampling_rate,
    }


def extract_event_data(trace, start, end):
    """Return a copy of the trace trimmed to the requested event window."""
    event = trace.copy()
    event.trim(starttime=start, endtime=end, pad=False)
    return event


def calculate_event_sdr(reference_trace, estimate_trace, start, end):
    """Calculate SDR for two traces over the same absolute-time event window."""
    reference_event = extract_event_data(reference_trace, start, end)
    estimate_event = extract_event_data(estimate_trace, start, end)

    ref_rate = float(reference_event.stats.sampling_rate)
    est_rate = float(estimate_event.stats.sampling_rate)

    if not np.isclose(ref_rate, est_rate):
        # Resample the estimate onto the reference sampling rate.
        # For example, 100 Hz -> 50 Hz uses a 1/2 polyphase conversion.
        ratio = Fraction(ref_rate / est_rate).limit_denominator(10000)
        estimate_event.data = resample_poly(
            estimate_event.data.astype(float),
            up=ratio.numerator,
            down=ratio.denominator,
        )
        estimate_event.stats.sampling_rate = ref_rate

    # ObsPy trims each trace on its own sample grid. Truncate to the common
    # available number of samples if their endpoints differ by one sample.
    n = min(len(reference_event.data), len(estimate_event.data))
    if n == 0:
        raise ValueError("Event window contains no overlapping samples.")

    return calculate_sdr(
        reference_event.data[:n],
        estimate_event.data[:n],
        ref_rate,
    )


def match_events(filtered_intervals, denoised_intervals):
    """Match each denoised interval to the first unused overlapping filter interval."""
    used_filtered = set()
    matches = []

    for d_start, d_end in denoised_intervals:
        for f_index, (f_start, f_end) in enumerate(filtered_intervals):
            if f_index in used_filtered:
                continue

            if intervals_overlap(d_start, d_end, f_start, f_end):
                used_filtered.add(f_index)
                matches.append((d_start, d_end, f_start, f_end))
                break

    return matches


def calculate_event_metrics(trace, start, end, noise_seconds):
    """Use the project's existing SNR/RMS measurement function."""
    signal_rms, noise_rms, snr_db, max_amplitude = window_metrics(
        trace,
        start,
        end,
        noise_seconds,
    )

    return {
        "signal_rms": float(signal_rms),
        "noise_rms": float(noise_rms),
        "snr_db": float(snr_db),
        "max_amplitude": float(max_amplitude),
    }


def build_comparison_catalog(
    cfg,
    raw_traces,
    bandpass_traces,
    denoised_traces,
    matches,
):
    """Build one row per event, station, and waveform type."""
    noise_seconds = float(cfg.trigger.lta_seconds)

    # A station is usable only if all three waveform versions are available.
    stations = sorted(
        set(raw_traces)
        & set(bandpass_traces)
        & set(denoised_traces)
    )
    if not stations:
        raise RuntimeError(
            "No stations have raw, bandpass-filtered, and "
            "denoised/highpass-filtered traces."
        )

    rows = []

    for event_number, (d_start, d_end, f_start, f_end) in enumerate(
        matches, start=1
    ):
        # Use the denoised event's interval as the shared measurement window.
        start, end = d_start, d_end

        for station in stations:
            traces = {
                "raw": raw_traces[station],
                "bandpass_filtered": bandpass_traces[station],
                "denoised_highpass_filtered": denoised_traces[station],
            }

            measurements = {
                kind: calculate_event_metrics(
                    trace, start, end, noise_seconds
                )
                for kind, trace in traces.items()
            }

            # Both SDR calculations use raw as the reference.
            # The processed waveform is the estimate.
            try:
                bandpass_sdr = calculate_event_sdr(
                    raw_traces[station],
                    bandpass_traces[station],
                    start,
                    end,
                )
            except ValueError as exc:
                print(
                    f"Event {event_number}, {station}, bandpass SDR "
                    f"skipped: {exc}"
                )
                bandpass_sdr = {
                    "sdr_db": np.nan,
                    "max_correlation": np.nan,
                    "best_lag_seconds": np.nan,
                }

            try:
                denoised_sdr = calculate_event_sdr(
                    raw_traces[station],
                    denoised_traces[station],
                    start,
                    end,
                )
            except ValueError as exc:
                print(
                    f"Event {event_number}, {station}, denoised SDR "
                    f"skipped: {exc}"
                )
                denoised_sdr = {
                    "sdr_db": np.nan,
                    "max_correlation": np.nan,
                    "best_lag_seconds": np.nan,
                }

            raw_snr = measurements["raw"]["snr_db"]

            for kind, metrics in measurements.items():
                if kind == "raw":
                    delta_snr = 0.0
                    sdr_metrics = {
                        "sdr_db": np.nan,
                        "max_correlation": np.nan,
                        "best_lag_seconds": np.nan,
                    }
                elif kind == "bandpass_filtered":
                    processed_snr = metrics["snr_db"]
                    delta_snr = (
                        processed_snr - raw_snr
                        if np.isfinite(processed_snr)
                        and np.isfinite(raw_snr)
                        else np.nan
                    )
                    sdr_metrics = bandpass_sdr
                else:
                    processed_snr = metrics["snr_db"]
                    delta_snr = (
                        processed_snr - raw_snr
                        if np.isfinite(processed_snr)
                        and np.isfinite(raw_snr)
                        else np.nan
                    )
                    sdr_metrics = denoised_sdr

                rows.append({
                    "event_number": event_number,
                    "station": station,
                    "detection_type": kind,
                    "start_time": start.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                    "end_time": end.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                    "duration_seconds": float(end - start),
                    "signal_rms": metrics["signal_rms"],
                    "noise_rms": metrics["noise_rms"],
                    "snr_db": metrics["snr_db"],
                    "max_amplitude": metrics["max_amplitude"],
                    "reference_snr_db": raw_snr,
                    "delta_snr_db": delta_snr,
                    "sdr_db": sdr_metrics["sdr_db"],
                    "max_correlation": sdr_metrics["max_correlation"],
                    "best_lag_seconds": sdr_metrics["best_lag_seconds"],
                    "filtered_start_time": f_start.strftime(
                        "%Y-%m-%dT%H:%M:%S.%f"
                    ),
                    "filtered_end_time": f_end.strftime(
                        "%Y-%m-%dT%H:%M:%S.%f"
                    ),
                })

    return pd.DataFrame(rows)


def _plot_two_histograms(
    first_values,
    second_values,
    first_title,
    second_title,
    x_label,
    y_label,
    figure_title,
    output_path,
):
    """Plot two distributions side by side, each with mean and ±1 SD."""
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 5),
        sharey=True,
    )

    datasets = [
        (axes[0], first_values, first_title),
        (axes[1], second_values, second_title),
    ]
    # Calculate shared x-axis limits from both distributions
    combined = np.concatenate([
        first_values[np.isfinite(first_values)],
        second_values[np.isfinite(second_values)],
    ])

    if len(combined):
        x_min = combined.min()
        x_max = combined.max()

        # Add 5% padding on either side
        padding = (x_max - x_min) * 0.05
        if padding == 0:
            padding = 1.0

        x_min -= padding
        x_max += padding

        # Apply identical limits to both panels
        for ax in axes:
            ax.set_xlim(x_min, x_max)

    for ax, values, title in datasets:
        values = values[np.isfinite(values)]

        if len(values) == 0:
            ax.text(
                0.5, 0.5, "No finite values",
                ha="center", va="center",
                transform=ax.transAxes,
            )
        else:
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0

            ax.hist(
                values,
                bins=20,
                alpha=0.75,
                edgecolor="black",
            )

            ax.axvspan(
                mean - std,
                mean + std,
                alpha=0.20,
                label=f"Mean ± 1 SD ({mean:.2f} ± {std:.2f})",
            )
            ax.axvline(
                mean,
                linestyle="-.",
                linewidth=2,
                label=f"Mean: {mean:.2f}",
            )
            ax.legend(fontsize=8)

        ax.axvline(
            0,
            linestyle="--",
            color="black",
            linewidth=1.2,
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(y_label)
    fig.suptitle(figure_title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_snr_differences(catalog, output_path):
    """Plot processed-minus-raw SNR distributions side by side."""
    bandpass = catalog.loc[
        catalog["detection_type"] == "bandpass_filtered",
        "delta_snr_db",
    ].to_numpy(dtype=float)

    denoised = catalog.loc[
        catalog["detection_type"] == "denoised_highpass_filtered",
        "delta_snr_db",
    ].to_numpy(dtype=float)

    _plot_two_histograms(
        bandpass,
        denoised,
        "Bandpass filtered vs. raw",
        "Denoised then highpass vs. raw",
        "ΔSNR relative to raw (dB)",
        "Number of event-station measurements",
        "SNR improvement relative to raw waveforms",
        output_path,
    )


def plot_sdr_distributions(catalog, output_path):
    """Plot SDR distributions side by side, using raw as reference."""
    bandpass = catalog.loc[
        catalog["detection_type"] == "bandpass_filtered",
        "sdr_db",
    ].to_numpy(dtype=float)

    denoised = catalog.loc[
        catalog["detection_type"] == "denoised_highpass_filtered",
        "sdr_db",
    ].to_numpy(dtype=float)

    # Infinite SDR occurs when estimate and reference are identical.
    n_infinite = int(
        np.isinf(bandpass).sum() + np.isinf(denoised).sum()
    )
    if n_infinite:
        print(
            f"Note: excluded {n_infinite} infinite SDR measurement(s) "
            "from the histograms."
        )

    _plot_two_histograms(
        bandpass,
        denoised,
        "Bandpass filtered vs. raw",
        "Denoised then highpass vs. raw",
        "SDR relative to raw (dB)",
        "Number of event-station measurements",
        "Signal-to-distortion ratio relative to raw waveforms",
        output_path,
    )

def plot_cc_distributions(catalog, output_path):
    """Plot peak cross-correlation distributions for both comparisons."""

    bandpass = catalog.loc[
        catalog["detection_type"] == "bandpass_filtered",
        "max_correlation",
    ].to_numpy(dtype=float)

    denoised = catalog.loc[
        catalog["detection_type"] == "denoised_highpass_filtered",
        "max_correlation",
    ].to_numpy(dtype=float)

    _plot_two_histograms(
        bandpass,
        denoised,
        "Bandpass filtered vs. raw",
        "Denoised then highpass vs. raw",
        "Peak cross-correlation (CC)",
        "Number of event-station measurements",
        "Peak cross-correlation relative to raw waveforms",
        output_path,
    )

def main():
    cfg, _args = get_cfg(__doc__)

    # Raw and denoised traces come from the denoise-then-filter run.
    raw_traces = store.load_traces(
        cfg, "denoise_then_filter", "raw"
    )
    denoised_traces = store.load_traces(
        cfg, "denoise_then_filter", "denoised"
    )

    # Bandpass-filtered traces come from the filter-then-denoise run.
    bandpass_traces = store.load_traces(
        cfg, "filter_then_denoise", "filtered"
    )

    # Load detections saved by Stage 2 for each processing pipeline.
    denoised_detections = store.load_detections(
        cfg, "denoise_then_filter"
    )
    bandpass_detections = store.load_detections(
        cfg, "filter_then_denoise"
    )

    # Network detections are represented by trigger intervals on the
    # primary station trace. Fall back to an available station if needed.
    denoised_primary = (
        cfg.data.primary_station
        if cfg.data.primary_station in denoised_traces
        else sorted(denoised_traces)[0]
    )
    bandpass_primary = (
        cfg.data.primary_station
        if cfg.data.primary_station in bandpass_traces
        else sorted(bandpass_traces)[0]
    )

    denoised_intervals = triggers_to_intervals(
        denoised_traces[denoised_primary],
        denoised_detections["triggers_denoised"],
    )
    bandpass_intervals = triggers_to_intervals(
        bandpass_traces[bandpass_primary],
        bandpass_detections["triggers_filtered"],
    )

    matches = match_events(
        bandpass_intervals,
        denoised_intervals,
    )

    print(f"Denoised events: {len(denoised_intervals)}")
    print(f"Bandpass events: {len(bandpass_intervals)}")
    print(f"Matched events: {len(matches)}")
    print(f"Pre-event noise window: {cfg.trigger.lta_seconds} seconds")

    if not matches:
        raise RuntimeError(
            "No overlapping denoised and bandpass events were found."
        )

    catalog = build_comparison_catalog(
        cfg,
        raw_traces,
        bandpass_traces,
        denoised_traces,
        matches,
    )

    output_dir = store.comparison_dir(cfg)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = output_dir / "catalog.csv"
    compact = bool(cfg.output.compact_csv_floats)
    if compact:
        catalog.to_csv(catalog_path, index=False, float_format="%.2g")
    else:
        catalog.to_csv(catalog_path, index=False)

    snr_figure_path = figure_dir / "snr_delta_histogram.png"
    plot_snr_differences(catalog, snr_figure_path)

    sdr_figure_path = figure_dir / "sdr_histograms.png"
    plot_sdr_distributions(catalog, sdr_figure_path)
    cc_figure_path = figure_dir / "cc_histograms.png"
    plot_cc_distributions(catalog, cc_figure_path)

    print(f"\nSaved comparison catalog: {catalog_path}")
    print(f"Saved SNR histogram: {snr_figure_path}")
    print(f"Saved SDR histograms: {sdr_figure_path}")
    print(f"Saved CC histograms: {cc_figure_path}")

    print("\nMedian ΔSNR relative to raw:")
    for kind in (
        "bandpass_filtered",
        "denoised_highpass_filtered",
    ):
        values = catalog.loc[
            catalog["detection_type"] == kind,
            "delta_snr_db",
        ].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            print(f"  {kind}: {np.median(values):.2f} dB")
        else:
            print(f"  {kind}: no finite values")

    print("\nMedian SDR relative to raw:")
    for kind in (
        "bandpass_filtered",
        "denoised_highpass_filtered",
    ):
        values = catalog.loc[
            catalog["detection_type"] == kind,
            "sdr_db",
        ].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            print(f"  {kind}: {np.median(values):.2f} dB")
        else:
            print(f"  {kind}: no finite values")

    print("\nMedian peak cross-correlation:")
    for kind in (
            "bandpass_filtered",
            "denoised_highpass_filtered",
    ):
        values = catalog.loc[
            catalog["detection_type"] == kind,
            "max_correlation",
        ].to_numpy(dtype=float)

        values = values[np.isfinite(values)]

        if len(values):
            print(f"  {kind}: {np.median(values):.3f}")
        else:
            print(f"  {kind}: no finite values")


if __name__ == "__main__":
    main()
