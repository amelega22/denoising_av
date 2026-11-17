
"""
Compare bandpass-filtered data with denoised + highpass-filtered data.

Loads saved weekly traces and detections only; neither processing pipeline
is rerun.

Uses the same overlapping event window and preceding LTA-length noise
window for both SNR measurements.
"""

from fractions import Fraction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from .config import Config
from scipy.signal import correlate, correlation_lags, resample_poly
from obspy import read

from _common import get_cfg
from shishaldin import store
from shishaldin.detection import triggers_to_intervals
from shishaldin.metrics import window_metrics


MAX_LAG_SECONDS = 2.0

COLUMNS = [
    "week",
    "event_number",
    "station",
    "reference_type",
    "estimate_type",
    "start_time",
    "end_time",
    "duration_seconds",
    "reference_signal_rms",
    "reference_noise_rms",
    "reference_snr_db",
    "estimate_signal_rms",
    "estimate_noise_rms",
    "estimate_snr_db",
    "delta_snr_db",
    "sdr_db",
    "max_correlation",
    "best_lag_seconds",
    "reference_detection_start",
    "reference_detection_end",
    "estimate_detection_start",
    "estimate_detection_end",
]


def load_weekly_traces(week_dir, kind):
    """Load all station traces of one type from a weekly folder."""

    traces = {}

    for path in sorted((week_dir / "traces").glob(f"{kind}.*.mseed")):
        station = path.stem.split(".", 1)[1]
        stream = read(str(path))

        if len(stream) != 1:
            raise RuntimeError(
                f"Expected one trace in {path}, found {len(stream)}."
            )

        traces[station] = stream[0]

    return traces


def _time_axis(ax, fmt="%H:%M"):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def _wave_axis(ax, cfg: Config, sci=(-6, -6)):
    ax.set_ylabel("Velocity (m/s)")
    ax.set_ylim(cfg.plot.waveform_ylim)
    ax.set_yticks(cfg.plot.waveform_yticks)
    ax.ticklabel_format(axis="y", style="sci", scilimits=sci)


def _make_spectrogram(tr, cfg: Config):
    p = cfg.plot
    fs = tr.stats.sampling_rate
    nper = int(p.spectrogram_window_seconds * fs)

    f, t, Pspec = spectrogram(
        tr.data,
        fs=fs,
        window="hann",
        scaling="density",
        nperseg=nper,
        noverlap=int(nper * p.spectrogram_overlap),
    )

    tdnum = mdates.date2num(tr.stats.starttime.datetime) + t / 86400.0
    PdB = 10 * np.log10(
        np.abs(Pspec) / np.power(p.spectrogram_ref, 2)
    )

    return f, tdnum, PdB


def _shared_clim(specs, cfg: Config):
    lo, hi = cfg.plot.spectrogram_ylim

    pooled = np.concatenate([
        PdB[(f >= lo) & (f <= hi), :].ravel()
        for f, _, PdB in specs
    ])

    pooled = pooled[np.isfinite(pooled)]

    return np.percentile(pooled, 35), np.percentile(pooled, 97.5)


def plot_waveforms_spectrograms(
    raw,
    filtered,
    denoised,
    cfg: Config,
    labels=("Raw Data", "Filtered Data", "Denoised Data"),
):
    p = cfg.plot
    traces = (raw, filtered, denoised)
    specs = [_make_spectrogram(tr, cfg) for tr in traces]

    # Optional manual override: plot.spectrogram_clim in config.yaml
    clim = getattr(p, "spectrogram_clim", None) or _shared_clim(
        specs, cfg
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(18, 9),
        sharex=True,
        constrained_layout=True,
    )

    for r, (title, tr, (f, tdnum, PdB)) in enumerate(
        zip(labels, traces, specs)
    ):
        ax = axes[r, 0]
        ax.plot(
            tr.times("matplotlib"),
            tr.data,
            linewidth=0.6,
            color="black",
        )
        ax.set_title(title, fontsize=16)
        _wave_axis(ax, cfg)
        ax.grid(True, alpha=0.3)

        ax = axes[r, 1]
        im = ax.imshow(
            PdB,
            extent=[tdnum[0], tdnum[-1], f[0], f[-1]],
            origin="lower",
            aspect="auto",
            interpolation="none",
            cmap=p.colormap,
            vmin=clim[0],
            vmax=clim[1],
        )
        ax.set_ylim(p.spectrogram_ylim)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{title} spectrogram", fontsize=16)

    # One colorbar spanning all three spectrogram panels
    fig.colorbar(
        im,
        ax=axes[:, 1],
        label="Power (dB)",
        pad=0.01,
    )

    for ax in axes.flat:
        ax.set_xlim(
            cfg.start.matplotlib_date,
            cfg.stop.matplotlib_date,
        )
        ax.xaxis_date()
        _time_axis(ax)

    axes[2, 0].set_xlabel("UTC Time")
    axes[2, 1].set_xlabel("UTC Time")

    return fig

def load_weekly_detections(week_dir):
    """Load the weekly detection arrays."""

    path = week_dir / "detections.npz"

    if not path.exists():
        raise FileNotFoundError(f"Missing weekly detection file: {path}")

    with np.load(path) as data:
        return {
            "triggers_filtered": np.asarray(
                data["triggers_filtered"], dtype=int
            ),
            "triggers_denoised": np.asarray(
                data["triggers_denoised"], dtype=int
            ),
        }


def calculate_sdr(
    reference,
    estimate,
    sampling_rate,
    max_lag_seconds=MAX_LAG_SECONDS,
):
    """Calculate SDR after finding the best waveform alignment."""

    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    n = min(len(reference), len(estimate))
    reference = reference[:n]
    estimate = estimate[:n]

    if n == 0:
        raise ValueError("Cannot calculate SDR for an empty waveform.")

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
        raise ValueError(
            "Cannot normalize correlation for zero-energy data."
        )

    normalized_corr = corr / denominator
    best_index = int(np.argmax(np.abs(normalized_corr)))

    best_lag_samples = int(lags[best_index])
    best_correlation = float(normalized_corr[best_index])

    # Apply the best lag to the estimate.
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
    """Extract an event using array indices without copying the full trace."""

    sampling_rate = float(trace.stats.sampling_rate)
    trace_start = trace.stats.starttime

    first = int(round(float(start - trace_start) * sampling_rate))
    last = int(round(float(end - trace_start) * sampling_rate))

    first = max(0, first)
    last = min(len(trace.data) - 1, last)

    if last < first:
        return np.array([], dtype=float)

    return np.asarray(trace.data[first:last + 1], dtype=float)


def calculate_event_sdr(reference_trace, estimate_trace, start, end):
    """Calculate SDR for the common event window."""

    reference_event = extract_event_data(reference_trace, start, end)
    estimate_event = extract_event_data(estimate_trace, start, end)

    ref_rate = float(reference_trace.stats.sampling_rate)
    est_rate = float(estimate_trace.stats.sampling_rate)

    if len(reference_event) == 0 or len(estimate_event) == 0:
        raise ValueError("Event window contains no overlapping samples.")

    if not np.isclose(ref_rate, est_rate):
        ratio = Fraction(ref_rate / est_rate).limit_denominator(10000)

        estimate_event = resample_poly(
            estimate_event,
            up=ratio.numerator,
            down=ratio.denominator,
        )

    n = min(len(reference_event), len(estimate_event))

    if n == 0:
        raise ValueError("Event window contains no overlapping samples.")

    return calculate_sdr(
        reference_event[:n],
        estimate_event[:n],
        ref_rate,
    )


def match_events(reference_intervals, estimate_intervals):
    """
    Match overlapping events chronologically.

    This avoids comparing every estimate against every reference.
    Each reference event can only be matched once.
    """

    reference_order = sorted(
        range(len(reference_intervals)),
        key=lambda i: reference_intervals[i][0],
    )

    estimate_order = sorted(
        range(len(estimate_intervals)),
        key=lambda i: estimate_intervals[i][0],
    )

    active = []
    next_reference = 0
    matches = []

    for estimate_index in estimate_order:
        e_start, e_end = estimate_intervals[estimate_index]

        # Add reference events that have started before this estimate ends.
        while next_reference < len(reference_order):
            r_index = reference_order[next_reference]
            r_start, r_end = reference_intervals[r_index]

            if r_start >= e_end:
                break

            active.append((r_index, r_start, r_end))
            next_reference += 1

        # Remove reference events that have already ended.
        active = [
            item for item in active
            if item[2] > e_start
        ]

        # Preserve the original behavior: choose the first available
        # overlapping reference event in original reference order.
        candidates = [
            item for item in active
            if item[1] < e_end and item[2] > e_start
        ]

        if not candidates:
            continue

        r_index, r_start, r_end = min(
            candidates,
            key=lambda item: item[0],
        )

        active = [
            item for item in active
            if item[0] != r_index
        ]

        start = max(e_start, r_start)
        end = min(e_end, r_end)

        if end > start:
            matches.append(
                (
                    start,
                    end,
                    r_start,
                    r_end,
                    e_start,
                    e_end,
                )
            )

    return matches


def calculate_event_snr(trace, start, end, noise_seconds):
    """Calculate SNR for the common event window."""

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
    reference_traces,
    estimate_traces,
    matches,
    week_name,
    first_event_number,
):
    """Calculate comparison measurements for all matched events."""

    noise_seconds = float(cfg.trigger.lta_seconds)

    stations = sorted(
        set(reference_traces) & set(estimate_traces)
    )

    if not stations:
        raise RuntimeError(
            "No stations have both the bandpass reference and "
            "denoised/highpass estimate traces."
        )

    rows = []

    for match_index, match in enumerate(matches):
        event_number = first_event_number + match_index

        start, end, r_start, r_end, e_start, e_end = match

        for station in stations:
            reference_trace = reference_traces[station]
            estimate_trace = estimate_traces[station]

            reference_metrics = calculate_event_snr(
                reference_trace,
                start,
                end,
                noise_seconds,
            )

            estimate_metrics = calculate_event_snr(
                estimate_trace,
                start,
                end,
                noise_seconds,
            )

            try:
                waveform_metrics = calculate_event_sdr(
                    reference_trace,
                    estimate_trace,
                    start,
                    end,
                )

            except ValueError as exc:
                print(f"Event {event_number}, {station}: "f"waveform comparison skipped: {exc}")
                waveform_metrics = {"sdr_db": np.nan, "max_correlation": np.nan,"best_lag_seconds": np.nan}

            reference_snr = reference_metrics["snr_db"]
            estimate_snr = estimate_metrics["snr_db"]

            delta_snr = (
                estimate_snr - reference_snr
                if np.isfinite(estimate_snr)
                and np.isfinite(reference_snr)
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

                "reference_detection_start": r_start.strftime(
                    "%Y-%m-%dT%H:%M:%S.%f"
                ),
                "reference_detection_end": r_end.strftime(
                    "%Y-%m-%dT%H:%M:%S.%f"
                ),
                "estimate_detection_start": e_start.strftime(
                    "%Y-%m-%dT%H:%M:%S.%f"
                ),
                "estimate_detection_end": e_end.strftime(
                    "%Y-%m-%dT%H:%M:%S.%f"
                ),
            })

    return pd.DataFrame(rows, columns=COLUMNS)


def plot_histogram(values, title, x_label, output_path):
    """Plot a histogram with mean and standard deviation."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(values) == 0:
        ax.text(
            0.5,
            0.5,
            "No finite values",
            ha="center",
            va="center",
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

        ax.legend(fontsize=9)

    ax.axvline(
        0,
        linestyle="--",
        color="black",
        linewidth=1.2,
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of event-station measurements")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    cfg, _args = get_cfg(__doc__)

    reference_weekly_dir = (
        store.stage_dir(cfg, "filter_then_denoise") / "weekly"
    )

    estimate_weekly_dir = (
        store.stage_dir(cfg, "denoise_then_filter") / "weekly"
    )

    reference_weeks = {
        path.name: path
        for path in reference_weekly_dir.glob("week_*")
        if path.is_dir()
    }

    estimate_weeks = {
        path.name: path
        for path in estimate_weekly_dir.glob("week_*")
        if path.is_dir()
    }

    common_weeks = sorted(
        set(reference_weeks) & set(estimate_weeks)
    )

    if not common_weeks:
        raise RuntimeError(
            "No matching weekly folders were found in both pipelines."
        )

    noise_seconds = float(cfg.trigger.lta_seconds)

    print(f"Weekly batches to compare: {len(common_weeks)}")
    print(f"Pre-event noise window: {noise_seconds} seconds")

    all_catalogs = []
    next_event_number = 1

    for week_name in common_weeks:
        print(f"\nProcessing {week_name}")

        reference_week_dir = reference_weeks[week_name]
        estimate_week_dir = estimate_weeks[week_name]
        week_number = int(week_name.removeprefix("week_"))

        # Reference: bandpass-filtered traces from Stage 1A.
        reference_traces = store.load_weekly_traces(
            cfg, "filter_then_denoise", week_number, "filtered"
        )

        # Estimate: denoised + highpass-filtered traces from Stage 1B.
        estimate_traces = store.load_weekly_traces(
            cfg, "denoise_then_filter", week_number, "denoised"
        )

        reference_detections = load_weekly_detections(
            reference_week_dir
        )

        estimate_detections = load_weekly_detections(
            estimate_week_dir
        )

        common_stations = (
            set(reference_traces) & set(estimate_traces)
        )

        if not common_stations:
            print(f"No common stations in {week_name}; skipping.")
            continue

        primary = (
            cfg.data.primary_station
            if cfg.data.primary_station in common_stations
            else sorted(common_stations)[0]
        )

        # Convert trigger sample indices to integers before making intervals.
        reference_triggers = reference_detections[
            "triggers_filtered"
        ]

        estimate_triggers = estimate_detections[
            "triggers_denoised"
        ]

        reference_intervals = triggers_to_intervals(
            reference_traces[primary],
            reference_triggers,
        )

        estimate_intervals = triggers_to_intervals(
            estimate_traces[primary],
            estimate_triggers,
        )

        matches = match_events(
            reference_intervals,
            estimate_intervals,
        )

        print(f"Primary station: {primary}")
        print(f"Bandpass-filtered events: {len(reference_intervals)}")
        print(f"Denoised + highpass events: {len(estimate_intervals)}")
        print(f"Matched events: {len(matches)}")
        print(f"Stations compared: {sorted(common_stations)}")

        if not matches:
            print(f"No overlapping events in {week_name}; skipping.")
            continue

        week_catalog = build_comparison_catalog(
            cfg,
            reference_traces,
            estimate_traces,
            matches,
            week_name,
            next_event_number,
        )

        all_catalogs.append(week_catalog)
        next_event_number += len(matches)

        print(
            f"Finished {week_name}: "
            f"{len(week_catalog)} event-station measurements"
        )

    if not all_catalogs:
        raise RuntimeError(
            "No overlapping events were found in any weekly batch."
        )

    catalog = pd.concat(
        all_catalogs,
        ignore_index=True,
    )

    output_dir = store.comparison_dir(cfg)
    figure_dir = output_dir / "figures"

    figure_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = output_dir / "catalog.csv"

    if bool(cfg.output.compact_csv_floats):
        catalog.to_csv(
            catalog_path,
            index=False,
            float_format="%.5g",
        )
    else:
        catalog.to_csv(
            catalog_path,
            index=False,
        )

    # Save one histogram for each comparison metric.
    snr_figure_path = figure_dir / "snr_delta_histogram.png"
    sdr_figure_path = figure_dir / "sdr_histogram.png"
    cc_figure_path = figure_dir / "cc_histogram.png"

    plot_histogram(
        catalog["delta_snr_db"],
        "SNR difference: denoised + highpass vs bandpass",
        "ΔSNR (estimate − reference, dB)",
        snr_figure_path,
    )

    plot_histogram(
        catalog["sdr_db"],
        "SDR: denoised + highpass relative to bandpass reference",
        "SDR (dB)",
        sdr_figure_path,
    )

    plot_histogram(catalog["max_correlation"],
        "Peak CC: denoised + highpass vs bandpass reference",
        "Peak cross-correlation (CC)",
        cc_figure_path,
    )

    print("\nComparison finished.")
    print(f"Total matched events: {next_event_number - 1}")
    print(f"Total event-station measurements: {len(catalog)}")
    print(f"Saved comparison catalog: {catalog_path}")
    print(f"Saved SNR histogram: {snr_figure_path}")
    print(f"Saved SDR histogram: {sdr_figure_path}")
    print(f"Saved CC histogram: {cc_figure_path}")

    for metric, label, suffix in (("delta_snr_db", "Median ΔSNR (estimate − reference)", " dB"), ("sdr_db", "Median SDR relative to bandpass reference", " dB"), ("max_correlation", "Median peak cross-correlation", "")):
        values = catalog[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values):
            print(f"\n{label}: {np.median(values):.3f}{suffix}")
        else:
            print(f"\n{label}: no finite values")


if __name__ == "__main__":
    main()
