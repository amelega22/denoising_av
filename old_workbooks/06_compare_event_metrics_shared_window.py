# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal as scipy_signal
from obspy import UTCDateTime, read

from _common import get_cfg
from shishaldin import store


# %% Settings
MAX_LAG_SECONDS = 0.5

CATALOG_COLUMNS = [
    "event_number", "detection_type", "start_time", "end_time",
    "duration_seconds", "signal_rms", "noise_rms", "snr_db",
    "max_amplitude", "extra"
]

COLUMNS = [
    "event_number", "week",
    "reference_type", "estimate_type",
    "start_time", "end_time", "duration_seconds",
    "reference_signal_rms", "reference_noise_rms", "reference_snr_db",
    "estimate_signal_rms", "estimate_noise_rms", "estimate_snr_db",
    "delta_snr_db", "sdr_db", "max_correlation", "best_lag_seconds",
    "reference_event_number", "estimate_event_number",
    "reference_detection_start", "reference_detection_end",
    "estimate_detection_start", "estimate_detection_end"
]


# %% Load catalogs
def load_catalog(path, detection_type):
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")

    catalog = pd.read_csv(
        path,
        header=None,
        names=CATALOG_COLUMNS
    )

    catalog = catalog[
        catalog["detection_type"].astype(str).str.strip() == detection_type
    ].copy()

    if catalog.empty:
        raise RuntimeError(
            f"No '{detection_type}' events found in {path}"
        )

    catalog["start_time"] = pd.to_datetime(catalog["start_time"])
    catalog["end_time"] = pd.to_datetime(catalog["end_time"])

    for column in [
        "signal_rms", "noise_rms", "snr_db", "max_amplitude"
    ]:
        catalog[column] = pd.to_numeric(
            catalog[column], errors="coerce"
        )

    return catalog.sort_values("start_time").reset_index(drop=True)


# %% Load primary-station waveforms
def load_primary_trace(week_dir, kind, primary_station):
    paths = sorted(
        (week_dir / "traces").glob(f"{kind}.*.mseed")
    )

    for path in paths:
        station = path.stem.split(".", 1)[1]

        if station == primary_station:
            stream = read(str(path))

            if len(stream) != 1:
                raise RuntimeError(
                    f"Expected one trace in {path}, found {len(stream)}."
                )

            return stream[0]

    raise FileNotFoundError(
        f"No {kind} waveform found for primary station "
        f"{primary_station} in {week_dir}"
    )


def catalog_rows_for_week(catalog, trace):
    trace_start = pd.Timestamp(trace.stats.starttime.datetime)
    trace_end = pd.Timestamp(trace.stats.endtime.datetime)

    return catalog[
        (catalog["start_time"] <= trace_end)
        & (catalog["end_time"] >= trace_start)
    ].copy()


# %% Match catalog events by overlapping intervals
def match_events(reference_catalog, estimate_catalog):
    reference_rows = list(reference_catalog.iterrows())
    estimate_rows = list(estimate_catalog.iterrows())

    active = []
    next_reference = 0
    matches = []

    for estimate_index, estimate_row in estimate_rows:
        e_start = estimate_row["start_time"]
        e_end = estimate_row["end_time"]

        while next_reference < len(reference_rows):
            reference_index, reference_row = reference_rows[next_reference]

            if reference_row["start_time"] >= e_end:
                break

            active.append((reference_index, reference_row))
            next_reference += 1

        active = [
            item for item in active
            if item[1]["end_time"] > e_start
        ]

        candidates = [
            item for item in active
            if item[1]["start_time"] < e_end
            and item[1]["end_time"] > e_start
        ]

        if not candidates:
            continue

        # Select the reference event with the largest temporal overlap.
        reference_index, reference_row = max(
            candidates,
            key=lambda item: (
                min(item[1]["end_time"], e_end)
                - max(item[1]["start_time"], e_start)
            )
        )

        active = [
            item for item in active
            if item[0] != reference_index
        ]

        overlap_start = max(reference_row["start_time"], e_start)
        overlap_end = min(reference_row["end_time"], e_end)

        if overlap_end > overlap_start:
            matches.append({
                "reference": reference_row,
                "estimate": estimate_row,
                "start": overlap_start,
                "end": overlap_end
            })

    return matches


# %% Extract matching waveform samples
def extract_common_event_waveforms(
    reference_trace, estimate_trace, start, end
):
    ref_start = reference_trace.stats.starttime
    est_start = estimate_trace.stats.starttime

    ref_rate = float(reference_trace.stats.sampling_rate)
    est_rate = float(estimate_trace.stats.sampling_rate)

    ref_end = ref_start + (len(reference_trace.data) - 1) / ref_rate
    est_end = est_start + (len(estimate_trace.data) - 1) / est_rate

    start_utc = UTCDateTime(start.to_pydatetime())
    end_utc = UTCDateTime(end.to_pydatetime())

    common_start = max(start_utc, ref_start, est_start)
    common_end = min(end_utc, ref_end, est_end)

    if common_end <= common_start:
        raise ValueError("No common waveform coverage for this event.")

    # Use the lower sampling rate as the common comparison rate.
    sampling_rate = min(ref_rate, est_rate)

    count = int(
        np.floor(float(common_end - common_start) * sampling_rate)
    ) + 1

    offsets = np.arange(count) / sampling_rate

    ref_indices = (
        float(common_start - ref_start) + offsets
    ) * ref_rate

    est_indices = (
        float(common_start - est_start) + offsets
    ) * est_rate

    reference = np.interp(
        ref_indices,
        np.arange(len(reference_trace.data)),
        np.asarray(reference_trace.data, dtype=float)
    )

    estimate = np.interp(
        est_indices,
        np.arange(len(estimate_trace.data)),
        np.asarray(estimate_trace.data, dtype=float)
    )

    return (
        reference,
        estimate,
        sampling_rate,
        common_start,
        common_end
    )


# %% SDR and cross-correlation
def signal_distortion_ratio(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    if reference.shape != estimate.shape:
        raise ValueError(
            "Reference and estimate must have the same length."
        )

    distortion_power = np.sum((reference - estimate) ** 2)
    reference_power = np.sum(reference ** 2)

    if distortion_power == 0:
        return float("inf")

    if reference_power == 0:
        return float("-inf")

    return float(
        10 * np.log10(reference_power / distortion_power)
    )


def best_lag_and_correlation(
    reference, estimate, sampling_rate,
    max_lag_seconds=MAX_LAG_SECONDS
):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    if len(reference) == 0 or len(estimate) == 0:
        raise ValueError("Cannot correlate empty waveforms.")

    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    if np.std(reference) == 0 or np.std(estimate) == 0:
        raise ValueError("Cannot correlate constant waveforms.")

    correlation = scipy_signal.correlate(
        estimate,
        reference,
        mode="full",
        method="fft"
    )

    lags = scipy_signal.correlation_lags(
        len(estimate),
        len(reference),
        mode="full"
    )

    max_lag_samples = int(
        round(max_lag_seconds * sampling_rate)
    )

    valid = np.abs(lags) <= max_lag_samples
    correlation = correlation[valid]
    lags = lags[valid]

    norm = np.sqrt(
        np.sum(reference ** 2) * np.sum(estimate ** 2)
    )

    if norm == 0:
        raise ValueError("Cannot normalize cross-correlation.")

    normalized = correlation / norm
    best_index = int(np.argmax(np.abs(normalized)))

    return (
        int(lags[best_index]),
        float(normalized[best_index])
    )


def align_and_score(reference, estimate, sampling_rate):
    lag_samples, max_correlation = best_lag_and_correlation(
        reference,
        estimate,
        sampling_rate
    )

    if lag_samples > 0:
        aligned_estimate = estimate[lag_samples:]
        aligned_reference = reference[:len(aligned_estimate)]

    elif lag_samples < 0:
        aligned_reference = reference[-lag_samples:]
        aligned_estimate = estimate[:len(aligned_reference)]

    else:
        aligned_reference = reference
        aligned_estimate = estimate

    n = min(len(aligned_reference), len(aligned_estimate))

    if n == 0:
        raise ValueError("No samples remain after lag alignment.")

    sdr_db = signal_distortion_ratio(
        aligned_reference[:n],
        aligned_estimate[:n]
    )

    return {
        "sdr_db": sdr_db,
        "max_correlation": max_correlation,
        "best_lag_seconds": lag_samples / sampling_rate
    }


# %% Build one comparison row per matched catalog event
def build_comparison_catalog(
    reference_trace,
    estimate_trace,
    matches,
    week_name,
    first_event_number
):
    rows = []

    for match_index, match in enumerate(matches):
        reference_row = match["reference"]
        estimate_row = match["estimate"]

        start = match["start"]
        end = match["end"]

        event_number = first_event_number + match_index

        reference_metrics = {
            "signal_rms": reference_row["signal_rms"],
            "noise_rms": reference_row["noise_rms"],
            "snr_db": reference_row["snr_db"]
        }

        estimate_metrics = {
            "signal_rms": estimate_row["signal_rms"],
            "noise_rms": estimate_row["noise_rms"],
            "snr_db": estimate_row["snr_db"]
        }

        reference_snr = reference_metrics["snr_db"]
        estimate_snr = estimate_metrics["snr_db"]

        delta_snr = (
            estimate_snr - reference_snr
            if np.isfinite(reference_snr)
            and np.isfinite(estimate_snr)
            else np.nan
        )

        try:
            (
                reference,
                estimate,
                sampling_rate,
                common_start,
                common_end
            ) = extract_common_event_waveforms(
                reference_trace,
                estimate_trace,
                start,
                end
            )

            waveform_metrics = align_and_score(
                reference,
                estimate,
                sampling_rate
            )

            output_start = common_start.isoformat()
            output_end = common_end.isoformat()
            duration = float(common_end - common_start)

        except ValueError as exc:
            print(
                f"Event {event_number}: waveform comparison unavailable "
                f"({exc})."
            )

            waveform_metrics = {
                "sdr_db": np.nan,
                "max_correlation": np.nan,
                "best_lag_seconds": np.nan
            }

            output_start = start.isoformat()
            output_end = end.isoformat()
            duration = float((end - start).total_seconds())

        rows.append({
            "event_number": event_number,
            "week": week_name,
            "reference_type": "bandpass_filtered",
            "estimate_type": "denoised_highpass_filtered",
            "start_time": output_start,
            "end_time": output_end,
            "duration_seconds": duration,
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
            "reference_event_number": reference_row["event_number"],
            "estimate_event_number": estimate_row["event_number"],
            "reference_detection_start": reference_row["start_time"].isoformat(),
            "reference_detection_end": reference_row["end_time"].isoformat(),
            "estimate_detection_start": estimate_row["start_time"].isoformat(),
            "estimate_detection_end": estimate_row["end_time"].isoformat()
        })

    return pd.DataFrame(rows, columns=COLUMNS)


# %% Plotting
def plot_histogram(values, title, x_label, output_path):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(values):
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0

        ax.hist(
            values,
            bins=30,
            alpha=0.75,
            edgecolor="black"
        )

        ax.axvspan(
            mean - std,
            mean + std,
            alpha=0.20,
            label=f"Mean ± 1 SD ({mean:.2f} ± {std:.2f})"
        )

        ax.axvline(
            mean,
            linestyle="-.",
            linewidth=2,
            label=f"Mean: {mean:.2f}"
        )

        ax.legend(fontsize=9)

    else:
        ax.text(
            0.5, 0.5, "No finite values",
            ha="center", va="center",
            transform=ax.transAxes
        )

    ax.axvline(
        0,
        linestyle="--",
        color="black",
        linewidth=1.2
    )

    ax.set(
        title=title,
        xlabel=x_label,
        ylabel="Matched events"
    )

    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_over_time(
    catalog, column, title, y_label, output_path
):
    data = catalog[["start_time", column]].copy()
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[np.isfinite(data[column])]
    data["start_time"] = pd.to_datetime(data["start_time"])
    data = data.sort_values("start_time")

    fig, ax = plt.subplots(figsize=(12, 5))

    if len(data):
        ax.scatter(
            data["start_time"],
            data[column],
            s=15,
            alpha=0.55
        )
    else:
        ax.text(
            0.5, 0.5, "No finite values",
            ha="center", va="center",
            transform=ax.transAxes
        )

    ax.set(
        title=title,
        xlabel="Event time",
        ylabel=y_label
    )

    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# %% Main
def main():
    cfg, _args = get_cfg(__doc__)

    reference_catalog_path = (
        store.stage_dir(cfg, "filter_then_denoise") / "catalog.csv"
    )

    estimate_catalog_path = (
        store.stage_dir(cfg, "denoise_then_filter") / "catalog.csv"
    )

    # Use only the requested detection types from each Stage 4 catalog.
    reference_catalog = load_catalog(
        reference_catalog_path,
        "filtered"
    )

    estimate_catalog = load_catalog(
        estimate_catalog_path,
        "denoised"
    )

    # Load the saved waveform branches.
    reference_root = (
        store.stage_dir(cfg, "filter_then_denoise") / "weekly"
    )

    estimate_root = (
        store.stage_dir(cfg, "denoise_then_filter") / "weekly"
    )

    reference_weeks = {
        path.name: path
        for path in reference_root.glob("week_*")
        if path.is_dir()
    }

    estimate_weeks = {
        path.name: path
        for path in estimate_root.glob("week_*")
        if path.is_dir()
    }

    common_weeks = sorted(
        set(reference_weeks) & set(estimate_weeks)
    )

    if not common_weeks:
        raise RuntimeError(
            "No matching weekly folders were found in both pipelines."
        )

    # Use the configured primary station for waveform comparisons.
    primary_station = cfg.data.primary_station

    print(f"Primary waveform station: {primary_station}")
    print(f"Filtered catalog events: {len(reference_catalog)}")
    print(f"Denoised catalog events: {len(estimate_catalog)}")
    print(f"Weekly batches: {len(common_weeks)}")
    print(f"Maximum lag searched: +/- {MAX_LAG_SECONDS} seconds")

    all_catalogs = []
    next_event_number = 1

    for week_name in common_weeks:
        print(f"\nProcessing {week_name}")

        reference_trace = load_primary_trace(
            reference_weeks[week_name],
            "filtered",
            primary_station
        )

        estimate_trace = load_primary_trace(
            estimate_weeks[week_name],
            "denoised",
            primary_station
        )

        # Select catalog events covered by this week's primary-station traces.
        reference_week_catalog = catalog_rows_for_week(
            reference_catalog,
            reference_trace
        )

        estimate_week_catalog = catalog_rows_for_week(
            estimate_catalog,
            estimate_trace
        )

        # Match catalog events by temporal overlap.
        matches = match_events(
            reference_week_catalog,
            estimate_week_catalog
        )

        print(
            f"Filtered events: {len(reference_week_catalog)}"
        )
        print(
            f"Denoised events: {len(estimate_week_catalog)}"
        )
        print(f"Matched events: {len(matches)}")

        if not matches:
            print("No overlapping catalog events; skipping this week.")
            continue

        week_catalog = build_comparison_catalog(
            reference_trace,
            estimate_trace,
            matches,
            week_name,
            next_event_number
        )

        all_catalogs.append(week_catalog)
        next_event_number += len(matches)

        print(
            f"Completed {len(week_catalog)} event comparisons."
        )

    if not all_catalogs:
        raise RuntimeError(
            "No overlapping catalog events were found."
        )

    catalog = pd.concat(
        all_catalogs,
        ignore_index=True
    )

    # Save Stage 6 outputs only.
    output_dir = store.comparison_dir(cfg)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = output_dir / "catalog.csv"

    catalog.to_csv(
        catalog_path,
        index=False,
        float_format=(
            "%.5g"
            if bool(cfg.output.compact_csv_floats)
            else None
        )
    )

    # Histograms.
    histograms = [
        (
            "delta_snr_db",
            "SNR difference: denoised vs bandpass filtered",
            "ΔSNR (denoised − filtered, dB)",
            "snr_delta_histogram.png"
        ),
        (
            "sdr_db",
            "SDR: denoised relative to bandpass filtered",
            "SDR (dB)",
            "sdr_histogram.png"
        ),
        (
            "max_correlation",
            "Peak CC: denoised vs bandpass filtered",
            "Peak cross-correlation (-1 to +1)",
            "cc_histogram.png"
        )
    ]

    for column, title, label, filename in histograms:
        plot_histogram(
            catalog[column],
            title,
            label,
            figure_dir / filename
        )

    # Metrics through time.
    plot_metric_over_time(
        catalog,
        "sdr_db",
        "SDR through the comparison period",
        "SDR (dB)",
        figure_dir / "sdr_over_time.png"
    )

    plot_metric_over_time(
        catalog,
        "max_correlation",
        "Peak CC through the comparison period",
        "Peak cross-correlation (-1 to +1)",
        figure_dir / "cc_over_time.png"
    )

    # Summary.
    print("\nComparison finished.")
    print(f"Matched events: {len(catalog)}")
    print(f"Saved comparison catalog: {catalog_path}")

    for metric, label, suffix in [
        ("delta_snr_db", "Median ΔSNR", " dB"),
        ("sdr_db", "Median SDR", " dB"),
        ("max_correlation", "Median peak CC", "")
    ]:
        values = catalog[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values):
            print(
                f"{label}: {np.median(values):.3f}{suffix} "
                f"({len(values)} finite events)"
            )
        else:
            print(f"{label}: no finite values")

    print(f"Figures saved in: {figure_dir}")


if __name__ == "__main__":
    main()