
# %% Imports
import numpy as np
import pandas as pd

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

    catalog["start_time"] = pd.to_datetime(
        catalog["start_time"], errors="coerce"
    )
    catalog["end_time"] = pd.to_datetime(
        catalog["end_time"], errors="coerce"
    )

    catalog = catalog.dropna(
        subset=["start_time", "end_time"]
    ).copy()

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


# %% Find corresponding filtered catalog metrics
def find_filtered_event(denoised_row, filtered_catalog):
    candidates = filtered_catalog[
        (filtered_catalog["start_time"] < denoised_row["end_time"])
        & (filtered_catalog["end_time"] > denoised_row["start_time"])
    ]

    if candidates.empty:
        return None

    overlap_start = candidates["start_time"].clip(
        lower=denoised_row["start_time"]
    )
    overlap_end = candidates["end_time"].clip(
        upper=denoised_row["end_time"]
    )
    overlaps = overlap_end - overlap_start

    return candidates.loc[overlaps.idxmax()]


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


# %% SDR
def signal_distortion_ratio(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    valid = np.isfinite(reference) & np.isfinite(estimate)
    reference = reference[valid]
    estimate = estimate[valid]

    if len(reference) == 0:
        return np.nan

    distortion_power = np.sum((reference - estimate) ** 2)
    reference_power = np.sum(reference ** 2)

    if reference_power == 0:
        return np.nan

    if distortion_power == 0:
        return float("inf")

    return float(
        10 * np.log10(reference_power / distortion_power)
    )


# %% Cross-correlation
def best_lag_and_correlation(
    reference, estimate, sampling_rate,
    max_lag_seconds=MAX_LAG_SECONDS
):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    valid = np.isfinite(reference) & np.isfinite(estimate)
    reference = reference[valid]
    estimate = estimate[valid]

    if len(reference) == 0:
        return np.nan, np.nan

    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    # A constant waveform has no defined normalized correlation.
    if np.std(reference) == 0 or np.std(estimate) == 0:
        return np.nan, np.nan

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

    valid_lags = np.abs(lags) <= max_lag_samples
    correlation = correlation[valid_lags]
    lags = lags[valid_lags]

    norm = np.sqrt(
        np.sum(reference ** 2) * np.sum(estimate ** 2)
    )

    if norm == 0:
        return np.nan, np.nan

    normalized = correlation / norm
    best_index = int(np.argmax(np.abs(normalized)))

    return (
        int(lags[best_index]),
        float(normalized[best_index])
    )


def align_and_score(reference, estimate, sampling_rate):
    # SDR is calculated even if cross-correlation is undefined.
    lag_samples, max_correlation = best_lag_and_correlation(
        reference,
        estimate,
        sampling_rate
    )

    if not np.isfinite(lag_samples):
        aligned_reference = reference
        aligned_estimate = estimate
        lag_seconds = np.nan

    elif lag_samples > 0:
        aligned_estimate = estimate[lag_samples:]
        aligned_reference = reference[:len(aligned_estimate)]
        lag_seconds = lag_samples / sampling_rate

    elif lag_samples < 0:
        aligned_reference = reference[-lag_samples:]
        aligned_estimate = estimate[:len(aligned_reference)]
        lag_seconds = lag_samples / sampling_rate

    else:
        aligned_reference = reference
        aligned_estimate = estimate
        lag_seconds = 0.0

    n = min(len(aligned_reference), len(aligned_estimate))

    if n == 0:
        sdr_db = np.nan
    else:
        sdr_db = signal_distortion_ratio(
            aligned_reference[:n],
            aligned_estimate[:n]
        )

    return {
        "sdr_db": sdr_db,
        "max_correlation": max_correlation,
        "best_lag_seconds": lag_seconds
    }


# %% Build one comparison row per denoised event
def build_comparison_row(
    reference_trace,
    estimate_trace,
    denoised_row,
    filtered_row,
    week_name
):
    # The denoised catalog defines the event window.
    start = denoised_row["start_time"]
    end = denoised_row["end_time"]

    if filtered_row is not None:
        reference_signal_rms = filtered_row["signal_rms"]
        reference_noise_rms = filtered_row["noise_rms"]
        reference_snr = filtered_row["snr_db"]
        reference_event_number = filtered_row["event_number"]
        reference_start = filtered_row["start_time"].isoformat()
        reference_end = filtered_row["end_time"].isoformat()
    else:
        reference_signal_rms = np.nan
        reference_noise_rms = np.nan
        reference_snr = np.nan
        reference_event_number = np.nan
        reference_start = ""
        reference_end = ""

    estimate_signal_rms = denoised_row["signal_rms"]
    estimate_noise_rms = denoised_row["noise_rms"]
    estimate_snr = denoised_row["snr_db"]

    delta_snr = (
        estimate_snr - reference_snr
        if np.isfinite(estimate_snr)
        and np.isfinite(reference_snr)
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

    except (ValueError, IndexError) as exc:
        print(
            f"Event {denoised_row['event_number']}: "
            f"waveform comparison unavailable ({exc})."
        )

        waveform_metrics = {
            "sdr_db": np.nan,
            "max_correlation": np.nan,
            "best_lag_seconds": np.nan
        }

        output_start = start.isoformat()
        output_end = end.isoformat()
        duration = float((end - start).total_seconds())

    return {
        "event_number": denoised_row["event_number"],
        "week": week_name,
        "reference_type": "bandpass_filtered",
        "estimate_type": "denoised_highpass_filtered",
        "start_time": output_start,
        "end_time": output_end,
        "duration_seconds": duration,
        "reference_signal_rms": reference_signal_rms,
        "reference_noise_rms": reference_noise_rms,
        "reference_snr_db": reference_snr,
        "estimate_signal_rms": estimate_signal_rms,
        "estimate_noise_rms": estimate_noise_rms,
        "estimate_snr_db": estimate_snr,
        "delta_snr_db": delta_snr,
        "sdr_db": waveform_metrics["sdr_db"],
        "max_correlation": waveform_metrics["max_correlation"],
        "best_lag_seconds": waveform_metrics["best_lag_seconds"],
        "reference_event_number": reference_event_number,
        "estimate_event_number": denoised_row["event_number"],
        "reference_detection_start": reference_start,
        "reference_detection_end": reference_end,
        "estimate_detection_start": start.isoformat(),
        "estimate_detection_end": end.isoformat()
    }


# %% Main
def main():
    cfg, _args = get_cfg(__doc__)

    # Denoised catalog defines the events and their times.
    estimate_catalog = load_catalog(
        store.stage_dir(cfg, "denoise_then_filter") / "catalog.csv",
        "denoised"
    )

    # Filtered catalog is used only for existing SNR/RMS values.
    reference_catalog = load_catalog(
        store.stage_dir(cfg, "filter_then_denoise") / "catalog.csv",
        "filtered"
    )

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

    primary_station = cfg.data.primary_station

    print(f"Primary station: {primary_station}")
    print(f"Denoised events to analyse: {len(estimate_catalog)}")
    print(f"Filtered events available for metric lookup: {len(reference_catalog)}")
    print(f"Weekly batches: {len(common_weeks)}")

    all_rows = []
    processed_events = set()

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

        week_events = catalog_rows_for_week(
            estimate_catalog,
            estimate_trace
        )

        print(f"Denoised events in this week's trace: {len(week_events)}")

        for _, denoised_row in week_events.iterrows():
            event_number = denoised_row["event_number"]

            if event_number in processed_events:
                continue

            processed_events.add(event_number)

            filtered_row = find_filtered_event(
                denoised_row,
                reference_catalog
            )

            row = build_comparison_row(
                reference_trace,
                estimate_trace,
                denoised_row,
                filtered_row,
                week_name
            )

            all_rows.append(row)

        print(f"Completed {len(week_events)} events in {week_name}.")

    if not all_rows:
        raise RuntimeError(
            "No denoised catalog events overlapped the primary-station "
            "waveform coverage."
        )

    catalog = pd.DataFrame(all_rows, columns=COLUMNS)
    catalog = catalog.sort_values("start_time").reset_index(drop=True)

    output_dir = store.comparison_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = output_dir / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    print("\nComparison catalog complete.")
    print(f"Denoised events processed: {len(catalog)}")
    print(f"Events with finite SDR: {np.isfinite(catalog['sdr_db']).sum()}")
    print(
        "Events with finite CC: "
        f"{np.isfinite(catalog['max_correlation']).sum()}"
    )
    print(
        "Events without filtered catalog match: "
        f"{catalog['reference_event_number'].isna().sum()}"
    )
    print(f"Saved comparison catalog: {catalog_path}")


if __name__ == "__main__":
    main()
