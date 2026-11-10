"""Compare the two seismic processing methodologies.

Traditional method:
    filter_then_denoise -> filtered (bandpass-filtered waveform)

ML method:
    denoise_then_filter -> denoised (denoised + highpass-filtered waveform)

Compares the two final outputs using matched events.

Outputs:
    comparison/method_comparison.csv
    comparison/figures/
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fractions import Fraction
from scipy.signal import correlate, correlation_lags, resample_poly
from _common import get_cfg
from shishaldin import plotting as P, store, snr

def calculate_sdr_cc(reference, estimate, sampling_rate, max_lag_seconds=2.0):
    """Calculate SDR and peak CC after aligning estimate to reference."""

    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    n = min(len(reference), len(estimate))
    reference = reference[:n]
    estimate = estimate[:n]

    if n < 2:
        raise ValueError("Not enough samples.")

    ref_centered = reference - np.mean(reference)
    est_centered = estimate - np.mean(estimate)

    corr = correlate(est_centered, ref_centered, mode="full", method="fft")
    lags = correlation_lags(len(est_centered), len(ref_centered), mode="full")

    valid = np.abs(lags) <= int(max_lag_seconds * sampling_rate)
    corr = corr[valid]
    lags = lags[valid]

    normalized = []
    for lag, value in zip(lags, corr):
        if lag > 0:
            r, e = ref_centered[:-lag], est_centered[lag:]
        elif lag < 0:
            r, e = ref_centered[-lag:], est_centered[:lag]
        else:
            r, e = ref_centered, est_centered

        denom = np.sqrt(np.sum(r**2) * np.sum(e**2))
        normalized.append(value / denom if denom > 0 else np.nan)

    normalized = np.asarray(normalized)
    if not np.isfinite(normalized).any():
        raise ValueError("Could not calculate cross-correlation.")

    best = int(np.nanargmax(np.abs(normalized)))
    lag = int(lags[best])
    cc = float(normalized[best])

    if lag > 0:
        reference = reference[:-lag]
        estimate = estimate[lag:]
    elif lag < 0:
        reference = reference[-lag:]
        estimate = estimate[:lag]

    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]

    distortion = np.sum((reference - estimate) ** 2)
    power = np.sum(reference ** 2)

    if distortion == 0:
        sdr = np.inf
    elif power == 0:
        sdr = -np.inf
    else:
        sdr = 10 * np.log10(power / distortion)

    return float(sdr), cc


def get_event_data(trace, start, end, target_rate):
    """Extract event waveform and resample to the reference sampling rate."""

    event = trace.copy()
    event.trim(starttime=start, endtime=end, pad=False)

    if len(event.data) == 0:
        raise ValueError("Empty event window.")

    if not np.isclose(event.stats.sampling_rate, target_rate):
        ratio = Fraction(
            target_rate / event.stats.sampling_rate
        ).limit_denominator(10000)

        event.data = resample_poly(
            event.data.astype(float),
            ratio.numerator,
            ratio.denominator,
        )

    return np.asarray(event.data, dtype=float)


def calculate_matched_sdr_cc(paired, bandpass_traces, ml_traces):
    """Calculate SDR and CC for the same events matched by match_denoised()."""

    rows = []

    stations = sorted(set(bandpass_traces) & set(ml_traces))

    for event_number, row in paired.iterrows():
        # Use the bandpass catalog's event start time.
        start = row["start_time_a"]

        # Find the corresponding event end time from the bandpass catalog.
        # The catalog event is identified by its start time.
        for station in stations:
            reference_trace = bandpass_traces[station]
            estimate_trace = ml_traces[station]

            # Use the matched event start/end from the corresponding catalogs.
            # The caller supplies the event intervals below.
            rows.append({
                "event_number": event_number + 1,
                "station": station,
            })

    return rows

def main():
    cfg, _ = get_cfg(__doc__)

    traditional, ml = store.PIPELINES

    # Load saved catalogs from both processing pipelines.
    catalogs = snr.load_catalogs(cfg)

    # Select the final output from each methodology.
    traditional_catalog = catalogs[traditional]
    ml_catalog = catalogs[ml]

    # Match events detected in both pipelines.
    paired, traditional_only, ml_only = snr.match_denoised(
        traditional_catalog,
        ml_catalog,
    )

    if paired.empty:
        raise RuntimeError("No matched events were found.")

    # Keep only the matched events and their SNR measurements.
    comparison = paired.copy()

    # Calculate the difference: ML method minus traditional method.
    comparison["snr_difference_db"] = (
        comparison["snr_b_db"] - comparison["snr_a_db"]
    )

    # Save comparison table.
    out = store.comparison_dir(cfg)
    comparison.to_csv(out / "method_comparison.csv", index=False)

    # Print results.
    differences = comparison["snr_difference_db"].dropna()

    print("\nPROCESSING METHOD COMPARISON")
    print(f"Matched events: {len(comparison)}")
    print(f"Only traditional: {traditional_only}")
    print(f"Only ML: {ml_only}")

    if len(differences):
        print("\nSNR difference (ML minus traditional):")
        print(f"  Median: {differences.median():+.2f} dB")
        print(f"  Mean:   {differences.mean():+.2f} dB")
        print(
            f"  ML method higher for: "
            f"{(differences > 0).mean() * 100:.1f}% of events"
        )

    # Plot paired SNR comparison.
    fig = P.plot_snr_paired(
        paired,
        "Traditional bandpass",
        "ML denoising + highpass",
        traditional_only,
        ml_only,
    )

    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        figure_dir / "snr_method_comparison.png",
        dpi=150,
    )
    plt.close(fig)

    print(f"\nSaved comparison to {out}")


if __name__ == "__main__":
    main()