
"""Calculate waveform amplitude, SNR, SDR, and cross-correlation metrics."""

from __future__ import annotations

import numpy as np
from obspy import Trace
from scipy.signal import correlate, correlation_lags


def window_metrics(trace: Trace, window_start, window_end, noise_seconds: float):
    data = np.asarray(trace.data, dtype=float)
    start_time = trace.stats.starttime
    sampling_rate = float(trace.stats.sampling_rate)
    n_samples = len(data)

    event_start = int(round(float(window_start - start_time) * sampling_rate))
    event_end = int(round(float(window_end - start_time) * sampling_rate))
    event_start = max(0, event_start)
    event_end = min(n_samples, event_end + 1)

    if event_end <= event_start:
        return np.nan, np.nan, np.nan, np.nan

    signal = data[event_start:event_end]
    signal = signal[np.isfinite(signal)]

    if len(signal) == 0:
        return np.nan, np.nan, np.nan, np.nan

    signal_rms = float(np.sqrt(np.mean(signal ** 2)))
    max_amplitude = float(np.max(np.abs(signal)))

    noise_start_time = window_start - noise_seconds
    noise_start = int(round(float(noise_start_time - start_time) * sampling_rate))
    noise_start = max(0, noise_start)

    # Exclude the event's first sample from the noise window.
    noise = data[noise_start:event_start]
    noise = noise[np.isfinite(noise)]

    if len(noise):
        noise_rms = float(np.sqrt(np.mean(noise ** 2)))
        snr = (
            float(20 * np.log10(signal_rms / noise_rms))
            if noise_rms > 0 and signal_rms > 0 else np.nan
        )
    else:
        noise_rms, snr = np.nan, np.nan

    return signal_rms, noise_rms, snr, max_amplitude


def whole_period_snr(trace: Trace, triggers):
    data = np.asarray(trace.data, dtype=float)
    signal_mask = np.zeros(len(data), dtype=bool)

    for on, off in triggers:
        on = max(0, min(int(round(float(on))), len(data)))
        off = max(0, min(int(round(float(off))), len(data)))
        if off > on:
            signal_mask[on:off] = True

    signal_data = data[signal_mask]
    noise_data = data[~signal_mask]
    signal_data = signal_data[np.isfinite(signal_data)]
    noise_data = noise_data[np.isfinite(noise_data)]

    if len(signal_data) == 0 or len(noise_data) == 0:
        return np.nan, np.nan, np.nan

    signal_rms = float(np.sqrt(np.mean(signal_data ** 2)))
    noise_rms = float(np.sqrt(np.mean(noise_data ** 2)))
    snr_db = (
        float(20 * np.log10(signal_rms / noise_rms))
        if noise_rms > 0 and signal_rms > 0 else np.nan
    )

    return signal_rms, noise_rms, snr_db


def waveform_comparison_metrics(
    filtered_trace: Trace,
    denoised_bandpass_trace: Trace,
    window_start,
    window_end,
    max_lag_seconds: float = 1.0
):
    missing = {
        "sdr_db": np.nan,
        "max_correlation": np.nan,
        "cc_lag_seconds": np.nan
    }

    if filtered_trace is None or denoised_bandpass_trace is None:
        return missing.copy()

    if (filtered_trace.stats.starttime > window_start or
            filtered_trace.stats.endtime < window_end or
            denoised_bandpass_trace.stats.starttime > window_start or
            denoised_bandpass_trace.stats.endtime < window_end):
        return missing.copy()

    filtered = filtered_trace.copy().trim(
        starttime=window_start, endtime=window_end,
        nearest_sample=True, pad=False
    )
    denoised = denoised_bandpass_trace.copy().trim(
        starttime=window_start, endtime=window_end,
        nearest_sample=True, pad=False
    )

    # Resample the denoised snippet to the filtered trace's sample rate.
    target_rate = float(filtered.stats.sampling_rate)
    denoised_rate = float(denoised.stats.sampling_rate)

    if not np.isclose(target_rate, denoised_rate):
        denoised.resample(target_rate)

    delta = float(filtered.stats.delta)
    offset_float = float(
        denoised.stats.starttime - filtered.stats.starttime
    ) / delta
    offset = int(round(offset_float))

    if not np.isclose(offset_float, offset, rtol=0, atol=1e-3):
        return missing.copy()

    reference = np.asarray(filtered.data, dtype=float)
    estimate = np.asarray(denoised.data, dtype=float)

    # Align the snippets to the same starting sample.
    if offset > 0:
        reference = reference[offset:]
    elif offset < 0:
        estimate = estimate[-offset:]

    n_samples = min(len(reference), len(estimate))
    if n_samples < 3:
        return missing.copy()

    reference = reference[:n_samples]
    estimate = estimate[:n_samples]

    valid = np.isfinite(reference) & np.isfinite(estimate)
    reference = reference[valid]
    estimate = estimate[valid]

    if len(reference) < 3:
        return missing.copy()

    # Remove the mean before comparing waveform shapes.
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    reference_energy = float(np.sum(reference ** 2))
    estimate_energy = float(np.sum(estimate ** 2))
    distortion_energy = float(np.sum((reference - estimate) ** 2))

    if reference_energy > 0:
        sdr_db = (
            float(10 * np.log10(reference_energy / distortion_energy))
            if distortion_energy > 0 else np.inf
        )
    else:
        sdr_db = np.nan

    # Calculate normalized cross-correlation within the allowed lag.
    denominator = np.sqrt(reference_energy * estimate_energy)

    if denominator > 0:
        correlations = correlate(
            estimate, reference, mode="full", method="fft"
        )
        lags = correlation_lags(
            len(estimate), len(reference), mode="full"
        )
        max_lag_samples = int(round(max_lag_seconds / delta))
        allowed = np.abs(lags) <= max_lag_samples

        if np.any(allowed):
            normalized = correlations[allowed] / denominator
            allowed_lags = lags[allowed]
            peak_index = int(np.argmax(np.abs(normalized)))
            max_correlation = float(normalized[peak_index])
            cc_lag_seconds = float(allowed_lags[peak_index] * delta)
        else:
            max_correlation, cc_lag_seconds = np.nan, np.nan
    else:
        max_correlation, cc_lag_seconds = np.nan, np.nan

    return {
        "sdr_db": sdr_db,
        "max_correlation": max_correlation,
        "cc_lag_seconds": cc_lag_seconds
    }
