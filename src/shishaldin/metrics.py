"""Amplitude / SNR metrics."""
from __future__ import annotations

import numpy as np
from obspy import Trace


def window_metrics(trace: Trace, window_start, window_end, noise_seconds: float):
    """RMS, max amplitude and SNR (dB) of one detection window.

    Noise = the `noise_seconds` immediately BEFORE the window (the notebook used
    the LTA length). Returns (signal_rms, noise_rms, snr_db, max_amplitude).
    """
    event = trace.copy().trim(starttime=window_start, endtime=window_end)
    signal = event.data.astype(float)
    if len(signal) == 0:
        return np.nan, np.nan, np.nan, np.nan
    signal_rms = np.sqrt(np.mean(signal ** 2))
    max_amplitude = np.max(np.abs(signal))

    noise = trace.copy().trim(starttime=window_start - noise_seconds, endtime=window_start)
    if len(noise.data) > 0:
        noise_rms = np.sqrt(np.mean(noise.data.astype(float) ** 2))
        snr = 20 * np.log10(signal_rms / noise_rms) if noise_rms > 0 else np.nan
    else:
        noise_rms, snr = np.nan, np.nan

    return signal_rms, noise_rms, snr, max_amplitude


def whole_period_snr(trace: Trace, triggers):
    """SNR of everything inside any trigger vs everything outside.

    Returns (signal_rms, noise_rms, snr_db).
    """
    data = trace.data.astype(float)
    signal_mask = np.zeros(len(data), dtype=bool)
    for on, off in triggers:
        signal_mask[on:off] = True

    signal_data = data[signal_mask]
    noise_data = data[~signal_mask]  # TODO(from notebook): better ways to estimate noise?

    if len(signal_data) == 0 or len(noise_data) == 0:
        return np.nan, np.nan, np.nan

    signal_rms = np.sqrt(np.mean(signal_data ** 2))
    noise_rms = np.sqrt(np.mean(noise_data ** 2))
    snr_db = np.nan if noise_rms == 0 else 20 * np.log10(signal_rms / noise_rms)
    return signal_rms, noise_rms, snr_db
