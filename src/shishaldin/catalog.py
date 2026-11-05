"""Build, summarise and save the long-format event catalog.

One row per DETECTION (filtered or denoised). Events found in both traces share
an event_number; events found in only one still get their own event_number.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from obspy import Trace

from .detection import Interval, intervals_overlap, overlap_flags, triggers_to_intervals
from .metrics import window_metrics

COLUMNS = ["event_number", "detection_type", "start_time", "end_time",
           "duration_seconds", "signal_rms", "noise_rms", "snr_db", "max_amplitude"]

_TIME_FMT = "%Y-%m-%dT%H:%M:%S.%f"


def _row(event_number: int, kind: str, trace: Trace, start, end, noise_seconds: float) -> dict:
    signal_rms, noise_rms, snr_db, max_amp = window_metrics(trace, start, end, noise_seconds)
    return {
        "event_number": event_number,
        "detection_type": kind,
        "start_time": start.strftime(_TIME_FMT),
        "end_time": end.strftime(_TIME_FMT),
        "duration_seconds": float(end - start),
        "signal_rms": float(signal_rms),
        "noise_rms": float(noise_rms),
        "snr_db": float(snr_db),
        "max_amplitude": float(max_amp),
    }


def build_catalog(filtered_trace: Trace, denoised_trace: Trace,
                  triggers_filtered, triggers_denoised, noise_seconds: float):
    """Returns (catalog_df, summary_dict).

    Numbering: filtered detections are numbered first, in order. A denoised
    detection overlapping a filtered one (first unmatched match) gets that SAME
    event_number. Leftover denoised detections get new event_numbers.
    """
    f_int = triggers_to_intervals(filtered_trace, triggers_filtered)
    d_int = triggers_to_intervals(denoised_trace, triggers_denoised)

    event_number = 0
    matched = set()
    rows: List[dict] = []

    for f_start, f_end in f_int:
        event_number += 1
        rows.append(_row(event_number, "filtered", filtered_trace, f_start, f_end, noise_seconds))

        for d_idx, (d_start, d_end) in enumerate(d_int):
            if d_idx in matched:
                continue
            if intervals_overlap(f_start, f_end, d_start, d_end):
                matched.add(d_idx)
                rows.append(_row(event_number, "denoised", denoised_trace, d_start, d_end, noise_seconds))
                break

    for d_idx, (d_start, d_end) in enumerate(d_int):
        if d_idx in matched:
            continue
        event_number += 1
        rows.append(_row(event_number, "denoised", denoised_trace, d_start, d_end, noise_seconds))

    df = (pd.DataFrame(rows, columns=COLUMNS)
          .sort_values(["event_number", "detection_type"])
          .reset_index(drop=True))

    summary = {
        "filtered_detections": len(f_int),
        "denoised_detections": len(d_int),
        "matched": len(matched),
        "filtered_only": len(f_int) - len(matched),
        "denoised_only": len(d_int) - len(matched),
        "rows": len(df),
    }
    return df, summary


def overlap_summary(filtered_trace: Trace, denoised_trace: Trace,
                    triggers_filtered, triggers_denoised) -> Dict[str, int]:
    """Counts using 'ANY overlap' (not one-to-one), so these can be larger than
    the `matched` count in build_catalog when one detection overlaps several."""
    f_int = triggers_to_intervals(filtered_trace, triggers_filtered)
    d_int = triggers_to_intervals(denoised_trace, triggers_denoised)
    f_flags = overlap_flags(f_int, d_int)
    d_flags = overlap_flags(d_int, f_int)
    return {
        "filtered_total": len(f_int),
        "filtered_overlapping": sum(f_flags),
        "filtered_only": len(f_flags) - sum(f_flags),
        "denoised_total": len(d_int),
        "denoised_overlapping": sum(d_flags),
        "denoised_only": len(d_flags) - sum(d_flags),
    }


def filtered_only_report(filtered_trace: Trace, denoised_trace: Trace,
                         triggers_filtered, triggers_denoised, cft_denoised) -> pd.DataFrame:
    """For each filtered detection with no denoised counterpart: how close did
    the denoised STA/LTA come to triggering in that window?"""
    f_int = triggers_to_intervals(filtered_trace, triggers_filtered)
    d_int = triggers_to_intervals(denoised_trace, triggers_denoised)
    flags = overlap_flags(f_int, d_int)

    fs = denoised_trace.stats.sampling_rate
    t0 = denoised_trace.stats.starttime
    rows = []
    for (start, end), overlaps in zip(f_int, flags):
        if overlaps:
            continue
        on = max(int((start - t0) * fs), 0)
        off = min(int((end - t0) * fs), len(cft_denoised))
        window = cft_denoised[on:off]
        rows.append({
            "start_time": start.strftime(_TIME_FMT),
            "end_time": end.strftime(_TIME_FMT),
            "duration_seconds": float(end - start),
            "peak_denoised_sta_lta": float(np.nanmax(window)) if len(window) else np.nan,
        })
    return pd.DataFrame(rows, columns=["start_time", "end_time", "duration_seconds",
                                       "peak_denoised_sta_lta"])


def format_float(x) -> str:
    """Compact CSV format used by the original notebook: 2 decimals, or 2-digit
    scientific notation when the magnitude is < 1e-3 or >= 1e3."""
    if pd.isna(x):
        return ""
    x = float(x)
    if x == 0:
        return "0.00"
    exponent = int(np.floor(np.log10(abs(x))))
    return f"{x:.2e}" if (exponent >= 3 or exponent <= -3) else f"{x:.2f}"


def compact_floats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include="number").columns.drop("event_number", errors="ignore"):
        out[col] = out[col].map(format_float)
    out["event_number"] = out["event_number"].astype(int)
    return out


def save_catalog(df: pd.DataFrame, path, compact: bool = True) -> None:
    (compact_floats(df) if compact else df).to_csv(path, index=False)
