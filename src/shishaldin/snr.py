"""Compare SNR between the two pipelines (filter->denoise vs denoise->filter).

Inputs are the stage-3 products of each pipeline: catalog.csv and catalog_manifest.json.

Caveats: per-event SNR = RMS inside the detection window / RMS of the
`trigger.lta_seconds` just before it, within the SAME trace. The two pipelines detect
DIFFERENT event sets, so the paired comparison only uses events detected in both.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import store
from .config import Config


def pipeline_labels(cfg: Config) -> Dict[str, str]:
    bp, hp = cfg.filter.bandpass, cfg.filter.highpass
    return {
        "filter_then_denoise": f"filter → denoise (bandpass {bp.freqmin:g}–{bp.freqmax:g} Hz)",
        "denoise_then_filter": f"denoise → filter (highpass {hp.freq:g} Hz)",
    }


def load_catalogs(cfg: Config) -> Dict[str, pd.DataFrame]:
    """pipeline -> catalog DataFrame (times parsed). Raises with a hint if one is missing."""
    out = {}
    for pipe in store.PIPELINES:
        if not store.has_catalog(cfg, pipe):
            raise FileNotFoundError(
                f"No catalog for '{pipe}'. Run scripts 01/02, 03 and 04 for that pipeline "
                f"(same window) first.")
        df = pd.read_csv(store.stage_dir(cfg, pipe) / "catalog.csv",
                         parse_dates=["start_time", "end_time"])
        out[pipe] = df
    return out


def load_whole_period_snr(cfg: Config) -> Dict[str, dict]:
    """pipeline -> {"filtered": {signal_rms, noise_rms, snr_db}, "denoised": {...}}"""
    return {pipe: store.load_json(cfg, pipe, "catalog_manifest.json")["whole_period_snr"]
            for pipe in store.PIPELINES}


def summarize(catalogs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-event SNR statistics for every (pipeline, detection_type)."""
    rows = []
    for pipe, df in catalogs.items():
        for kind, g in df.groupby("detection_type"):
            snr = g["snr_db"].dropna()
            rows.append({
                "pipeline": pipe, "detection_type": kind, "n_detections": len(g),
                "n_with_snr": len(snr),
                "mean_snr_db": snr.mean(), "median_snr_db": snr.median(),
                "std_snr_db": snr.std(), "min_snr_db": snr.min(), "max_snr_db": snr.max(),
            })
    return pd.DataFrame(rows)


def match_denoised(cat_a: pd.DataFrame, cat_b: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """Pair denoised detections of A and B that overlap in time (one-to-one; each A
    event takes the first unmatched overlapping B event, as in the catalog numbering).

    Returns (paired_df, n_a_only, n_b_only). paired_df has snr_a, snr_b, diff_db (= b - a).
    """
    a = cat_a[cat_a["detection_type"] == "denoised"].sort_values("start_time").reset_index(drop=True)
    b = cat_b[cat_b["detection_type"] == "denoised"].sort_values("start_time").reset_index(drop=True)

    sa, ea = a["start_time"].to_numpy(), a["end_time"].to_numpy()
    sb, eb = b["start_time"].to_numpy(), b["end_time"].to_numpy()
    used = np.zeros(len(b), dtype=bool)

    rows = []
    for i in range(len(a)):
        cand = np.flatnonzero(~used & (sb < ea[i]) & (sa[i] < eb))
        if cand.size == 0:
            continue
        j = cand[0]
        used[j] = True
        rows.append({
            "start_time_a": a.loc[i, "start_time"], "start_time_b": b.loc[j, "start_time"],
            "snr_a": a.loc[i, "snr_db"], "snr_b": b.loc[j, "snr_db"],
        })

    paired = pd.DataFrame(rows, columns=["start_time_a", "start_time_b", "snr_a", "snr_b"])
    paired["diff_db"] = paired["snr_b"] - paired["snr_a"]
    return paired, len(a) - len(paired), len(b) - len(paired)