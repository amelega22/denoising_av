"""Inspect intermediate denoising stages without changing the processing pipeline."""
import json
import logging
from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream
from scipy.signal import welch

log = logging.getLogger(__name__)


def check_trace(trace):
    data = np.asarray(trace.data, dtype=float)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError(f"{trace.id}: empty waveform or nonfinite samples")
    if not np.isfinite(trace.stats.sampling_rate) or trace.stats.sampling_rate <= 0:
        raise ValueError(f"{trace.id}: invalid sampling rate")
    return data


def save_denoising_diagnostics(directory, stages, model, cfg):
    """Save padded, continuous segments; spectra exclude 30 s at each end.

    The preprocessed stage includes SeisBench's automatic resampling. The
    output stage precedes the final highpass, allowing each step to be isolated.
    Band powers are descriptive, not a test of whether a signal is real.
    """
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "model_rate_hz": float(model.sampling_rate),
        "pretrained": cfg.denoise.pretrained,
        "instrument_correction": cfg.data.instrument_correction,
        "versions": {p: version(p) for p in ("seisbench", "torch", "obspy", "scipy", "numpy")},
        "filters": cfg.to_dict()["filter"],
        "spectral_edge_exclusion_seconds": 30,
        "stages": {}, "warnings": [],
    }
    # All spectra cover identical physical times, without interpolating samples.
    start = max(cfg.start, max(tr.stats.starttime for tr in stages.values()) + 30)
    stop = min(cfg.stop, min(tr.stats.endtime for tr in stages.values()) - 30)
    if stop <= start:
        report["skipped"] = "No common interior interval within the requested window (padding-only segment)"
        # Avoid leaving an older spectrum with a different analysis interval.
        (directory / "stage_spectra.png").unlink(missing_ok=True)
        (directory / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
        return
    report["spectral_interval"] = [str(start), str(stop)]
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    try:
        for name, trace in stages.items():
            data = check_trace(trace)
            fs = float(trace.stats.sampling_rate)
            interior = trace.copy().trim(start, stop).data
            f, psd = welch(interior, fs=fs, nperseg=min(len(interior), round(60 * fs)))
            df = float(f[1] - f[0])
            bands = {"below_0.9": (0, .9), "0.9_to_8": (.9, 8),
                     "above_8": (8, fs / 2 + df)}
            powers = {label: float(np.sum(psd[(f >= lo) & (f < hi)]) * df)
                      for label, (lo, hi) in bands.items()}
            # Record PSD at suspected artifact frequencies and their neighborhoods.
            grid_checks = {}
            for target in (model.sampling_rate / 60, model.sampling_rate / 30,
                           model.sampling_rate / 15):
                center = np.abs(f - target) <= .1
                neighbors = (np.abs(f - target) >= .2) & (np.abs(f - target) <= .5)
                if center.any() and neighbors.any():
                    baseline = float(np.median(psd[neighbors]))
                    peak = float(np.max(psd[center]))
                    grid_checks[f"{target:.4f}_Hz"] = (
                        float(10 * np.log10(peak / baseline))
                        if peak > 0 and baseline > 0 else None)
            report["stages"][name] = {
                "sampling_rate_hz": fs, "npts": int(data.size),
                "start": str(trace.stats.starttime), "end": str(trace.stats.endtime),
                "duration_seconds": float(trace.stats.endtime - trace.stats.starttime),
                "nonfinite_samples": 0, "zero_fraction": float(np.mean(data == 0)),
                "mean": float(np.mean(data)), "rms": float(np.sqrt(np.mean(data ** 2))),
                "absolute_peak": float(np.max(np.abs(data))),
                "band_power": powers, "grid_peak_above_neighbors_db": grid_checks,
            }
            Stream([trace.copy()]).write(str(directory / f"{name}.mseed"), format="MSEED")
            ax.plot(f[1:], 10 * np.log10(np.maximum(psd[1:], np.finfo(float).tiny)), label=name)

        source, output = stages["model_preprocessed"], stages["model_output"]
        shifts = {"start_shift_seconds": float(output.stats.starttime - source.stats.starttime),
                  "end_shift_seconds": float(output.stats.endtime - source.stats.endtime)}
        report["model_timing"] = shifts
        if output.stats.sampling_rate != model.sampling_rate:
            report["warnings"].append("Model output rate differs from model sampling rate")
        if any(abs(value) > source.stats.delta * 1.5 for value in shifts.values()):
            report["warnings"].append("Model output coverage differs by more than 1.5 input samples")
        for warning in report["warnings"]:
            log.warning("%s: %s", directory.name, warning)
        ax.set(xscale="log", xlim=(.02, min(tr.stats.sampling_rate / 2 for tr in stages.values())),
               xlabel="Frequency (Hz)", ylabel="PSD (dB re 1 (m/s)²/Hz)", title=directory.name)
        for frequency in (.9, model.sampling_rate / 60, model.sampling_rate / 30,
                          model.sampling_rate / 15, 8):
            ax.axvline(frequency, color="gray", alpha=.3, linewidth=.7)
        ax.legend()
        ax.grid(True, alpha=.2)
        fig.savefig(directory / "stage_spectra.png", dpi=150)
        (directory / "checks.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        plt.close(fig)
