"""All figures. Every function takes data + cfg and RETURNS a matplotlib Figure
(nothing is shown or saved here), so they work in a notebook and in a script."""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from .config import Config
from .detection import overlap_flags, sta_lta_components, triggers_to_intervals

OVERLAP_COLOR = "green"
FILTERED_ONLY_COLOR = "orange"
DENOISED_ONLY_COLOR = "purple"
BOTH_COLOR = "green"
ONE_STATION_COLOR = "orange"


def _time_axis(ax, fmt="%H:%M"):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def _wave_axis(ax, cfg: Config, sci=(-6, -6)):
    ax.set_ylabel("Velocity (m/s)")
    ax.set_ylim(cfg.plot.waveform_ylim)
    ax.set_yticks(cfg.plot.waveform_yticks)
    ax.ticklabel_format(axis="y", style="sci", scilimits=sci)


# ------------------------------------------------------------ STA/LTA explainer
def plot_sta_lta_explainer(trace, triggers, cfg: Config, event_idx: Optional[int] = ...):
    """Walk through how STA/LTA triggers on one detection.
    event_idx: index into `triggers`; None = biggest ratio; default = cfg.plot value."""
    p, t = cfg.plot, cfg.trigger
    if event_idx is ...:
        event_idx = p.explainer_event_idx
    if len(triggers) == 0:
        raise ValueError("No network events in this window to explain.")
    if event_idx is not None and event_idx >= len(triggers):
        event_idx = len(triggers) - 1

    fs = trace.stats.sampling_rate
    sta, lta, ratio = sta_lta_components(trace, t)
    if event_idx is None:
        event_idx = int(np.argmax([ratio[on:off].max() for on, off in triggers]))
    on, off = triggers[event_idx]

    sl = slice(max(int(on - p.explainer_pad_before * fs), 0),
               min(int(off + p.explainer_pad_after * fs), len(trace.data)))
    times = trace.times("matplotlib")
    t_on, t_off = times[on], times[off]
    sec = 1 / 86400  # one second in matplotlib date units

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1, 1]})

    ax = axes[0]
    ax.plot(times[sl], trace.data[sl], color="black", linewidth=0.6)
    ax.axvspan(t_on - t.sta_seconds * sec, t_on, color="purple", alpha=0.5,
               label=f"STA window ({t.sta_seconds:g} s)")
    ax.axvspan(t_on - t.lta_seconds * sec, t_on, color="orange", alpha=0.15,
               label=f"LTA window ({t.lta_seconds:g} s)")
    ax.set_title(f"Filtered waveform, {cfg.data.primary_station}")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times[sl], sta[sl], color="tab:orange", linewidth=1.5, label="STA (short-term avg of energy)")
    ax.plot(times[sl], lta[sl], color="tab:blue", linewidth=1.5, label="LTA (long-term avg of energy)")
    ax.set_title("Running averages of squared velocity")
    ax.set_ylabel("Energy (m²/s²)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(times[sl], ratio[sl], color="black", linewidth=1.2, label="STA / LTA")
    ax.axhline(t.thr_on, color="red", linestyle="--", label=f"Trigger on ({t.thr_on:g})")
    ax.axhline(t.thr_off, color="blue", linestyle="--", label=f"Trigger off ({t.thr_off:g})")
    ax.set_title("Ratio crosses on-threshold, then falls back below the off-threshold")
    ax.set_ylabel("STA / LTA")
    ax.set_xlabel("UTC Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.axvline(t_on, color="red", linewidth=1.5)
        ax.axvline(t_off, color="blue", linewidth=1.5)
        ax.axvspan(t_on, t_off, color="green", alpha=0.12)
        ax.margins(x=0)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    axes[0].set_title(
        axes[0].get_title()
        + f"\n(green = detected network event, {(off - on) / fs:.1f} s; on/off are the network "
          f"times, so they can sit slightly off this station's own crossing)",
        fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------- raw / filtered / denoised + spectrograms
def _make_spectrogram(tr, cfg: Config):
    p = cfg.plot
    fs = tr.stats.sampling_rate
    nper = int(p.spectrogram_window_seconds * fs)
    f, t, Pspec = spectrogram(tr.data, fs=fs, window="hann", scaling="density",
                              nperseg=nper, noverlap=int(nper * p.spectrogram_overlap))
    tdnum = mdates.date2num(tr.stats.starttime.datetime) + t / 86400.0
    PdB = 10 * np.log10(np.abs(Pspec) / np.power(p.spectrogram_ref, 2))
    return f, tdnum, PdB

def _shared_clim(specs, cfg: Config):
    """One (vmin, vmax) for all spectrograms, from the pooled values inside the plotted band."""
    lo, hi = cfg.plot.spectrogram_ylim
    pooled = np.concatenate([PdB[(f >= lo) & (f <= hi), :].ravel() for f, _, PdB in specs])
    pooled = pooled[np.isfinite(pooled)]
    return np.percentile(pooled, 35), np.percentile(pooled, 97.5)


def plot_waveforms_spectrograms(raw, filtered, denoised, cfg: Config,
                                labels=("Raw Data", "Filtered Data", "Denoised Data")):
    p = cfg.plot
    traces = (raw, filtered, denoised)
    specs = [_make_spectrogram(tr, cfg) for tr in traces]

    # optional manual override: plot.spectrogram_clim: [vmin, vmax] in config.yaml
    clim = getattr(p, "spectrogram_clim", None) or _shared_clim(specs, cfg)

    fig, axes = plt.subplots(3, 2, figsize=(18, 9), sharex=True, constrained_layout=True)

    for r, (title, tr, (f, tdnum, PdB)) in enumerate(zip(labels, traces, specs)):
        ax = axes[r, 0]
        ax.plot(tr.times("matplotlib"), tr.data, linewidth=0.6, color="black")
        ax.set_title(title, fontsize=16)
        _wave_axis(ax, cfg)
        ax.grid(True, alpha=0.3)

        ax = axes[r, 1]
        im = ax.imshow(PdB, extent=[tdnum[0], tdnum[-1], f[0], f[-1]], origin="lower",
                       aspect="auto", interpolation="none", cmap=p.colormap,
                       vmin=clim[0], vmax=clim[1])
        ax.set_ylim(p.spectrogram_ylim)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{title} spectrogram", fontsize=16)

    # one colorbar spanning all three spectrogram panels
    fig.colorbar(im, ax=axes[:, 1], label="Power (dB)", pad=0.01)

    for ax in axes.flat:
        ax.set_xlim(cfg.start.matplotlib_date, cfg.stop.matplotlib_date)
        ax.xaxis_date()
        _time_axis(ax)
    axes[2, 0].set_xlabel("UTC Time")
    axes[2, 1].set_xlabel("UTC Time")
    return fig


# --------------------------------------------- filtered vs denoised with triggers
def _shade_events(ax, intervals, flags, only_label, only_color):
    seen_overlap = seen_only = False
    for (start, end), overlaps in zip(intervals, flags):
        label = None
        if overlaps and not seen_overlap:
            label, seen_overlap = "Overlaps both", True
        elif not overlaps and not seen_only:
            label, seen_only = only_label, True
        ax.axvspan(start.matplotlib_date, end.matplotlib_date, alpha=0.3, label=label,
                   color=OVERLAP_COLOR if overlaps else only_color)


def plot_triggers(filtered, denoised, cft_filtered, cft_denoised,
                  triggers_filtered, triggers_denoised, cfg: Config, mode: str = "overlap"):
    """Filtered vs denoised waveform + STA/LTA with detections.

    mode="lines"   : red/blue vertical lines at trigger on/off
    mode="overlap" : shaded spans, green if the other trace also detected it
    """
    if mode not in ("lines", "overlap"):
        raise ValueError("mode must be 'lines' or 'overlap'")
    t = cfg.trigger
    t_f, t_d = filtered.times("matplotlib"), denoised.times("matplotlib")
    f_int = triggers_to_intervals(filtered, triggers_filtered)
    d_int = triggers_to_intervals(denoised, triggers_denoised)
    f_flags, d_flags = overlap_flags(f_int, d_int), overlap_flags(d_int, f_int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col")

    panels = [
        # (col, name, times, wave, cft, trigger idx, intervals, flags, only-label, only-color, sci)
        (0, "Filtered", t_f, filtered.data, cft_filtered, triggers_filtered, f_int, f_flags,
         "Filtered only", FILTERED_ONLY_COLOR, (-6, -6)),
        (1, "Denoised", t_d, denoised.data, cft_denoised, triggers_denoised, d_int, d_flags,
         "Denoised only", DENOISED_ONLY_COLOR, (-6, 6)),
    ]
    for col, name, times, wave, cft, trig_idx, ints, flags, only_label, only_color, sci in panels:
        ax = axes[0, col]
        ax.plot(times, wave, linewidth=0.8, color="black")
        ax.set_title(name, fontsize=18)
        _wave_axis(ax, cfg, sci=sci)

        ax2 = axes[1, col]
        ax2.plot(times, cft, linewidth=0.8, color="black")
        ax2.axhline(t.thr_on, color="red", linestyle="--", linewidth=1.5, label="Trigger On")
        ax2.axhline(t.thr_off, color="blue", linestyle="--", linewidth=1.5, label="Trigger Off")
        ax2.set_ylabel("STA/LTA")
        ax2.set_xlabel("UTC Time")
        ax2.set_title(f"{name} STA/LTA")

        if mode == "lines":
            for i, (on, off) in enumerate(trig_idx):
                ax.axvline(times[on], color="red", linewidth=2, alpha=0.9,
                           label="Trigger On" if i == 0 else None)
                ax.axvline(times[off], color="blue", linewidth=2, alpha=0.9,
                           label="Trigger Off" if i == 0 else None)
        else:
            _shade_events(ax, ints, flags, only_label, only_color)
            _shade_events(ax2, ints, flags, only_label, only_color)

        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.margins(x=0)
        _time_axis(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ------------------------------ independent station triggers vs network coincidence
def _shade_station_triggers(ax, intervals, flags):
    seen_both = seen_one = False
    for (start, end), coincident in zip(intervals, flags):
        if coincident:
            color, label, seen_both = BOTH_COLOR, (None if seen_both else "Part of network coincidence"), True
        else:
            color, label, seen_one = ONE_STATION_COLOR, (None if seen_one else "Not part of coincidence"), True
        ax.axvspan(start.matplotlib_date, end.matplotlib_date, color=color, alpha=0.3, label=label)


def plot_station_triggers(filtered_traces: Dict, denoised_traces: Dict,
                          filtered_results: Dict, denoised_results: Dict, cfg: Config):
    stations, t = cfg.data.stations, cfg.trigger
    columns = [("Filtered", filtered_traces, filtered_results),
               ("Denoised", denoised_traces, denoised_results)]

    fig, axes = plt.subplots(2 * len(stations), 2, figsize=(15, 4 * len(stations)),
                             sharex="col", squeeze=False)

    for s_idx, stn in enumerate(stations):
        for col, (name, traces, results) in enumerate(columns):
            tr = traces[stn]
            times = tr.times("matplotlib")
            res = results[stn]

            ax = axes[2 * s_idx, col]
            ax.plot(times, tr.data, color="black", linewidth=0.8)
            _shade_station_triggers(ax, res["intervals"], res["coincident_flags"])
            ax.set_title(f"{name} waveform — {stn}")
            _wave_axis(ax, cfg)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right", fontsize=8)

            ax = axes[2 * s_idx + 1, col]
            ax.plot(times, res["cft"], color="black", linewidth=0.8)
            ax.axhline(t.thr_on, color="red", linestyle="--", linewidth=1.2, label="Trigger on")
            ax.axhline(t.thr_off, color="blue", linestyle="--", linewidth=1.2, label="Trigger off")
            _shade_station_triggers(ax, res["intervals"], res["coincident_flags"])
            ax.set_title(f"{name} STA/LTA — {stn}")
            ax.set_ylabel("STA/LTA")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

    plot_start = cfg.start + t.startup_seconds
    for ax in axes.flat:
        ax.margins(x=0)
        ax.set_xlim(plot_start.matplotlib_date, cfg.stop.matplotlib_date)
        _time_axis(ax)
    axes[-1, 0].set_xlabel("UTC Time")
    axes[-1, 1].set_xlabel("UTC Time")
    fig.suptitle("Independent station STA/LTA with network coincidence classification "
                 f"({cfg.coincidence_sum} of {len(stations)} stations required)", fontsize=14)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- durations
def plot_duration_hist(catalog, bins: int = 30):
    fig, ax = plt.subplots()
    catalog["duration_seconds"].hist(bins=bins, ax=ax)
    ax.set_xlabel("Event duration (s)")
    ax.set_title("Detected event duration distribution")
    return fig

# -------------------------------------------------- SNR comparison between pipelines
_COLOR_A, _COLOR_B = "tab:blue", "tab:red"


def plot_snr_distributions(catalogs: Dict, labels: Dict, bins: int = 30):
    """Left: overlaid histograms of per-event SNR for the two DENOISED sets.
    Right: boxplots for filtered and denoised detections of each pipeline."""
    pipes = list(labels)
    colors = dict(zip(pipes, (_COLOR_A, _COLOR_B)))

    def vals(pipe, kind):
        df = catalogs[pipe]
        return df.loc[df["detection_type"] == kind, "snr_db"].dropna().to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})

    allv = np.concatenate([vals(p, "denoised") for p in pipes])
    edges = np.linspace(allv.min(), allv.max(), bins + 1) if allv.size else bins
    for pipe in pipes:
        v = vals(pipe, "denoised")
        ax1.hist(v, bins=edges, alpha=0.5, color=colors[pipe],
                 label=f"{labels[pipe]}  (n={len(v)}, median {np.median(v):.1f} dB)" if len(v) else labels[pipe])
        if len(v):
            ax1.axvline(np.median(v), color=colors[pipe], linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Per-event SNR (dB)")
    ax1.set_ylabel("Number of denoised detections")
    ax1.set_title("Denoised detections: SNR distribution")
    ax1.margins(y=0.3)   # headroom so the legend doesn't cover the bars
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    groups, ticks, box_colors = [], [], []
    for pipe in pipes:
        for kind in ("filtered", "denoised"):
            groups.append(vals(pipe, kind))
            ticks.append(f"{kind}\n({'F→D' if pipe == 'filter_then_denoise' else 'D→F'})")
            box_colors.append(colors[pipe])
    bp = ax2.boxplot([g if len(g) else [np.nan] for g in groups], patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax2.set_xticks(range(1, len(groups) + 1))
    ax2.set_xticklabels(ticks)
    ax2.set_ylabel("Per-event SNR (dB)")
    ax2.set_title("Per-event SNR by pipeline and trace type\n(F→D = filter then denoise, D→F = denoise then filter)")
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_whole_period_snr(whole: Dict, labels: Dict):
    """Grouped bars: whole-period SNR (all detection windows vs everything else)."""
    pipes = list(labels)
    x = np.arange(len(pipes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k, (kind, color) in enumerate((("filtered", "tab:gray"), ("denoised", "tab:purple"))):
        h = [whole[p][kind]["snr_db"] for p in pipes]
        bars = ax.bar(x + (k - 0.5) * w, h, w, label=kind.capitalize(), color=color, alpha=0.75)
        for b, v in zip(bars, h):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[p].replace(" (", "\n(") for p in pipes], fontsize=9)
    ax.set_ylabel("Whole-period SNR (dB)")
    ax.set_title("Whole-period SNR (detection windows vs. the rest of the trace)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_snr_paired(paired, label_a: str, label_b: str, n_a_only: int = 0, n_b_only: int = 0):
    """Events detected (denoised) in BOTH pipelines: SNR in A vs B, and the B - A difference."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    d = paired.dropna(subset=["snr_a", "snr_b"])

    if len(d):
        lo = float(min(d["snr_a"].min(), d["snr_b"].min()))
        hi = float(max(d["snr_a"].max(), d["snr_b"].max()))
        ax1.scatter(d["snr_a"], d["snr_b"], s=18, alpha=0.6, color="tab:purple")
        ax1.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label="1:1")
        ax1.legend()
        ax2.hist(d["diff_db"], bins=30, color="tab:purple", alpha=0.7)
        med = float(d["diff_db"].median())
        ax2.axvline(0, color="black", linewidth=1)
        ax2.axvline(med, color="red", linestyle="--", label=f"median {med:+.2f} dB")
        ax2.legend()
    ax1.set_xlabel(f"SNR (dB): {label_a}")
    ax1.set_ylabel(f"SNR (dB): {label_b}")
    ax1.set_title(f"Matched denoised events (n={len(d)}; {n_a_only} only in first, {n_b_only} only in second)")
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("SNR difference (dB):  second − first  (positive = second pipeline higher)")
    ax2.set_ylabel("Number of matched events")
    ax2.set_title("Per-event SNR difference")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
