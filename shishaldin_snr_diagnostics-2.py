# %% Imports
"""Two diagnostics for judging whether low-SNR denoised detections are
real signals or noise the denoiser hallucinated/passed through:

1. Visual comparison: pull ~10 of the lowest-SNR matched events and ~10 of
   the highest-SNR ones, and plot their raw/filtered/denoised waveforms
   side by side. High-SNR events are your "known good" reference for what
   a real signal looks like; compare the low-SNR ones against that by eye.

2. Quantitative comparison: for each event, cross-correlate the filtered
   and denoised waveforms (best-lag alignment + peak correlation
   coefficient) and compute the Signal-to-Distortion Ratio (SDR, in dB)
   between them after alignment. A real signal that survived denoising
   coherently should stay well-correlated with the filtered trace even at
   low SNR; a detection that's mostly noise is more likely to decorrelate,
   since the denoiser has less real signal structure to preserve. SDR here
   treats the filtered trace as the reference and the denoised trace as
   the estimate -- it measures how much the denoiser changed the waveform
   shape, not how "correct" either one is against some absolute truth
   (there isn't one for field data).

Deliberately self-contained (no import from shishaldin_negative_snr_waveforms.py)
so it doesn't inherit that file's load_waveforms import fragility -- only
needs matched_snr_differences from shishaldin_snr_comparison.py, which in
turn only needs load_catalogs (not load_waveforms) from
shishaldin_catalog_explorer.py.

This file's select_random_negative_events()/visualize_negative_events()
cover the same random-negative-event sampling that
shishaldin_negative_snr_waveforms.py did, so that file is no longer
needed -- everything here supersedes it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import UTCDateTime
from scipy import signal as scipy_signal

from shishaldin_snr_comparison import matched_snr_differences

# %% Editable settings
SNR_COLUMN = 'denoised_snr_db'   # which per-event SNR to rank on; 'filtered_snr_db' also available
N_SAMPLES = 10                   # how many low- and high-SNR events to pull
EVENT_PADDING_SECONDS = 30       # extra context shown on either side of each event, for the waveform plots
METRICS_PADDING_SECONDS = 0      # padding used for the SDR/cross-correlation window (0 = tight detection window)
MAX_LAG_SECONDS = 0.5            # how far to search for the best alignment lag between filtered and denoised
WAVEFORM_YLIM = (-10e-6, 10e-6)
WAVEFORM_YTICKS = [-5e-6, 0, 5e-6]
WAVEFORM_UNITS = 'Velocity (m/s)'
TIME_LABEL = 'Catalog time (timezone unspecified)'
EXPORT_DIR = Path('/Users/amelega/Desktop/figures')
EXPORT_DPI = 300


# %% Event window + waveform plot (self-contained copies; see module docstring)
def event_window(catalog, event_number, padding_seconds=EVENT_PADDING_SECONDS):
    """(start, end) covering all of one event's detection rows, padded."""
    rows = catalog[catalog.event_number == event_number]
    if rows.empty:
        raise ValueError(f'event_number {event_number} not found in this catalog sheet')
    pad = pd.Timedelta(seconds=padding_seconds)
    return rows.start_time.min() - pad, rows.end_time.max() + pad


def plot_event_waveforms(event_number, label, window_start, window_end, waveforms,
                         export_dir=None, dpi=EXPORT_DPI):
    """3-row raw/filtered/denoised waveform figure for one event's window."""
    start_utc = UTCDateTime(pd.Timestamp(window_start).to_pydatetime())
    end_utc = UTCDateTime(pd.Timestamp(window_end).to_pydatetime())

    channels = [name for name in ('raw', 'filtered', 'denoised') if name in waveforms]
    if not channels:
        raise ValueError('waveforms dict has none of raw/filtered/denoised')

    fig, axes = plt.subplots(len(channels), 1, figsize=(11, 2.6 * len(channels) + 1), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, channel in zip(axes, channels):
        sliced = waveforms[channel].copy().trim(starttime=start_utc, endtime=end_utc)
        ax.plot(sliced.times('matplotlib'), sliced.data, linewidth=.6, color='black')
        ax.set_title(channel.title(), loc='left', fontsize=11)
        ax.set_ylabel(WAVEFORM_UNITS)
        if WAVEFORM_YLIM is not None:
            ax.set_ylim(WAVEFORM_YLIM)
        if WAVEFORM_YTICKS is not None:
            ax.set_yticks(WAVEFORM_YTICKS)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-6, -6))
        ax.margins(x=0)
        ax.grid(alpha=.25)

    axes[-1].set_xlabel(TIME_LABEL)
    locator = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate()

    fig.suptitle(f'Event {event_number}  |  {label}', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))

    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / f'shishaldin_event_{event_number}_snr_check.png'
        fig.savefig(path, dpi=dpi, facecolor='white')
        print(f'Saved: {path}')

    return fig


# %% 1) Low vs. high SNR selection and visualization
def select_extreme_events(differences, n=N_SAMPLES, snr_column=SNR_COLUMN, kind='low'):
    """Return the n events with the lowest (kind='low') or highest
    (kind='high') value of `snr_column`."""
    if kind not in ('low', 'high'):
        raise ValueError("kind must be 'low' or 'high'")
    n = min(n, len(differences))
    ascending = kind == 'low'
    return differences.sort_values(snr_column, ascending=ascending).head(n).sort_values('event_time')


def select_random_negative_events(differences, n=N_SAMPLES, seed=None):
    """Return up to n rows of `differences` with a negative SNR difference
    (denoiser did worse than filtering), chosen uniformly at random
    without replacement. Different from select_extreme_events(): this
    samples by the sign of the *difference*, not by absolute SNR rank."""
    negative = differences[differences['snr_difference_db'] < 0]
    if negative.empty:
        raise ValueError('No negative SNR differences to sample from')
    n = min(n, len(negative))
    if n < N_SAMPLES:
        print(f'Only {len(negative)} negative event(s) available; using all of them.')
    rng = np.random.default_rng(seed)
    chosen_index = rng.choice(negative.index, size=n, replace=False)
    return negative.loc[chosen_index].sort_values('event_time')


def visualize_negative_events(sample, catalog, waveforms, export_dir=None):
    """Plot waveforms for a sample from select_random_negative_events(),
    titled with each event's SNR difference (mirrors the old
    shishaldin_negative_snr_waveforms.py behavior)."""
    figures = []
    for row in sample.itertuples():
        start, end = event_window(catalog, row.event_number)
        label = f'SNR difference (denoised \u2212 filtered) = {row.snr_difference_db:.2f} dB'
        fig = plot_event_waveforms(row.event_number, label, start, end, waveforms,
                                   export_dir=None)
        figures.append(fig)
    return figures


def visualize_snr_groups(low_events, high_events, catalog, waveforms,
                         snr_column=SNR_COLUMN, export_dir=None):
    """Plot each event's waveforms, tagged as LOW or HIGH SNR in the title
    and console output, so you can flip through both groups and judge by
    eye whether the low-SNR detections look like real signals."""
    figures = {'low': [], 'high': []}
    for group_name, events in (('low', low_events), ('high', high_events)):
        print(f'\n--- {group_name.upper()} SNR group ({snr_column}) ---')
        for row in events.itertuples():
            print(f'  Event {row.event_number}: filtered={row.filtered_snr_db:.2f} dB, '
                  f'denoised={row.denoised_snr_db:.2f} dB, '
                  f'difference={row.snr_difference_db:.2f} dB')
            start, end = event_window(catalog, row.event_number)
            snr_value = getattr(row, snr_column)
            label = f'{group_name.upper()} SNR ({snr_column} = {snr_value:.2f} dB)'
            fig = plot_event_waveforms(row.event_number, label, start, end, waveforms,
                                       export_dir=export_dir)
            figures[group_name].append(fig)
    return figures


# %% 2) SDR and cross-correlation between filtered and denoised
def signal_distortion_ratio(reference, estimate):
    """SDR (dB) = 10*log10(||reference||^2 / ||reference - estimate||^2).
    Higher means the estimate stayed closer to the reference waveform
    shape; lower/negative means it diverged substantially."""
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError('reference and estimate must be the same length')
    distortion_power = np.sum((reference - estimate) ** 2)
    reference_power = np.sum(reference ** 2)
    if distortion_power == 0:
        return float('inf')
    if reference_power == 0:
        return float('-inf')
    return 10 * np.log10(reference_power / distortion_power)


def best_lag_and_correlation(reference, estimate, sampling_rate, max_lag_seconds=MAX_LAG_SECONDS):
    """Normalized cross-correlation between reference and estimate,
    searched over +/- max_lag_seconds. Returns (best_lag_samples,
    peak_correlation) where peak_correlation is in [-1, 1]."""
    reference = np.asarray(reference, dtype=float) - np.mean(reference)
    estimate = np.asarray(estimate, dtype=float) - np.mean(estimate)

    correlation = scipy_signal.correlate(estimate, reference, mode='full')
    lags = scipy_signal.correlation_lags(len(estimate), len(reference), mode='full')

    max_lag_samples = int(round(max_lag_seconds * sampling_rate))
    window = np.abs(lags) <= max_lag_samples
    correlation, lags = correlation[window], lags[window]

    norm = np.sqrt(np.sum(reference ** 2) * np.sum(estimate ** 2))
    normalized = correlation / norm if norm > 0 else np.zeros_like(correlation)

    best_idx = np.argmax(np.abs(normalized))
    return int(lags[best_idx]), float(normalized[best_idx])


def align_and_score(reference, estimate, sampling_rate, max_lag_seconds=MAX_LAG_SECONDS):
    """Find the best alignment lag, shift to overlap, and return SDR and
    peak correlation at that alignment."""
    best_lag_samples, best_correlation = best_lag_and_correlation(
        reference, estimate, sampling_rate, max_lag_seconds)

    if best_lag_samples > 0:
        aligned_estimate = estimate[best_lag_samples:]
        aligned_reference = reference[:len(aligned_estimate)]
    elif best_lag_samples < 0:
        aligned_reference = reference[-best_lag_samples:]
        aligned_estimate = estimate[:len(aligned_reference)]
    else:
        aligned_reference, aligned_estimate = reference, estimate

    n = min(len(aligned_reference), len(aligned_estimate))
    if n == 0:
        return {'best_lag_seconds': best_lag_samples / sampling_rate,
                'max_correlation': best_correlation, 'sdr_db': float('nan')}

    sdr_db = signal_distortion_ratio(aligned_reference[:n], aligned_estimate[:n])
    return {'best_lag_seconds': best_lag_samples / sampling_rate,
           'max_correlation': best_correlation, 'sdr_db': sdr_db}


def compute_event_metrics(event_number, catalog, waveforms, reference_label='filtered',
                          estimate_label='denoised', padding_seconds=METRICS_PADDING_SECONDS,
                          max_lag_seconds=MAX_LAG_SECONDS):
    """Cross-correlation + SDR between two channels for one event's
    (tightly-windowed, by default) detection interval."""
    start, end = event_window(catalog, event_number, padding_seconds)
    start_utc = UTCDateTime(pd.Timestamp(start).to_pydatetime())
    end_utc = UTCDateTime(pd.Timestamp(end).to_pydatetime())

    reference_trace = waveforms[reference_label].copy().trim(starttime=start_utc, endtime=end_utc)
    estimate_trace = waveforms[estimate_label].copy().trim(starttime=start_utc, endtime=end_utc)
    if reference_trace.stats.sampling_rate != estimate_trace.stats.sampling_rate:
        raise ValueError(f'{reference_label} and {estimate_label} traces have different '
                         'sampling rates; resample one to match before calling this')
    if len(reference_trace.data) == 0 or len(estimate_trace.data) == 0:
        raise ValueError(f'Event {event_number}: empty trace after trimming to its window')

    metrics = align_and_score(reference_trace.data.astype(float), estimate_trace.data.astype(float),
                              reference_trace.stats.sampling_rate, max_lag_seconds)
    metrics['event_number'] = event_number
    return metrics


def summarize_group_metrics(events, catalog, waveforms, group_label):
    """compute_event_metrics() for every event in `events`, tagged with
    group_label, as a DataFrame."""
    rows = []
    for event_number in events['event_number']:
        metrics = compute_event_metrics(event_number, catalog, waveforms)
        metrics['group'] = group_label
        rows.append(metrics)
    return pd.DataFrame(rows)


def plot_metric_comparison(metrics_low, metrics_high, export_dir=None, dpi=EXPORT_DPI):
    """Side-by-side boxplots of peak correlation and SDR for the low- vs.
    high-SNR groups, to see whether low-SNR events are also systematically
    less internally consistent (lower correlation/SDR) -- one more piece
    of evidence for "real signal" vs. "noise the denoiser reshaped"."""
    fig, (ax_corr, ax_sdr) = plt.subplots(1, 2, figsize=(10, 5))

    ax_corr.boxplot([metrics_low['max_correlation'], metrics_high['max_correlation']],
                    labels=['Low SNR', 'High SNR'])
    ax_corr.set_ylabel('Peak cross-correlation (filtered vs. denoised)')
    ax_corr.set_title('Waveform correlation')
    ax_corr.grid(alpha=.25)

    sdr_low = metrics_low.loc[np.isfinite(metrics_low['sdr_db']), 'sdr_db']
    sdr_high = metrics_high.loc[np.isfinite(metrics_high['sdr_db']), 'sdr_db']
    n_dropped = (len(metrics_low) - len(sdr_low)) + (len(metrics_high) - len(sdr_high))
    if n_dropped:
        print(f'Note: excluded {n_dropped} event(s) with infinite SDR '
              f'(filtered and denoised were identical) from the SDR boxplot.')
    ax_sdr.boxplot([sdr_low, sdr_high], labels=['Low SNR', 'High SNR'])
    ax_sdr.set_ylabel('SDR (dB), filtered as reference')
    ax_sdr.set_title('Signal-to-distortion ratio')
    ax_sdr.grid(alpha=.25)

    fig.suptitle('Low-SNR vs. high-SNR events: waveform consistency between '
                'filtered and denoised')
    fig.tight_layout()

    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / 'shishaldin_low_vs_high_snr_metrics.png'
        fig.savefig(path, dpi=dpi, facecolor='white')
        print(f'Saved: {path}')

    return fig


# %% Run
if __name__ == '__main__':
    # This block assumes `catalog` and `waveforms` are already built the
    # same way as in your notebook (parsed-datetime catalog DataFrame,
    # and a {'raw':..., 'filtered':..., 'denoised':...} dict of ObsPy
    # traces). See the accompanying cell snippet for the notebook version.
    raise SystemExit(
        'This module is meant to be imported (into a notebook that already '
        'has `catalog` and `waveforms` defined), not run standalone. '
        'See the docstring / accompanying usage snippet.')
