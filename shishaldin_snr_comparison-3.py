# %% Imports
"""Compare filtered vs. denoised SNR for matched Shishaldin detections.

Loads a catalog sheet with shishaldin_catalog_explorer.load_catalogs(),
matches each event's 'filtered' and 'denoised' detection rows, and computes
the per-event SNR difference (denoised - filtered, in dB). Two views of
that same difference are plotted: a histogram with mean +/- 1 SD, and a
time series across the sheet's full span, which is the more useful one for
spotting whether the denoiser's advantage is steady or clusters around
particular stretches of time (storms, tremor episodes, time of day, etc.)
-- something a single aggregate number can hide.

Run this from the same folder as shishaldin_catalog_explorer.py (it
imports load_catalogs from there rather than duplicating the loader).
The sheet you point it at must already contain BOTH detection_type rows
for the events you want to compare.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from shishaldin_catalog_explorer import load_catalogs

# %% Editable settings
CATALOG_PATH = Path(
    '/Users/amelega/Desktop/figures/'
    'Longer period Shishaldin Event Detection (hours).xlsx'
)
SHEET = '2 hours'          # Sheet to compare; must contain both labels below
FILTERED_LABEL = 'filtered'
DENOISED_LABEL = 'denoised'
HIST_BINS = 20
ROLLING_WINDOW = '1D'      # pandas offset alias: smoothing window for the time-series trend line
TIME_LABEL = 'Catalog time (timezone unspecified)'
EXPORT_DIR = Path('/Users/dfee/repos/avo_research/figs/volc_denoise_catalog')
EXPORT_DPI = 300

# Record the parameters actually used to produce the "filtered" catalog
# rows here (edit these to match your own processing pipeline). They are
# only for display -- printed to the console and annotated on the plot --
# and play no part in the SNR calculation itself.
FILTER_PARAMS = {
    'filter_type': 'bandpass',
    'freqmin_hz': 0.9,
    'freqmax_hz': 8.0,
    'corners': 2,
}


# %% Matching and stats
def matched_snr_differences(catalog, filtered_label=FILTERED_LABEL, denoised_label=DENOISED_LABEL):
    """Return a per-event DataFrame of filtered SNR, denoised SNR, and
    their difference, restricted to events that have both detection types.

    If an event has more than one row of a given detection_type, its SNR
    values are averaged first (with a printed note) so each event
    contributes exactly one paired comparison.
    """
    present = set(catalog.detection_type.unique())
    missing = {filtered_label, denoised_label} - present
    if missing:
        raise ValueError(
            f'Sheet is missing detection_type(s) {sorted(missing)}; '
            f'found {sorted(present)}')

    counts = catalog.groupby(['event_number', 'detection_type']).size()
    duplicated = counts[counts > 1]
    if not duplicated.empty:
        print(f'Note: averaging {len(duplicated)} event/type combination(s) '
              f'that had more than one detection row.')

    per_event = (catalog.groupby(['event_number', 'detection_type'])['snr_db']
                 .mean()
                 .unstack('detection_type'))

    both = per_event[[filtered_label, denoised_label]].dropna()
    skipped = len(per_event) - len(both)
    if skipped:
        print(f'Skipping {skipped} event(s) missing a {filtered_label!r} or '
              f'{denoised_label!r} detection.')
    if both.empty:
        raise ValueError('No events have both a filtered and a denoised detection')

    result = both.rename(columns={filtered_label: 'filtered_snr_db',
                                  denoised_label: 'denoised_snr_db'})
    result['snr_difference_db'] = result['denoised_snr_db'] - result['filtered_snr_db']

    # Earliest start_time across all of that event's rows (any detection
    # type), used as its single position on the time-series plot below.
    event_time = catalog.groupby('event_number')['start_time'].min()
    result = result.join(event_time.rename('event_time'))

    return result.reset_index()


def plot_snr_difference_histogram(differences, sheet, filter_params=FILTER_PARAMS,
                                  bins=HIST_BINS, export_dir=None, dpi=EXPORT_DPI):
    """Histogram the per-event SNR difference; return (figure, mean, std)."""
    values = differences['snr_difference_db'].to_numpy()
    mean = values.mean()
    std = values.std(ddof=1) if len(values) > 1 else float('nan')

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values, bins=bins, color='tab:blue', edgecolor='white', alpha=.85)
    ax.axvline(mean, color='black', linewidth=1.5, label=f'Mean = {mean:.2f} dB')
    if not np.isnan(std):
        ax.axvspan(mean - std, mean + std, color='black', alpha=.08,
                  label=f'±1 SD = {std:.2f} dB')
    ax.axvline(0, color='0.5', linewidth=1, linestyle='--', label='No improvement')
    ax.set_xlabel('SNR improvement, denoised \u2212 filtered (dB)')
    ax.set_ylabel('Number of events')
    ax.set_title(f'DeepDenoiser vs. filtering SNR improvement \u2014 {sheet}\n'
                f'n = {len(values)} matched events')
    ax.legend(loc='upper right')
    ax.grid(alpha=.25)

    param_lines = '\n'.join(f'{key}: {value}' for key, value in filter_params.items())
    ax.text(.02, .98, f'Filter parameters\n{param_lines}', transform=ax.transAxes,
           fontsize=8, family='monospace', va='top', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=.85, edgecolor='0.7'))

    fig.tight_layout()

    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        sheet_tag = ''.join(c if c.isalnum() else '_' for c in sheet)
        path = export_dir / f'shishaldin_{sheet_tag}_snr_difference_hist.png'
        fig.savefig(path, dpi=dpi, facecolor='white')
        print(f'Saved histogram: {path}')

    return fig, mean, std


def plot_snr_difference_timeseries(differences, sheet, rolling_window=ROLLING_WINDOW,
                                   export_dir=None, dpi=EXPORT_DPI):
    """Scatter the per-event SNR difference against event time, with a
    rolling-mean trend line, so you can see whether the denoiser's
    advantage holds steady across the sheet or clusters in certain
    stretches (storms, tremor episodes, time of day, etc.). Cross-reference
    any clusters you spot against your own weather/tremor/activity records
    -- this plot only shows *that* it clusters, not *why*.

    Returns the figure.
    """
    series = differences.set_index('event_time')['snr_difference_db'].sort_index()
    rolling_mean = series.rolling(rolling_window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors = np.where(series.to_numpy() >= 0, 'tab:blue', 'tab:red')
    ax.scatter(series.index, series.to_numpy(), c=colors, s=28, alpha=.75,
              label='Per-event improvement', zorder=3)
    ax.plot(rolling_mean.index, rolling_mean.to_numpy(), color='black', linewidth=1.8,
           label=f'{rolling_window} rolling mean', zorder=4)
    ax.axhline(0, color='0.5', linewidth=1, linestyle='--', zorder=2)

    ax.set_ylabel('SNR improvement, denoised \u2212 filtered (dB)')
    ax.set_xlabel(TIME_LABEL)
    ax.set_title(f'SNR improvement over time \u2014 {sheet}\n'
                f'n = {len(series)} matched events; blue = denoiser ahead, red = filtering ahead')
    ax.legend(loc='upper right')
    ax.grid(alpha=.25)

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate()
    fig.tight_layout()

    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        sheet_tag = ''.join(c if c.isalnum() else '_' for c in sheet)
        path = export_dir / f'shishaldin_{sheet_tag}_snr_difference_timeseries.png'
        fig.savefig(path, dpi=dpi, facecolor='white')
        print(f'Saved time series: {path}')

    return fig


# %% Run
if __name__ == '__main__':
    catalogs = load_catalogs(CATALOG_PATH)
    if SHEET not in catalogs:
        raise ValueError(f'Unknown sheet {SHEET!r}; choose {list(catalogs)}')
    catalog = catalogs[SHEET]

    differences = matched_snr_differences(catalog)
    fig, mean_diff, std_diff = plot_snr_difference_histogram(
        differences, SHEET, export_dir=EXPORT_DIR)
    timeseries_fig = plot_snr_difference_timeseries(
        differences, SHEET, export_dir=EXPORT_DIR)

    print(f'\nSheet: {SHEET}')
    print(f'Matched events: {len(differences)}')
    print(f'Mean SNR improvement: {mean_diff:.2f} dB')
    if np.isnan(std_diff):
        print('Standard deviation:   n/a (only one matched event)')
    else:
        print(f'Standard deviation:   {std_diff:.2f} dB')

    print('\nFiltering parameters used to produce the "filtered" catalog:')
    for key, value in FILTER_PARAMS.items():
        print(f'  {key}: {value}')

    plt.show()
