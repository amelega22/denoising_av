# %% Imports
"""Explore Shishaldin detection catalogs, cell by cell or as a script.

Requires pandas, openpyxl, and matplotlib. In IPython, run `%matplotlib qt`
(or `%matplotlib widget` in Jupyter with ipympl installed) before the plot
cell. In PyCharm, use an external interactive plot window if its embedded
plot pane does not support widgets. Spreadsheet timestamps are displayed
as written; no timezone is assumed. This workbook contains catalog metrics,
not waveforms. Selecting an event zooms to all its detection rows.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button, RadioButtons, RangeSlider, TextBox

# %% Editable settings
CATALOG_PATH = Path(
    '/Users/dfee/Documents/AVO/volc_denoising/'
    'Longer period Shishaldin Event Detection (hours).xlsx'
)
SHEET = '2 hours'
EVENT = 3                 # Event number, or None for the full catalog
START_TIME = None            # e.g. '2025-11-01 00:15:00'
END_TIME = None              # e.g. '2025-11-01 00:45:00'
EVENT_PADDING_SECONDS = 30
COLORS = {'filtered': 'tab:orange', 'denoised': 'tab:blue'}
EXPORT_DIR = Path('/Users/dfee/repos/avo_research/figs/volc_denoise_catalog')
EXPORT_TITLE = 'Shishaldin event detections'
RMS_UNITS = 'catalog units'   # Replace with the physical units when known
TIME_LABEL = 'Catalog time (timezone unspecified)'
EXPORT_DPI = 300             # 16 x 9 inches gives 4800 x 2700 pixels


# %% Load catalogs (available here for your own cell-based analysis)
def load_catalogs(path):
    """Read nonempty catalog rows and reject malformed event intervals."""
    sheets = pd.read_excel(path, sheet_name=None, engine='openpyxl')
    catalogs = {}
    required = ['event_number', 'detection_type', 'start_time', 'end_time',
                'duration_seconds', 'snr_db', 'signal_rms', 'noise_rms']
    for name, frame in sheets.items():
        frame = frame.dropna(how='all').copy()
        frame.columns = frame.columns.astype(str).str.strip()
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f'{name}: missing columns {sorted(missing)}')
        frame = frame.loc[:, ~frame.columns.str.startswith('Unnamed:')]
        for col in ['start_time', 'end_time']:
            frame[col] = pd.to_datetime(frame[col], errors='raise')
        for col in ['event_number', 'duration_seconds', 'snr_db', 'signal_rms', 'noise_rms']:
            frame[col] = pd.to_numeric(frame[col], errors='raise')
        if frame[required].isna().any().any():
            raise ValueError(f'{name}: missing required catalog values')
        if (frame.end_time < frame.start_time).any():
            raise ValueError(f'{name}: end time precedes start time')
        if (frame.event_number % 1 != 0).any():
            raise ValueError(f'{name}: event numbers must be integers')
        frame['event_number'] = frame.event_number.astype(int)
        frame['detection_type'] = frame.detection_type.astype(str).str.strip()
        if not frame.empty:
            catalogs[name] = frame.sort_values('start_time').reset_index(drop=True)
    if not catalogs:
        raise ValueError('No catalog detections found')
    return catalogs


# %% Interactive plotting helpers
class CatalogExplorer:
    def __init__(self, catalogs, sheet, padding=30):
        if sheet not in catalogs:
            raise ValueError(f'Unknown sheet {sheet!r}; choose {list(catalogs)}')
        if padding < 0:
            raise ValueError('Event padding must be nonnegative')
        self.catalogs, self.sheet, self.padding = catalogs, sheet, padding
        self.selected = None
        self.fig, (self.timeline, self.metric, self.rms) = plt.subplots(
            3, 1, figsize=(13, 10), sharex=True)
        self.fig.subplots_adjust(left=.09, right=.77, bottom=.32, top=.91, hspace=.16)
        self.sheet_control = RadioButtons(self.fig.add_axes([.81, .74, .17, .16]), list(catalogs),
                                          active=list(catalogs).index(sheet))
        self.event_box = TextBox(self.fig.add_axes([.85, .65, .11, .045]), 'Event ', initial='')
        self.previous = Button(self.fig.add_axes([.80, .58, .08, .045]), 'Previous')
        self.next = Button(self.fig.add_axes([.89, .58, .08, .045]), 'Next')
        self.reset = Button(self.fig.add_axes([.81, .50, .15, .05]), 'Full catalog')
        self.start_box = TextBox(self.fig.add_axes([.14, .17, .26, .04]), 'Start ')
        self.end_box = TextBox(self.fig.add_axes([.49, .17, .26, .04]), 'End ')
        self.export_button = Button(self.fig.add_axes([.80, .10, .16, .045]), 'Export PNG')
        self.export_button.on_clicked(self.export_png)
        self.apply = Button(self.fig.add_axes([.80, .17, .16, .04]), 'Apply times')
        self.details = self.fig.text(.09, .04, '', fontsize=9, va='bottom', family='monospace')
        self.status = self.fig.text(.09, .255, '', fontsize=9)
        self.slider_ax = self.fig.add_axes([.14, .225, .61, .02])
        self.sheet_control.on_clicked(self.change_sheet)
        self.event_box.on_submit(self.select_event)
        self.previous.on_clicked(lambda _: self.step(-1))
        self.next.on_clicked(lambda _: self.step(1))
        self.reset.on_clicked(lambda _: self.full_catalog())
        self.apply.on_clicked(lambda _: self.apply_times())
        self.change_sheet(sheet)

    @property
    def data(self):
        return self.catalogs[self.sheet]

    def change_sheet(self, name):
        self.sheet, self.selected = name, None
        self.sheet_control.eventson = False
        self.sheet_control.set_active(list(self.catalogs).index(name))
        self.sheet_control.eventson = True
        self.events = sorted(self.data.event_number.unique())
        self.origin = self.data.start_time.min()
        self.limit = max((self.data.end_time.max() - self.origin).total_seconds(), 1)
        if hasattr(self, 'slider'):
            self.slider.disconnect_events()
        self.slider_ax.clear()
        self.slider = RangeSlider(self.slider_ax, 'Minutes ', 0, self.limit / 60,
                                  valinit=(0, self.limit / 60), valfmt='%.2f')
        self.slider.on_changed(self.slider_changed)
        self.set_event_text('')
        self.set_window(self.origin, self.origin + pd.Timedelta(seconds=self.limit))

    def set_event_text(self, text):
        self.event_box.eventson = False
        self.event_box.set_val(text)
        self.event_box.eventson = True

    def set_window(self, start, end):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if pd.isna(start) or pd.isna(end) or start.tzinfo or end.tzinfo:
            raise ValueError('Enter timezone-free catalog timestamps')
        if end <= start:
            raise ValueError('End must be later than start')
        if start < self.origin or end > self.origin + pd.Timedelta(seconds=self.limit):
            raise ValueError('Time window must be inside the catalog extent')
        self.start, self.end = start, end
        self.slider.eventson = False
        self.slider.set_val(((start-self.origin).total_seconds()/60,
                             (end-self.origin).total_seconds()/60))
        self.slider.eventson = True
        self.start_box.set_val(str(start))
        self.end_box.set_val(str(end))
        self.draw()

    def slider_changed(self, values):
        if values[1] <= values[0]:
            self.status.set_text('Choose a nonzero time window')
            self.fig.canvas.draw_idle()
            return
        self.set_window(*(self.origin + pd.Timedelta(minutes=v) for v in values))

    def apply_times(self):
        try:
            self.set_window(self.start_box.text, self.end_box.text)
        except (ValueError, TypeError) as exc:
            self.status.set_text(str(exc))
            self.fig.canvas.draw_idle()

    def select_event(self, value):
        try:
            event = int(value)
            if event not in self.events:
                raise ValueError('Event number not found in this sheet')
        except (ValueError, TypeError):
            self.status.set_text('Enter an event number from this sheet')
            self.fig.canvas.draw_idle()
            return
        self.selected = event
        self.set_event_text(str(event))
        rows = self.data[self.data.event_number == event]
        pad = pd.Timedelta(seconds=self.padding)
        start = max(self.origin, rows.start_time.min() - pad)
        end = min(self.origin + pd.Timedelta(seconds=self.limit), rows.end_time.max() + pad)
        if end <= start:
            start = max(self.origin, start - pd.Timedelta(seconds=1))
            end = min(self.origin + pd.Timedelta(seconds=self.limit), end + pd.Timedelta(seconds=1))
        self.set_window(start, end)

    def step(self, direction):
        index = self.events.index(self.selected) + direction if self.selected in self.events else (0 if direction > 0 else len(self.events)-1)
        self.select_event(self.events[max(0, min(index, len(self.events)-1))])

    def full_catalog(self):
        self.selected = None
        self.set_event_text('')
        self.set_window(self.origin, self.origin + pd.Timedelta(seconds=self.limit))

    def render_panels(self, axes, rows, start, end, presentation=False):
        """Draw the same catalog data for the explorer and the clean export."""
        timeline, metric, rms = axes
        types = sorted(self.data.detection_type.unique())
        timeline.clear()
        metric.clear()
        rms.clear()
        for y, kind in enumerate(types):
            group = rows[rows.detection_type == kind]
            color = COLORS.get(kind, f'C{y}')
            timeline.hlines([y]*len(group), group.start_time, group.end_time,
                                  color=color, linewidth=7 if presentation else 5, label=kind)
            metric.scatter(group.start_time, group.snr_db, color=color, s=48 if presentation else 24, label=kind)
            for column, marker, label in [('signal_rms', 'o', 'signal'),
                                          ('noise_rms', 'x', 'noise')]:
                rms.scatter(group.start_time, group[column], color=color,
                                 marker=marker, s=48 if presentation else 24, label=f'{kind} {label}')
            selected = group[group.event_number == self.selected]
            if not selected.empty:
                timeline.scatter(selected.start_time, [y]*len(selected), s=140 if presentation else 100,
                                      facecolors='none', edgecolors='black', zorder=4)
                metric.scatter(selected.start_time, selected.snr_db, s=140 if presentation else 100,
                                    facecolors='none', edgecolors='black', zorder=4)
                for column in ('signal_rms', 'noise_rms'):
                    rms.scatter(selected.start_time, selected[column], s=140 if presentation else 100,
                                     facecolors='none', edgecolors='black', zorder=4)
        timeline.set_yticks(range(len(types)), types)
        timeline.set_ylim(-.6, len(types)-.4)
        if not presentation:
            timeline.set_title(f'Shishaldin detections — {self.sheet}')
        metric.set_ylabel('SNR (dB)')
        rms.set_ylabel(f'RMS ({RMS_UNITS})')
        rms.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        rms.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12 if presentation else 8)
        rms.set_xlabel(TIME_LABEL)
        locator = mdates.AutoDateLocator()
        rms.xaxis.set_major_locator(locator)
        rms.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        rms.set_xlim(start, end)
        rms.set_yscale('linear')
        for ax in (timeline, metric, rms):
            ax.grid(alpha=.25)

    def export_png(self, _event=None, output_dir=None):
        """Save the displayed time window as a clean PNG; return its path.

        Can also be called from a cell: explorer.export_png(). Repeated
        exports get numbered filenames rather than overwriting earlier ones.
        """
        figure = None
        try:
            folder = Path(output_dir) if output_dir is not None else EXPORT_DIR
            folder.mkdir(parents=True, exist_ok=True)
            # Read the axes so toolbar pan/zoom is included in the export.
            limits = sorted(self.rms.get_xlim())
            start, end = [pd.Timestamp(mdates.num2date(v)).tz_localize(None)
                          for v in limits]
            rows = self.data[(self.data.end_time >= start) & (self.data.start_time <= end)]
            with plt.rc_context({'font.size': 14, 'axes.labelsize': 15,
                                 'xtick.labelsize': 13, 'ytick.labelsize': 13,
                                 'savefig.facecolor': 'white'}):
                figure, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
                figure.subplots_adjust(left=.12, right=.80, bottom=.12,
                                       top=.81, hspace=.38)
                self.render_panels(axes, rows, start, end, presentation=True)
                for ax, title in zip(axes, ['A  Detections', 'B  SNR', 'C  RMS']):
                    ax.set_title(title, loc='left', fontsize=16, fontweight='bold')
                    ax.spines[['top', 'right']].set_visible(False)
                figure.suptitle(EXPORT_TITLE, x=.12, y=.97, ha='left',
                                fontsize=24, fontweight='bold')
                subtitle = f'{self.sheet} catalog   |   {start} to {end}'
                figure.text(.12, .92, subtitle, fontsize=13)
                note = f'{len(rows)} detections / {rows.event_number.nunique()} events in view'
                if self.selected is not None and (rows.event_number == self.selected).any():
                    note += f'   |   Event {self.selected} outlined in black'
                figure.text(.12, .875, note, fontsize=12, color='0.3')
                sheet_tag = ''.join(c if c.isalnum() else '_' for c in self.sheet)
                event_tag = f'event_{self.selected}' if self.selected is not None else 'overview'
                stem = (f'shishaldin_{sheet_tag}_{event_tag}_'
                        f'{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}')
                path = folder / f'{stem}.png'
                number = 2
                while path.exists():
                    path = folder / f'{stem}_{number}.png'
                    number += 1
                # Keep the exact 16:9 canvas (no tight bounding-box crop).
                figure.savefig(path, dpi=EXPORT_DPI, format='png',
                               facecolor='white', transparent=False, bbox_inches=None)
            self.status.set_text(f'Saved PNG: {path.name}')
            print(f'Saved presentation figure: {path}')
            return path
        except (OSError, ValueError) as exc:
            self.status.set_text(f'PNG export failed: {exc}')
            print(f'PNG export failed: {exc}')
            return None
        finally:
            if figure is not None:
                plt.close(figure)
            self.fig.canvas.draw_idle()

    def draw(self):
        rows = self.data[(self.data.end_time >= self.start) & (self.data.start_time <= self.end)]
        self.render_panels((self.timeline, self.metric, self.rms), rows, self.start, self.end)
        self.status.set_text(f'{len(rows)} detection rows / {rows.event_number.nunique()} events overlap this window')
        if self.selected is None:
            self.details.set_text('Select an event number to zoom and compare its detections.\nDrag the minute handles or enter exact start/end timestamps and click Apply times.')
        else:
            selected = self.data[self.data.event_number == self.selected]
            lines = [f'Event {self.selected} (this sheet only)']
            for row in selected.itertuples():
                lines.append(f'{row.detection_type:10s} {row.start_time}  to  {row.end_time}  '
                             f'duration={row.duration_seconds:g} s  SNR={row.snr_db:g} dB')
            self.details.set_text('\n'.join(lines))
        self.fig.canvas.draw_idle()


# %% Read workbook
if __name__ == '__main__':
    catalogs = load_catalogs(CATALOG_PATH)
    catalog = catalogs[SHEET]  # DataFrame for additional analysis in your editor
    print({name: len(frame) for name, frame in catalogs.items()})

# %% Launch interactive figure (keep explorer alive so controls remain responsive)
if __name__ == '__main__':
    explorer = CatalogExplorer(catalogs, SHEET, EVENT_PADDING_SECONDS)
    if EVENT is not None:
        explorer.select_event(EVENT)
    if START_TIME is not None or END_TIME is not None:
        explorer.set_window(START_TIME if START_TIME is not None else explorer.start,
                            END_TIME if END_TIME is not None else explorer.end)
    plt.show()
