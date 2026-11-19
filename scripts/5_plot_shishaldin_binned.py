# %% Plot color code, reduced displacement, VOISS-Net classes, and detections
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import dates, rcParams
from matplotlib.colors import ListedColormap
import numpy as np
from obspy import UTCDateTime
import pandas as pd


# %% Settings
# STOPTIME is exclusive, so this plots through the end of November 15.
STARTTIME = UTCDateTime(2025, 8, 1)
STOPTIME = UTCDateTime(2026, 11, 16)

TOP_DIR = Path("/Users/dfee/Documents/AVO/shishaldin/2023")
COLOR_CODE_PATH = TOP_DIR / "shish_colorcode_2016-2025.csv"
INDICATORS_PATH = TOP_DIR / "indicators/shishaldin_2025-2026_indicators.pkl"
DR_PATHS = [
    TOP_DIR / "figs/longterm_plot/shish_dr_20250101_20260101_5.0min.npy",
    TOP_DIR / "figs/longterm_plot/shish_dr_20260101_20261023_5.0min.npy",
]

# Set this to a specific run folder if more than one denoising run overlaps the
# plotting window. If left as None, the overlapping run with the widest time
# coverage is used.
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"
DETECTION_RUN_DIR = None
DETECTION_PIPELINE = "denoise_then_filter"

SV_DIR = '/Users/dfee/Documents/AVO/denoising/figs/'
SAVE = True
SHOW = False
TRANSPARENT = None
FIGSIZE = (9, 7)
HEIGHT_RATIOS = [1.2, 3, 5, 3]
FS = 8
rcParams.update({"font.size": FS})

CLASSIFICATION_INTERVAL = 60
BINNING_INTERVAL = 60 * 60
PNORM_THRESH = 0.4
DR_SCA = 1

CLASS_DICT = {
    0: ("BT", [193, 39, 45]),
    1: ("HT", [0, 129, 118]),
    2: ("MT", [0, 0, 167]),
    3: ("EQ", [238, 204, 22]),
    4: ("LP", [103, 72, 132]),
    5: ("EX", [164, 98, 0]),
    6: ("NO", [40, 40, 40]),
}


# %% Load data
def load_color_code(path):
    color_code = pd.read_csv(path)
    color_code["start"] = pd.to_datetime(color_code["start"], utc=True)
    color_code["end"] = pd.to_datetime(color_code["end"], utc=True)
    color_code["color"] = color_code["code"].str.lower().replace(
        "unassigned", "grey"
    )
    return color_code


def load_reduced_displacement(paths, starttime, stoptime):
    time_arrays = []
    dr_arrays = []

    for path in paths:
        if not path.exists():
            print(f"Reduced-displacement file not found; skipping {path}")
            continue

        times, dr = np.load(path)
        keep = (
            (times >= starttime.matplotlib_date)
            & (times < stoptime.matplotlib_date)
        )
        time_arrays.append(times[keep])
        dr_arrays.append(dr[keep] / DR_SCA)

    if not time_arrays:
        return np.array([]), np.array([])

    times = np.concatenate(time_arrays)
    dr = np.concatenate(dr_arrays)
    order = np.argsort(times)
    return times[order], dr[order]


def indicators_to_binned_matrix(
    indicators_path,
    starttime,
    stoptime,
    nclasses,
    classification_interval=60,
    binning_interval=3600,
    pnorm_thresh=None,
):
    """Convert the station classifications into the binned VOISS-Net panel."""
    indicators = pd.read_pickle(indicators_path)
    indicators = [
        indicator
        for indicator in indicators
        if starttime <= indicator[1] < stoptime
    ]

    nsteps = int((stoptime - starttime) / classification_interval)
    intervals_per_bin = int(binning_interval / classification_interval)
    nbins = nsteps // intervals_per_bin
    nsteps = nbins * intervals_per_bin

    stations = sorted({indicator[0] for indicator in indicators})
    matrix_probs = np.zeros((len(stations), nsteps, nclasses), dtype=np.float32)

    for station, time, _, probabilities in indicators:
        col = int((time - starttime) / classification_interval)
        if 0 <= col < nsteps:
            row = stations.index(station)
            matrix_probs[row, col, :] = probabilities

    matrix_probs_sum = np.sum(matrix_probs, axis=0)
    contributing_stations = np.sum(np.sum(matrix_probs, axis=2) != 0, axis=0)
    voted_timeline = np.argmax(matrix_probs_sum, axis=1)
    voted_timeline[contributing_stations == 0] = nclasses

    if pnorm_thresh is not None:
        voted_probability = np.divide(
            np.max(matrix_probs_sum, axis=1),
            contributing_stations,
            out=np.zeros(nsteps, dtype=float),
            where=contributing_stations != 0,
        )
        voted_timeline[voted_probability < pnorm_thresh] = nclasses

    voted_timeline = voted_timeline.reshape(nbins, intervals_per_bin)
    count_array = np.zeros((nbins, nclasses), dtype=float)
    for class_number in range(nclasses):
        count_array[:, class_number] = np.sum(
            voted_timeline == class_number, axis=1
        )

    matrix_plot = np.repeat(np.arange(nclasses)[:, None], nbins, axis=1)
    matrix_plot = np.flip(matrix_plot, axis=0)
    matrix_alpha = np.sqrt(count_array / intervals_per_bin).T
    matrix_alpha = np.flip(matrix_alpha, axis=0)
    return matrix_plot, matrix_alpha


def parse_run_times(run_dir):
    match = re.search(
        r"_(\d{8}T\d{6})_(\d{8}T\d{6})$", run_dir.name
    )
    if match is None:
        return None
    return tuple(UTCDateTime(value) for value in match.groups())


def find_detection_run(output_root, pipeline, starttime, stoptime):
    candidates = []
    for run_dir in output_root.glob("*"):
        run_times = parse_run_times(run_dir)
        if run_times is None:
            continue
        run_start, run_stop = run_times
        summary_paths = sorted(
            (run_dir / pipeline / "weekly").glob(
                "week_*/detections_summary.json"
            )
        )
        overlap = max(
            0,
            min(float(run_stop), float(stoptime))
            - max(float(run_start), float(starttime)),
        )
        if summary_paths and overlap > 0:
            candidates.append((overlap, run_stop - run_start, run_dir))

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def load_detections(run_dir, pipeline, starttime, stoptime):
    if run_dir is None:
        empty = pd.DatetimeIndex([], tz="UTC")
        return {"filtered": empty, "denoised": empty}, None

    run_dir = Path(run_dir)
    event_times = {"filtered": [], "denoised": []}
    summary_paths = sorted(
        (run_dir / pipeline / "weekly").glob(
            "week_*/detections_summary.json"
        )
    )
    for path in summary_paths:
        with open(path, "r") as f:
            summary = json.load(f)
        for kind in event_times:
            key = f"{kind}_network_coincidences"
            for interval in summary.get(key, []):
                event_times[kind].append(interval["start"])

    for kind, values in event_times.items():
        times = pd.to_datetime(values, errors="coerce", utc=True)
        times = times[~times.isna()]
        keep = (
            (times >= pd.Timestamp(starttime.datetime, tz="UTC"))
            & (times < pd.Timestamp(stoptime.datetime, tz="UTC"))
        )
        event_times[kind] = times[keep].sort_values()

    return event_times, parse_run_times(run_dir)


def bin_detections(event_times, run_times, starttime, stoptime):
    bins = pd.date_range(
        start=starttime.datetime,
        end=stoptime.datetime,
        freq="1h",
        inclusive="left",
        tz="UTC",
    )
    counts = pd.Series(np.nan, index=bins, dtype=float)
    if run_times is None:
        return counts

    run_start, run_stop = run_times
    covered = (
        (bins >= pd.Timestamp(run_start.datetime, tz="UTC").floor("h"))
        & (bins < pd.Timestamp(run_stop.datetime, tz="UTC").ceil("h"))
    )
    counts.loc[covered] = 0

    if len(event_times):
        event_counts = pd.Series(1, index=event_times).resample("1h").sum()
        common = counts.index.intersection(event_counts.index)
        counts.loc[common] = event_counts.loc[common]
    return counts


# %% Plot
def main():
    nclasses = len(CLASS_DICT)
    xlim = [STARTTIME.matplotlib_date, STOPTIME.matplotlib_date]

    date_range = pd.date_range(
        start=STARTTIME.datetime, end=STOPTIME.datetime, freq="MS"
    )
    xticks = dates.date2num(date_range)
    xticklabels = [
        date.strftime("%Y/%m") if i % 2 == 0 else ""
        for i, date in enumerate(date_range)
    ]

    print("Loading color code and reduced displacement...")
    color_code = load_color_code(COLOR_CODE_PATH)
    dr_times, dr = load_reduced_displacement(
        DR_PATHS, STARTTIME, STOPTIME
    )

    print("Generating VOISS-Net binned timeline...")
    matrix_plot, matrix_alpha = indicators_to_binned_matrix(
        INDICATORS_PATH,
        STARTTIME,
        STOPTIME,
        nclasses,
        classification_interval=CLASSIFICATION_INTERVAL,
        binning_interval=BINNING_INTERVAL,
        pnorm_thresh=PNORM_THRESH,
    )

    detection_run = (
        Path(DETECTION_RUN_DIR)
        if DETECTION_RUN_DIR is not None
        else find_detection_run(
            OUTPUT_ROOT,
            DETECTION_PIPELINE,
            STARTTIME,
            STOPTIME,
        )
    )
    if detection_run is None:
        print("No overlapping denoising detection run found.")
    else:
        print(f"Loading denoising detections from {detection_run}")
    event_times, run_times = load_detections(
        detection_run,
        DETECTION_PIPELINE,
        STARTTIME,
        STOPTIME,
    )
    detection_counts = {
        kind: bin_detections(times, run_times, STARTTIME, STOPTIME)
        for kind, times in event_times.items()
    }

    # An earlier version of this script selected the non-interactive Agg
    # backend. IDE consoles keep that backend until they are restarted, so
    # switch back to the native macOS backend when needed.
    if SHOW and plt.get_backend().lower() == "agg":
        try:
            plt.switch_backend("MacOSX")
        except ImportError:
            print(
                "Matplotlib is using the non-interactive Agg backend. "
                "Restart the IDE's Python console, then run this file again."
            )
    print(f"Matplotlib backend: {plt.get_backend()}")

    if SHOW:
        plt.ion()
    else:
        plt.ioff()
    fig, axs = plt.subplots(
        4,
        1,
        figsize=FIGSIZE,
        sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS},
    )

    # AVO color code
    for row in color_code.itertuples():
        axs[0].axvspan(row.start, row.end, color=row.color)
    axs[0].set_ylim([0, 1])
    axs[0].set_yticks([])
    axs[0].set_ylabel(
        "AVO", rotation=0, labelpad=12, verticalalignment="center"
    )

    # Reduced displacement
    axs[1].scatter(
        dr_times,
        dr,
        s=1,
        color="k",
        marker=".",
        linewidth=0.1,
        rasterized=True,
    )
    axs[1].set_ylim([0, 8])
    axs[1].set_yticks([0, 4, 8])
    axs[1].set_ylabel("DR [cm$^2$]")
    axs[1].grid(visible=True, alpha=0.4)

    # VOISS-Net classes
    rgb_values = np.array([CLASS_DICT[i][1] for i in range(nclasses)])
    cmap = ListedColormap(rgb_values / 255)
    axs[2].imshow(
        matrix_plot,
        cmap=cmap,
        aspect="auto",
        interpolation="none",
        extent=[*xlim, 0, nclasses],
        alpha=matrix_alpha,
        origin="lower",
        rasterized=True,
    )
    axs[2].set_yticks(np.arange(0.5, nclasses, 1))
    axs[2].set_yticklabels(
        reversed([CLASS_DICT[i][0] for i in range(nclasses)])
    )
    axs[2].set_ylabel("VOISS-Net\nClass")

    # Network-coincident detections in the regular and denoised data
    axs[3].step(
        detection_counts["filtered"].index,
        detection_counts["filtered"].values,
        where="post",
        color="0.25",
        linewidth=0.7,
        label="Regular (filtered)",
    )
    axs[3].step(
        detection_counts["denoised"].index,
        detection_counts["denoised"].values,
        where="post",
        color="tab:blue",
        linewidth=0.7,
        label="Denoised",
    )
    if all(counts.notna().sum() == 0 for counts in detection_counts.values()):
        axs[3].text(
            0.5,
            0.5,
            "No detection output found",
            ha="center",
            va="center",
            transform=axs[3].transAxes,
        )
    else:
        axs[3].legend(loc="upper left", fontsize=FS)
    axs[3].set_ylim(bottom=0)
    axs[3].set_ylabel("LP Detections\nh$^{-1}$")
    axs[3].grid(visible=True, alpha=0.4)

    for i, ax in enumerate(axs):
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.text(
            -0.1,
            0.95,
            f"{chr(97 + i)})",
            transform=ax.transAxes,
            fontsize=FS + 2,
            fontweight="bold",
        )
        if i < len(axs) - 1:
            ax.tick_params(axis="x", labelbottom=False)

    axs[3].set_xticklabels(xticklabels)
    for label in axs[3].get_xticklabels():
        label.set_rotation(30)
        label.set_rotation_mode("anchor")
        label.set_ha("right")
    axs[3].set_xlabel("Date")

    fig.align_ylabels(axs)
    fig.subplots_adjust(hspace=0.12)

    if SAVE:
        file_label = (
            f"shish_binned_denoising_timeline_"
            f"{STARTTIME.strftime('%Y%m%d')}_"
            f"{(STOPTIME - 1).strftime('%Y%m%d')}_"
            f"{BINNING_INTERVAL}s_PNORM-{PNORM_THRESH}"
        )
        output_path = f'{SV_DIR}/{file_label}.png'
        fig.savefig(
            output_path,
            bbox_inches="tight",
            transparent=TRANSPARENT,
            dpi=300,
        )
        print(f"Saved {output_path}")

    if SHOW:
        plt.show(block=True)
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
