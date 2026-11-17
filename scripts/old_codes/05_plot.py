# %% Imports
import logging
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime

from _common import get_cfg
from shishaldin import store
from shishaldin.detection import triggers_to_intervals, intervals_overlap


# %% Settings
LOGGER = logging.getLogger(__name__)
PIPELINE = "denoise_then_filter"


# %% Configuration helpers
def get_stations(cfg):
    stations = list(cfg.data.stations)
    primary_station = cfg.data.primary_station

    if not stations:
        raise ValueError("No stations found in cfg.data.stations.")

    if primary_station not in stations:
        raise ValueError(
            f"Primary station {primary_station!r} is not in cfg.data.stations."
        )

    return stations, primary_station


def find_week_directory(week_value, weekly_dirs):
    """Match a catalog week value to its weekly directory."""
    week_string = str(week_value)

    # Exact directory-name match.
    for week_dir in weekly_dirs:
        if week_dir.name == week_string:
            return week_dir

    # Match numeric values such as 1 to directories such as week_0001.
    match = re.search(r"(\d+)", week_string)
    if match:
        week_number = int(match.group(1))

        for week_dir in weekly_dirs:
            dir_match = re.search(r"(\d+)$", week_dir.name)
            if dir_match and int(dir_match.group(1)) == week_number:
                return week_dir

    return None


# %% Data helpers
def parse_mixed_timestamps(values):
    return pd.to_datetime(values, errors="coerce", utc=True)


def numeric_values(values):
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def load_weekly_detections(week_dir):
    path = week_dir / "detections.npz"

    if not path.exists():
        LOGGER.warning("Missing detections file: %s", path)
        return None

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_station_trace(week_dir, kind, station):
    path = week_dir / "traces" / f"{kind}.{station}.mseed"

    if not path.exists():
        LOGGER.warning("Missing waveform: %s", path)
        return None

    stream = read(str(path))

    if not stream:
        LOGGER.warning("Empty waveform: %s", path)
        return None

    return stream[0]


# %% Station-specific detection intervals
def get_station_detection_intervals(
    detections,
    week_dir,
    kind,
    station,
    primary_station,
):
    """
    Get UTC detection intervals for one station and processing kind.

    The primary station uses the primary trigger arrays directly.
    Secondary stations use only triggers whose coincidence flag is True.
    """
    if station == primary_station:
        trigger_key = f"triggers_{kind}_primary"
        flags_key = None
    else:
        trigger_key = f"triggers_{kind}_{station}"
        flags_key = f"coincident_flags_{kind}_{station}"

    if trigger_key not in detections:
        return []

    triggers = np.asarray(detections[trigger_key])

    if flags_key is not None:
        if flags_key not in detections:
            LOGGER.warning(
                "Missing %s in %s; skipping %s detections.",
                flags_key,
                week_dir.name,
                station,
            )
            return []

        flags = np.asarray(detections[flags_key], dtype=bool)

        if len(flags) != len(triggers):
            raise ValueError(
                f"{flags_key} has {len(flags)} entries, but "
                f"{trigger_key} has {len(triggers)} triggers."
            )

        triggers = triggers[flags]

    if len(triggers) == 0:
        return []

    trace = load_station_trace(week_dir, kind, station)

    if trace is None:
        return []

    return triggers_to_intervals(trace, triggers)


def event_overlaps_detection(event_start, event_end, intervals):
    """Return True if an event overlaps any of the station's detections."""
    for interval_start, interval_end in intervals:
        if intervals_overlap(
            event_start,
            event_end,
            interval_start,
            interval_end,
        ):
            return True

    return False


def add_station_detection_flags(
    catalog,
    weekly_dirs,
    stations,
    primary_station,
):
    """
    Add a station_detected column to the catalog.

    A station is considered detected if it has an overlapping filtered
    OR denoised detection. Secondary-station detections must be coincident.
    """
    catalog = catalog.copy()
    catalog["station_detected"] = False

    if "week" not in catalog.columns:
        raise ValueError("Catalog is missing the 'week' column.")

    if "event_end_time" not in catalog.columns:
        raise ValueError("Catalog is missing the 'event_end_time' column.")

    for week_value, week_rows in catalog.groupby("week", dropna=False):
        if pd.isna(week_value):
            continue

        week_dir = find_week_directory(week_value, weekly_dirs)

        if week_dir is None:
            LOGGER.warning(
                "Could not find weekly directory for catalog week %r.",
                week_value,
            )
            continue

        detections = load_weekly_detections(week_dir)

        if detections is None:
            continue

        # Build station-specific intervals from both processing methods.
        intervals_by_station = {}

        for station in stations:
            station_intervals = []

            for kind in ("filtered", "denoised"):
                intervals = get_station_detection_intervals(
                    detections=detections,
                    week_dir=week_dir,
                    kind=kind,
                    station=station,
                    primary_station=primary_station,
                )

                station_intervals.extend(intervals)

            intervals_by_station[station] = station_intervals

        # Match each catalog row to detections from its own station.
        for row_index, row in week_rows.iterrows():
            station = row["station"]

            if station not in intervals_by_station:
                continue

            if pd.isna(row["event_start_time"]) or pd.isna(row["event_end_time"]):
                continue

            event_start = UTCDateTime(str(row["event_start_time"]))
            event_end = UTCDateTime(str(row["event_end_time"]))

            detected = event_overlaps_detection(
                event_start,
                event_end,
                intervals_by_station[station],
            )

            catalog.at[row_index, "station_detected"] = detected

    return catalog


# %% Histogram plots
def plot_station_histograms(
    catalog,
    stations,
    column,
    metric_name,
    x_label,
    output_dir,
):
    fig, axes = plt.subplots(
        1,
        len(stations),
        figsize=(5 * len(stations), 5),
        sharey=True,
        squeeze=False,
    )

    for ax, station in zip(axes[0], stations):
        station_data = catalog[
            (catalog["station"] == station)
            & catalog["station_detected"]
        ]

        values = numeric_values(station_data[column])
        values = values[np.isfinite(values)]

        if len(values):
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0

            ax.hist(
                values,
                bins=30,
                alpha=0.75,
                edgecolor="black",
            )

            ax.axvline(
                mean,
                linestyle="-.",
                linewidth=2,
                label=f"Mean: {mean:.2f}",
            )

            ax.axvspan(
                mean - std,
                mean + std,
                alpha=0.20,
                label=f"±1 SD: {std:.2f}",
            )

            ax.legend(fontsize=8)

        else:
            ax.text(
                0.5,
                0.5,
                "No finite values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.axvline(0, linestyle="--", color="black", linewidth=1)
        ax.set_title(f"{station}\nn = {len(values)}")
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.25)

    axes[0, 0].set_ylabel("Number of events")

    fig.suptitle(f"{metric_name} Distribution by Station")
    fig.tight_layout()

    fig.savefig(
        output_dir / f"{column}_station_histograms.png",
        dpi=150,
    )

    plt.close(fig)


def plot_sdr_histograms(catalog, stations, output_dir):
    data = catalog[
        catalog["station_detected"]
        & (catalog["detection_status"] == "both")
    ].copy()

    data["sdr_db"] = pd.to_numeric(data["sdr_db"], errors="coerce")
    data = data[np.isfinite(data["sdr_db"])]

    fig, axes = plt.subplots(
        1,
        len(stations),
        figsize=(5 * len(stations), 5),
        sharey=True,
        squeeze=False,
    )

    for ax, station in zip(axes[0], stations):
        values = data.loc[
            data["station"] == station,
            "sdr_db",
        ].to_numpy(dtype=float)

        if len(values):
            mean_sdr = values.mean()

            ax.hist(
                values,
                bins=20,
                color="tab:purple",
                edgecolor="white",
                alpha=0.85,
            )

            ax.axvline(
                mean_sdr,
                color="black",
                linewidth=1.5,
                label=f"Mean = {mean_sdr:.2f} dB",
            )

            ax.legend()

        else:
            ax.text(
                0.5,
                0.5,
                "No finite SDR values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title(f"{station}\nn = {len(values)}")
        ax.set_xlabel("SDR (dB), filtered as reference")
        ax.grid(alpha=0.25)

    axes[0, 0].set_ylabel("Number of events")

    fig.suptitle("Signal-to-distortion ratio by station")
    fig.tight_layout()

    fig.savefig(
        output_dir / "sdr_station_histograms.png",
        dpi=150,
    )

    plt.close(fig)


def plot_cc_histograms(catalog, stations, output_dir):
    data = catalog[
        catalog["station_detected"]
        & (catalog["detection_status"] == "both")
    ].copy()

    data["max_correlation"] = pd.to_numeric(
        data["max_correlation"],
        errors="coerce",
    )

    data = data[np.isfinite(data["max_correlation"])]

    fig, axes = plt.subplots(
        1,
        len(stations),
        figsize=(5 * len(stations), 5),
        sharey=True,
        squeeze=False,
    )

    for ax, station in zip(axes[0], stations):
        values = data.loc[
            data["station"] == station,
            "max_correlation",
        ].to_numpy(dtype=float)

        if len(values):
            ax.hist(
                values,
                bins=20,
                edgecolor="white",
                alpha=0.85,
            )

            ax.axvline(
                0.8,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="Peak correlation = 0.8",
            )

            ax.legend()

        else:
            ax.text(
                0.5,
                0.5,
                "No finite CC values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title(f"{station}\nn = {len(values)}")
        ax.set_xlabel("Peak CC")
        ax.grid(alpha=0.25)

    axes[0, 0].set_ylabel("Number of events")

    fig.suptitle("Peak cross-correlation by station")
    fig.tight_layout()

    fig.savefig(
        output_dir / "cc_station_histograms.png",
        dpi=150,
    )

    plt.close(fig)


# %% ΔSNR over time
def plot_delta_snr_over_time(catalog, stations, output_path):
    data = catalog[
        catalog["station_detected"]
    ][
        ["event_start_time", "station", "delta_snr_db"]
    ].copy()

    data["delta_snr_db"] = pd.to_numeric(
        data["delta_snr_db"],
        errors="coerce",
    )

    data["event_start_time"] = parse_mixed_timestamps(
        data["event_start_time"]
    )

    data = data.dropna(
        subset=["event_start_time", "delta_snr_db", "station"]
    )

    data = data[np.isfinite(data["delta_snr_db"])]
    data = data.sort_values("event_start_time")

    fig, axes = plt.subplots(
        1,
        len(stations),
        figsize=(5.3 * len(stations), 5),
        sharey=True,
        squeeze=False,
    )

    for ax, station in zip(axes[0], stations):
        station_data = data[data["station"] == station]

        if len(station_data):
            ax.scatter(
                station_data["event_start_time"],
                station_data["delta_snr_db"],
                s=15,
                alpha=0.65,
                c=np.where(
                    station_data["delta_snr_db"] < 0,
                    "red",
                    "tab:blue",
                ),
            )

        else:
            ax.text(
                0.5,
                0.5,
                "No finite values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.axhline(
            0,
            linestyle="--",
            color="black",
            linewidth=1,
        )

        ax.set_title(f"{station}\nn = {len(station_data)}")
        ax.set_xlabel("Event time")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=45)

    axes[0, 0].set_ylabel("ΔSNR (dB)")

    fig.suptitle("SNR Difference Through Time by Station")
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# %% Main
def main():
    cfg, _args = get_cfg(
        "Plot station-specific filtered-versus-denoised metrics"
    )

    stations, primary_station = get_stations(cfg)

    comparison_dir = store.comparison_dir(cfg)
    catalog_path = comparison_dir / "catalog.csv"

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Comparison catalog not found: {catalog_path}\n"
            "Run the comparison catalog script first."
        )

    catalog = pd.read_csv(catalog_path, low_memory=False)

    if catalog.empty:
        raise RuntimeError(f"Comparison catalog is empty: {catalog_path}")

    required = [
        "event_number",
        "week",
        "station",
        "event_start_time",
        "event_end_time",
        "detection_status",
        "filtered_snr_db",
        "denoised_snr_db",
        "sdr_db",
        "max_correlation",
    ]

    missing = [
        column for column in required
        if column not in catalog.columns
    ]

    if missing:
        raise ValueError(
            f"Catalog is missing required columns: {missing}"
        )

    # Calculate denoised-minus-filtered SNR difference.
    catalog["filtered_snr_db"] = pd.to_numeric(
        catalog["filtered_snr_db"],
        errors="coerce",
    )

    catalog["denoised_snr_db"] = pd.to_numeric(
        catalog["denoised_snr_db"],
        errors="coerce",
    )

    catalog["delta_snr_db"] = (
        catalog["denoised_snr_db"]
        - catalog["filtered_snr_db"]
    )

    # Locate weekly detection files.
    stage_dir = store.stage_dir(cfg, PIPELINE)
    weekly_dirs = sorted((stage_dir / "weekly").glob("week_*"))

    if not weekly_dirs:
        raise FileNotFoundError(
            f"No weekly directories found in {stage_dir}"
        )

    # Determine whether each station actually detected each event.
    catalog = add_station_detection_flags(
        catalog=catalog,
        weekly_dirs=weekly_dirs,
        stations=stations,
        primary_station=primary_station,
    )

    figure_dir = comparison_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # 1. ΔSNR histogram by station.
    plot_station_histograms(
        catalog,
        stations,
        "delta_snr_db",
        "SNR Difference",
        "ΔSNR (denoised − filtered, dB)",
        figure_dir,
    )

    # 2. SDR histogram by station.
    plot_sdr_histograms(
        catalog,
        stations,
        figure_dir,
    )

    # 3. CC histogram by station.
    plot_cc_histograms(
        catalog,
        stations,
        figure_dir,
    )

    # 4. ΔSNR over time by station.
    plot_delta_snr_over_time(
        catalog,
        stations,
        figure_dir / "snr_delta_over_time.png",
    )

    print("\nComparison plots complete.")
    print(f"Catalog rows: {len(catalog)}")
    print(f"Unique events: {catalog['event_number'].nunique()}")
    print(
        "Rows with station-specific detections: "
        f"{int(catalog['station_detected'].sum())}"
    )
    print(f"Stations: {', '.join(stations)}")
    print(f"Figures saved in: {figure_dir}")


if __name__ == "__main__":
    main()