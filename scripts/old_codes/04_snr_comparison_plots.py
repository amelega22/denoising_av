
# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import get_cfg
from shishaldin import store


# %% Settings
STATIONS = ["SSLS", "SSLN", "SSBA"]


# %% Data helpers
def parse_mixed_timestamps(values):
    return pd.to_datetime(values, errors="coerce", utc=True)


def numeric_values(values):
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


# %% Histogram plots
def plot_station_histograms(catalog, column, metric_name, x_label, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, station in zip(axes, STATIONS):
        values = numeric_values(catalog.loc[catalog["station"] == station, column])
        values = values[np.isfinite(values)]

        if len(values):
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0

            ax.hist(values, bins=30, alpha=0.75, edgecolor="black")
            ax.axvline(mean, linestyle="-.", linewidth=2,
                       label=f"Mean: {mean:.2f}")
            ax.axvspan(mean - std, mean + std, alpha=0.20,
                       label=f"±1 SD: {std:.2f}")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center",
                    transform=ax.transAxes)

        ax.axvline(0, linestyle="--", color="black", linewidth=1)
        ax.set_title(station)
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Number of events")
    fig.suptitle(f"{metric_name} Distribution by Station")
    fig.tight_layout()
    fig.savefig(output_dir / f"{column}_station_histograms.png", dpi=150)
    plt.close(fig)


def plot_sdr_histograms(catalog, output_dir):
    data = catalog.copy()
    data["sdr_db"] = pd.to_numeric(data["sdr_db"], errors="coerce")
    data = data[data["detection_status"] == "both"]
    data = data[np.isfinite(data["sdr_db"])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, station in zip(axes, STATIONS):
        station_data = data[data["station"] == station]
        values = station_data["sdr_db"].to_numpy(dtype=float)

        if len(values):
            mean_sdr = values.mean()
            ax.hist(values, bins=20, color="tab:purple",
                    edgecolor="white", alpha=0.85)
            ax.axvline(mean_sdr, color="black", linewidth=1.5,
                       label=f"Mean = {mean_sdr:.2f} dB")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No finite SDR values", ha="center",
                    va="center", transform=ax.transAxes)

        ax.set_title(f"{station}\nn = {len(values)} matched events")
        ax.set_xlabel("SDR (dB), filtered as reference")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Number of events")
    fig.suptitle("Signal-to-distortion ratio by station")
    fig.tight_layout()
    fig.savefig(output_dir / "sdr_station_histograms.png", dpi=150)
    plt.close(fig)


def plot_cc_histograms(catalog, output_dir):
    data = catalog.copy()
    data["max_correlation"] = pd.to_numeric(
        data["max_correlation"], errors="coerce"
    )
    data = data[data["detection_status"] == "both"]
    data = data[np.isfinite(data["max_correlation"])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, station in zip(axes, STATIONS):
        station_data = data[data["station"] == station]
        values = station_data["max_correlation"].to_numpy(dtype=float)

        if len(values):
            ax.hist(values, bins=20, edgecolor="white", alpha=0.85)
            ax.axvline(
                0.8, color="red", linestyle="--", linewidth=1.5,
                label="Peak correlation = 0.8"
            )
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No finite CC values", ha="center",
                    va="center", transform=ax.transAxes)

        ax.set_title(f"{station}\nn = {len(values)} matched events")
        ax.set_xlabel("Peak CC")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Number of events")
    fig.suptitle("Peak cross-correlation by station")
    fig.tight_layout()
    fig.savefig(output_dir / "cc_station_histograms.png", dpi=150)
    plt.close(fig)


# %% Over-time plots
def plot_metric_over_time(catalog, column, title, y_label, output_path):
    data = catalog[["event_start_time", "station", column]].copy()
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data["event_start_time"] = parse_mixed_timestamps(data["event_start_time"])
    data = data.dropna(subset=["event_start_time", column, "station"])
    data = data[np.isfinite(data[column])]
    data = data.sort_values("event_start_time")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, station in zip(axes, STATIONS):
        station_data = data[data["station"] == station]

        if len(station_data):
            ax.scatter(
                station_data["event_start_time"],
                station_data[column],
                s=15,
                alpha=0.65,
                c=np.where(station_data[column] < 0, "red", "tab:blue")
            )
        else:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center",
                    transform=ax.transAxes)

        ax.axhline(0, linestyle="--", color="black", linewidth=1)
        ax.set_title(station)
        ax.set_xlabel("Event time")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel(y_label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# %% Station comparison plots
def plot_station_metric(catalog, column, title, y_label, output_path):
    data = catalog[["station", column]].copy()
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["station", column])
    data = data[np.isfinite(data[column])]

    fig, ax = plt.subplots(figsize=(9, 5))
    values = [
        data.loc[data["station"] == station, column].to_numpy()
        for station in STATIONS
    ]

    if any(len(value) for value in values):
        ax.boxplot(values, labels=STATIONS)
    else:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center",
                transform=ax.transAxes)

    ax.set(title=title, xlabel="Station", ylabel=y_label)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# %% SNR improvement versus SDR
def plot_sdr_vs_snr(catalog, output_dir):
    data = catalog.copy()
    data["sdr_db"] = pd.to_numeric(data["sdr_db"], errors="coerce")
    data["filtered_snr_db"] = pd.to_numeric(
        data["filtered_snr_db"], errors="coerce"
    )
    data["denoised_snr_db"] = pd.to_numeric(
        data["denoised_snr_db"], errors="coerce"
    )
    data["snr_difference_db"] = (
        data["denoised_snr_db"] - data["filtered_snr_db"]
    )

    data = data[data["detection_status"] == "both"]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["sdr_db", "snr_difference_db", "station"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, station in zip(axes, STATIONS):
        station_data = data[data["station"] == station]

        if len(station_data):
            colors = np.where(
                (station_data["sdr_db"] < 0) |
                (station_data["snr_difference_db"] < 0),
                "red", "tab:blue"
            )
            ax.scatter(
                station_data["snr_difference_db"],
                station_data["sdr_db"],
                c=colors,
                alpha=0.75,
                s=20
            )
        else:
            ax.text(0.5, 0.5, "No finite matched events", ha="center",
                    va="center", transform=ax.transAxes)

        ax.axhline(0, linestyle="--", color="black", linewidth=1)
        ax.axvline(0, linestyle="--", color="black", linewidth=1)
        ax.set_title(station)
        ax.set_xlabel("SNR improvement (dB)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("SDR (dB)")
    fig.suptitle("SDR vs. SNR Improvement by Station\nRed = negative SDR or negative SNR improvement")
    fig.tight_layout()
    fig.savefig(output_dir / "sdr_vs_snr_by_station.png", dpi=150)
    plt.close(fig)


# %% Detection counts
def plot_detection_counts(catalog, output_path):
    if "event_number" not in catalog.columns or "detection_status" not in catalog.columns:
        return

    event_data = catalog.drop_duplicates("event_number")
    counts = event_data["detection_status"].value_counts()
    categories = ["both", "filtered_only", "denoised_only"]
    values = [int(counts.get(category, 0)) for category in categories]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Both methods", "Filtered only", "Denoised only"], values)
    ax.set(title="Event detection comparison", ylabel="Number of events")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# %% Main
def main():
    cfg, _args = get_cfg("Plot filtered-versus-denoised comparison results")

    output_dir = store.comparison_dir(cfg)
    catalog_path = output_dir / "catalog.csv"

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Comparison catalog not found: {catalog_path}\n"
            "Run the comparison catalog script first."
        )

    catalog = pd.read_csv(catalog_path, low_memory=False)

    if catalog.empty:
        raise RuntimeError(f"Comparison catalog is empty: {catalog_path}")

    required = [
        "event_number", "station", "event_start_time",
        "detection_status", "filtered_snr_db", "denoised_snr_db"
    ]
    missing = [column for column in required if column not in catalog.columns]

    if missing:
        raise ValueError(f"Catalog is missing required columns: {missing}")

    catalog["filtered_snr_db"] = pd.to_numeric(
        catalog["filtered_snr_db"], errors="coerce"
    )
    catalog["denoised_snr_db"] = pd.to_numeric(
        catalog["denoised_snr_db"], errors="coerce"
    )
    catalog["delta_snr_db"] = (
        catalog["denoised_snr_db"] - catalog["filtered_snr_db"]
    )

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Detection counts.
    # plot_detection_counts(catalog, figure_dir / "detection_counts.png")

    # SNR improvement.
    plot_station_histograms(
        catalog, "delta_snr_db", "SNR Difference",
        "ΔSNR (denoised − filtered, dB)", figure_dir
    )
    plot_metric_over_time(
        catalog, "delta_snr_db", "SNR Difference Through Time",
        "ΔSNR (dB)", figure_dir / "snr_delta_over_time.png"
    )
    # plot_station_metric(
    #     catalog, "delta_snr_db", "SNR Difference by Station",
    #     "ΔSNR (dB)", figure_dir / "snr_delta_by_station.png"
    # )

    # SDR and CC: matched events with finite values only.
    if "sdr_db" in catalog.columns:
        plot_sdr_histograms(catalog, figure_dir)
        # plot_metric_over_time(
        #     catalog, "sdr_db", "SDR Through Time",
        #     "SDR (dB)", figure_dir / "sdr_over_time.png"
        # )
        # plot_station_metric(
        #     catalog, "sdr_db", "SDR by Station",
        #     "SDR (dB)", figure_dir / "sdr_by_station.png"
        # )
        plot_sdr_vs_snr(catalog, figure_dir)

    if "max_correlation" in catalog.columns:
        plot_cc_histograms(catalog, figure_dir)
        # plot_metric_over_time(
        #     catalog, "max_correlation",
        #     "Peak Cross-Correlation Through Time",
        #     "Peak CC", figure_dir / "cc_over_time.png"
        # )
        # plot_station_metric(
        #     catalog, "max_correlation", "Cross-Correlation by Station",
        #     "Peak CC", figure_dir / "cc_by_station.png"
        # )

    print("\nComparison plots complete.")
    print(f"Catalog rows: {len(catalog)}")
    print(f"Unique events: {catalog['event_number'].nunique()}")
    print(f"Figures saved in: {figure_dir}")


if __name__ == "__main__":
    main()
