# %% Imports
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from _common import get_cfg
from shishaldin import store

PIPELINES = ("denoise_then_filter", "filter_then_denoise")
STATIONS = ["SSLS", "SSLN", "SSBA"]

def parse_mixed_timestamps(values):
    return pd.to_datetime(values, errors="coerce", utc=True)

def numeric_values(values):
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

def with_label(title, label):
    return f"{title} ({label})" if label else title


# histograms
def plot_station_histograms(catalog, column, metric_name, x_label, output_dir,label=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, station in zip(axes, STATIONS):
        values = numeric_values(catalog.loc[catalog["station"] == station, column])
        values = values[np.isfinite(values)]

        if len(values):
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ax.hist(values, bins=30, alpha=0.75, edgecolor="black")
            ax.axvline(mean, linestyle="-.", linewidth=2,label=f"Mean: {mean:.2f}")
            ax.axvspan(mean - std, mean + std, alpha=0.20,label=f"±1 SD: {std:.2f}")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center",transform=ax.transAxes)
        ax.axvline(0, linestyle="--", color="black", linewidth=1)
        ax.set_title(station)
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Number of events")
    fig.suptitle(with_label(f"{metric_name} Distribution by Station", label))
    fig.tight_layout()
    fig.savefig(output_dir / f"{column}_station_histograms.png", dpi=150)
    plt.close(fig)

def plot_sdr_histograms(catalog, output_dir, label=""):
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
            ax.hist(values, bins=20, color="tab:purple",edgecolor="white", alpha=0.85)
            ax.axvline(mean_sdr, color="black", linewidth=1.5, label=f"Mean = {mean_sdr:.2f} dB")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No finite SDR values", ha="center",va="center", transform=ax.transAxes)
        ax.set_title(f"{station}\nn = {len(values)} matched events")
        ax.set_xlabel("SDR (dB), filtered as reference")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Number of events")
    fig.suptitle(with_label("Signal-to-distortion ratio by station", label))
    fig.tight_layout()
    fig.savefig(output_dir / "sdr_station_histograms.png", dpi=150)
    plt.close(fig)

def plot_cc_histograms(catalog, output_dir, label=""):
    data = catalog.copy()
    data["max_correlation"] = pd.to_numeric(data["max_correlation"], errors="coerce")
    data = data[data["detection_status"] == "both"]
    data = data[np.isfinite(data["max_correlation"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, station in zip(axes, STATIONS):
        station_data = data[data["station"] == station]
        values = station_data["max_correlation"].to_numpy(dtype=float)
        if len(values):
            ax.hist(values, bins=20, edgecolor="white", alpha=0.85)
            ax.axvline(0.8, color="red", linestyle="--", linewidth=1.5,label="Peak correlation = 0.8")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No finite CC values", ha="center",va="center", transform=ax.transAxes)
        ax.set_title(f"{station}\nn = {len(values)} matched events")
        ax.set_xlabel("Peak CC")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Number of events")
    fig.suptitle(with_label("Peak cross-correlation by station", label))
    fig.tight_layout()
    fig.savefig(output_dir / "cc_station_histograms.png", dpi=150)
    plt.close(fig)

# plots over tiem
def plot_metric_over_time(catalog, column, title, y_label, output_path,label=""):
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
            ax.scatter(station_data["event_start_time"],station_data[column],s=15,alpha=0.65,c=np.where(station_data[column] < 0, "red", "tab:blue"))
        else:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center",transform=ax.transAxes)
        ax.axhline(0, linestyle="--", color="black", linewidth=1)
        ax.set_title(station)
        ax.set_xlabel("Event time")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel(y_label)
    fig.suptitle(with_label(title, label))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

# plots to compare stations
def plot_station_metric(catalog, column, title, y_label, output_path, label=""):
    data = catalog[["station", column]].copy()
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["station", column])
    data = data[np.isfinite(data[column])]
    fig, ax = plt.subplots(figsize=(9, 5))
    values = [data.loc[data["station"] == station, column].to_numpy() for station in STATIONS]

    if any(len(value) for value in values):
        try:
            ax.boxplot(values, tick_labels=STATIONS)
        except TypeError:
            # matplotlib older than 3.9 calls this argument "labels"
            ax.boxplot(values, labels=STATIONS)
    else:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center",transform=ax.transAxes)
    ax.set(title=with_label(title, label), xlabel="Station", ylabel=y_label)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

# snr improvement versus sdr
def plot_sdr_vs_snr(catalog, output_dir, label=""):
    data = catalog.copy()
    data["sdr_db"] = pd.to_numeric(data["sdr_db"], errors="coerce")
    data["filtered_snr_db"] = pd.to_numeric(data["filtered_snr_db"], errors="coerce")
    data["denoised_snr_db"] = pd.to_numeric(data["denoised_snr_db"], errors="coerce")
    data["snr_difference_db"] = (data["denoised_snr_db"] - data["filtered_snr_db"])
    data = data[data["detection_status"] == "both"]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["sdr_db", "snr_difference_db", "station"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, station in zip(axes, STATIONS):
        station_data = data[data["station"] == station]
        if len(station_data):
            colors = np.where((station_data["sdr_db"] < 0) | (station_data["snr_difference_db"] < 0), "red", "tab:blue")
            ax.scatter(station_data["snr_difference_db"],station_data["sdr_db"],c=colors,alpha=0.75,s=20)
        else:
            ax.text(0.5, 0.5, "No finite matched events", ha="center",va="center", transform=ax.transAxes)

        ax.axhline(0, linestyle="--", color="black", linewidth=1)
        ax.axvline(0, linestyle="--", color="black", linewidth=1)
        ax.set_title(station)
        ax.set_xlabel("SNR improvement (dB)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("SDR (dB)")
    fig.suptitle(with_label("SDR vs. SNR Improvement by Station", label) + "\nRed = negative SDR or negative SNR improvement" )
    fig.tight_layout()
    fig.savefig(output_dir / "sdr_vs_snr_by_station.png", dpi=150)
    plt.close(fig)

# snr and attentiation plot
def plot_signal_vs_noise_change(catalog, output_dir, label=""):
    required = ["signal_rms_change_db", "noise_rms_change_db", "station"]
    missing = [column for column in required if column not in catalog.columns]
    if missing:
        print(f"Skipping signal-vs-noise plot; missing columns: {missing}")
        return

    data = catalog.copy()
    for column in required[:2]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required)

    if "detection_status" in data.columns:
        data = data[data["detection_status"] == "both"]

    fig, ax = plt.subplots(figsize=(8, 7))
    station_colors = {"SSLS": "tab:blue", "SSLN": "tab:orange", "SSBA": "tab:green"}
    for station in STATIONS:
        station_data = data[data["station"] == station]
        if len(station_data):
            ax.scatter(station_data["signal_rms_change_db"],station_data["noise_rms_change_db"],label=f"{station} (n={len(station_data)})",alpha=0.7, s=24, color=station_colors.get(station))

    if len(data):
        low = float(min(data["signal_rms_change_db"].min(),data["noise_rms_change_db"].min()))
        high = float(max(data["signal_rms_change_db"].max(),data["noise_rms_change_db"].max()))
        padding = max((high - low) * 0.05, 1.0)
        limits = (low - padding, high + padding)
        ax.plot(limits, limits, linestyle="--", color="black",linewidth=1.5, label="Equal signal and noise change")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No finite matched-event values",ha="center", va="center", transform=ax.transAxes)

    ax.axhline(0, linestyle=":", color="grey", linewidth=1)
    ax.axvline(0, linestyle=":", color="grey", linewidth=1)
    ax.set(xlabel="Signal RMS change (dB; denoised − filtered)",ylabel="Noise RMS change (dB; denoised − filtered)",title=with_label("Signal attenuation vs. noise attenuation", label))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "signal_vs_noise_rms_change.png", dpi=150)
    plt.close(fig)

def plot_signal_change_vs_snr_change(catalog, output_dir, label=""):
    required = ["signal_rms_change_db", "snr_change_db", "station"]
    missing = [column for column in required if column not in catalog.columns]
    if missing:
        print(f"Skipping signal-change-vs-SNR plot; missing columns: {missing}")
        return

    data = catalog.copy()
    for column in required[:2]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required)

    if "detection_status" in data.columns:
        data = data[data["detection_status"] == "both"]

    fig, ax = plt.subplots(figsize=(8, 7))
    station_colors = {"SSLS": "tab:blue", "SSLN": "tab:orange", "SSBA": "tab:green"}
    for station in STATIONS:
        station_data = data[data["station"] == station]
        if len(station_data):
            ax.scatter(station_data["signal_rms_change_db"],station_data["snr_change_db"],label=f"{station} (n={len(station_data)})",alpha=0.7, s=24, color=station_colors.get(station))

    if len(data):
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No finite matched-event values",ha="center", va="center", transform=ax.transAxes)

    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.axvline(0, linestyle="--", color="black", linewidth=1)
    ax.set(xlabel="Signal RMS change (dB; denoised − filtered)",ylabel="SNR change (dB; denoised − filtered)",title=with_label("Signal attenuation vs. SNR improvement", label))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "signal_change_vs_snr_change.png", dpi=150)
    plt.close(fig)

# detection counts
def plot_detection_counts(manifest_path, output_path, label=""):
    if not manifest_path.exists():
        print(f"Skipping detection counts: no manifest at {manifest_path}")
        return

    with open(manifest_path, "r") as f:
        totals = json.load(f).get("totals", {})

    keys = ("matched_events_in_both", "filtered_only_events",
            "denoised_only_events")

    if any(key not in totals for key in keys):
        print("Skipping detection counts: manifest has no unmatched counts. "
              "Rebuild the catalog.")
        return

    values = [int(totals[key]) for key in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Both methods", "Filtered only", "Denoised only"], values)
    ax.set(title=with_label("Event detection comparison", label),ylabel="Number of events")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

# rms reduction plots
def plot_rms_reduction(catalog, output_dir, label=""):
    required = ["station", "signal_rms_change_db", "noise_rms_change_db"]
    missing = [column for column in required if column not in catalog.columns]
    if missing:
        print(f"Skipping RMS reduction plot; missing columns: {missing}")
        return
    data = catalog.copy()
    data = data[data["detection_status"] == "both"]
    for column in required[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, column, title in ((axes[0], "signal_rms_change_db", "Signal RMS reduction"),(axes[1], "noise_rms_change_db", "Noise RMS reduction")):
        medians = []
        for station in STATIONS:
            values = data.loc[data["station"] == station, column].dropna()
            if len(values):
                # convert the rms change to percentage amplitude reduction
                reduction = (1 - 10 ** (values / 20)) * 100
                medians.append(float(reduction.median()))
            else:
                medians.append(np.nan)

        bars = ax.bar(STATIONS, medians)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Station")
        ax.grid(axis="y", alpha=0.25)

        for bar, value in zip(bars, medians):
            if np.isfinite(value):
                ax.annotate(f"{value:.1f}%",(bar.get_x() + bar.get_width() / 2, value),xytext=(0, 5 if value >= 0 else -15),textcoords="offset points",ha="center")

    axes[0].set_ylabel("Median RMS amplitude reduction (%)")
    fig.suptitle(with_label("Signal and Noise Reduction", label))
    fig.tight_layout()
    fig.savefig(output_dir / "signal_noise_reduction.png", dpi=150)
    plt.close(fig)

# snr improvement plot
def plot_snr_improvement(catalog, output_dir, label=""):
    if "snr_change_db" not in catalog.columns:
        print("Skipping SNR improvement plot: snr_change_db is missing.")
        return

    data = catalog.copy()
    data = data[data["detection_status"] == "both"]
    data["snr_change_db"] = pd.to_numeric(data["snr_change_db"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["snr_change_db", "station"])

    fig, ax = plt.subplots(figsize=(9, 5))
    values = [data.loc[data["station"] == station, "snr_change_db"].to_numpy() for station in STATIONS]

    if any(len(value) for value in values):
        try:
            ax.boxplot(values, tick_labels=STATIONS, showfliers=False)
        except TypeError:
            ax.boxplot(values, labels=STATIONS, showfliers=False)
        for index, station_values in enumerate(values, start=1):
            if len(station_values):
                ax.scatter(np.random.normal(index, 0.045, len(station_values)),station_values,alpha=0.25, s=12)
    else:
        ax.text(0.5, 0.5, "No finite SNR changes", ha="center",va="center", transform=ax.transAxes)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5,label="No change")
    ax.set(title=with_label("SNR Improvement by Station", label),xlabel="Station",ylabel="SNR change (dB)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "snr_improvement_by_station.png", dpi=150)
    plt.close(fig)

# example waveform comparisons
def plot_example_waveforms(catalog, cfg, pipeline, output_dir, max_events=3):
    required = ["event_number", "week", "station","event_start_time", "event_end_time"]
    missing = [column for column in required if column not in catalog.columns]
    if missing:
        print(f"Skipping waveform examples; missing columns: {missing}")
        return

    station = cfg.data.primary_station
    events = catalog[catalog["station"] == station].copy()
    events = events.drop_duplicates(subset=["event_number"])
    events = events.sort_values("event_start_time")

    if "detection_status" in events.columns:
        events = events[events["detection_status"] == "both"]

    if events.empty:
        print(f"Skipping waveform examples: no matched events for {station}.")
        return

    indices = np.linspace(0, len(events) - 1, min(max_events, len(events)), dtype=int)
    examples = events.iloc[indices]

    for _, event in examples.iterrows():
        week = int(str(event["week"]).replace("week_", ""))
        start = UTCDateTime(event["event_start_time"])
        end = UTCDateTime(event["event_end_time"])

        traces = {}
        for kind in ("filtered", "denoised"):
            weekly = store.load_weekly_traces(cfg, pipeline, week, kind)
            if station not in weekly:
                traces = {}
                break
            trace = weekly[station].copy()
            trace.trim(start, end, pad=False)
            if trace.stats.npts == 0:
                traces = {}
                break
            traces[kind] = trace

        if len(traces) != 2:
            print(f"Skipping waveform example {event['event_number']}: ""waveform data unavailable.")
            continue

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for ax, kind, title in ((axes[0], "filtered", "Filtered reference"),(axes[1], "denoised", "Denoised output")):
            trace = traces[kind]
            times = trace.times(reftime=start)
            ax.plot(times, trace.data, linewidth=0.8)
            ax.set_title(title)
            ax.set_ylabel("Amplitude")
            ax.grid(alpha=0.25)

        normalized = {}
        for kind, trace in traces.items():
            values = np.asarray(trace.data, dtype=float)
            peak = np.max(np.abs(values))
            if peak > 0:
                normalized[kind] = values / peak

        for kind, trace in traces.items():
            if kind not in normalized:
                continue
            axes[2].plot(trace.times(reftime=start),normalized[kind],linewidth=0.9,label=kind.capitalize(),alpha=0.8)
        axes[2].set_title("Normalized waveform shape comparison")
        axes[2].set_xlabel("Seconds from event start")
        axes[2].set_ylabel("Normalized amplitude")
        axes[2].legend()
        axes[2].grid(alpha=0.25)

        fig.suptitle(f"{pipeline} | {station} | Event {event['event_number']}")
        fig.tight_layout()
        event_id = int(event["event_number"])
        fig.savefig(output_dir / f"waveform_event_{event_id}.png", dpi=150)
        plt.close(fig)

def load_catalog(catalog_path):
    try:
        return pd.read_csv(catalog_path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def plot_pipeline(cfg, pipeline):
    output_dir = store.comparison_dir(cfg) / pipeline
    catalog_path = output_dir / "catalog.csv"

    if not catalog_path.exists():
        raise FileNotFoundError(f"Comparison catalog not found for {pipeline}: {catalog_path}\n" "Run the comparison catalog script first.")
    catalog = load_catalog(catalog_path)

    if catalog.empty:
        print(f"\n[{pipeline}] catalog is empty (no matched events); "
              "nothing to plot.")
        return

    required = ["event_number", "station", "event_start_time","detection_status", "filtered_snr_db", "denoised_snr_db"]
    missing = [column for column in required if column not in catalog.columns]

    if missing:
        raise ValueError(f"Catalog is missing required columns: {missing}")

    catalog["filtered_snr_db"] = pd.to_numeric(catalog["filtered_snr_db"], errors="coerce")
    catalog["denoised_snr_db"] = pd.to_numeric(catalog["denoised_snr_db"], errors="coerce")
    catalog["delta_snr_db"] = (catalog["denoised_snr_db"] - catalog["filtered_snr_db"])
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # # rms and snr plots
    # plot_rms_reduction(catalog, figure_dir, label=pipeline)
    # plot_snr_improvement(catalog, figure_dir, label=pipeline)

    # # Example waveform comparisons for matched events.
    # plot_example_waveforms(catalog, cfg, pipeline, figure_dir)

    # Detection counts (matched / filtered only / denoised only).
    # plot_detection_counts(
    #     output_dir / "catalog_manifest.json",
    #     figure_dir / "detection_counts.png",
    #     label=pipeline
    # )

    # Signal attenuation versus noise attenuation, and signal attenuation
    # versus SNR improvement.
    # plot_signal_vs_noise_change(catalog, figure_dir, label=pipeline)
    # plot_signal_change_vs_snr_change(catalog, figure_dir, label=pipeline)

    # SNR improvement.
    plot_station_histograms(catalog, "delta_snr_db", "SNR Difference","ΔSNR (denoised − filtered, dB)", figure_dir, label=pipeline)
    plot_metric_over_time(catalog, "delta_snr_db", "SNR Difference Through Time","ΔSNR (dB)", figure_dir / "snr_delta_over_time.png", label=pipeline)
    # plot_station_metric(
    #     catalog, "delta_snr_db", "SNR Difference by Station",
    #     "ΔSNR (dB)", figure_dir / "snr_delta_by_station.png", label=pipeline
    # )

    # SDR and CC: matched events with finite values only.
    if "sdr_db" in catalog.columns:
        plot_sdr_histograms(catalog, figure_dir, label=pipeline)
        # plot_metric_over_time(
        #     catalog, "sdr_db", "SDR Through Time",
        #     "SDR (dB)", figure_dir / "sdr_over_time.png", label=pipeline
        # )
        # plot_station_metric(
        #     catalog, "sdr_db", "SDR by Station",
        #     "SDR (dB)", figure_dir / "sdr_by_station.png", label=pipeline
        # )
        # plot_sdr_vs_snr(catalog, figure_dir, label=pipeline)

    if "max_correlation" in catalog.columns:
        plot_cc_histograms(catalog, figure_dir, label=pipeline)
        # plot_metric_over_time(
        #     catalog, "max_correlation",
        #     "Peak Cross-Correlation Through Time",
        #     "Peak CC", figure_dir / "cc_over_time.png", label=pipeline
        # )
        # plot_station_metric(
        #     catalog, "max_correlation", "Cross-Correlation by Station",
        #     "Peak CC", figure_dir / "cc_by_station.png", label=pipeline
        # )

    print(f"\n[{pipeline}] comparison plots complete.")
    print(f"Catalog rows: {len(catalog)}")
    print(f"Unique events: {catalog['event_number'].nunique()}")
    print(f"Figures saved in: {figure_dir}")

def main():
    cfg, _args = get_cfg("Plot filtered-versus-denoised comparison results for both pipelines")

    for pipeline in PIPELINES:
        plot_pipeline(cfg, pipeline)


if __name__ == "__main__":
    main()
