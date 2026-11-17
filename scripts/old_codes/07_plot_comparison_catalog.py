
# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import get_cfg
from shishaldin import store


# %% Plotting functions
def parse_mixed_timestamps(values):
    # Parse individually to support timestamps with and without fractions.
    return pd.to_datetime(
        values.astype(str).map(
            lambda value: pd.to_datetime(value, errors="coerce")
        ),
        errors="coerce"
    )


def plot_histogram(values, title, x_label, output_path):
    values = pd.to_numeric(
        pd.Series(values), errors="coerce"
    ).to_numpy(dtype=float)

    values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(values):
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0

        ax.hist(
            values,
            bins=30,
            alpha=0.75,
            edgecolor="black"
        )

        ax.axvspan(
            mean - std,
            mean + std,
            alpha=0.20,
            label=f"Mean ± 1 SD ({mean:.2f} ± {std:.2f})"
        )

        ax.axvline(
            mean,
            linestyle="-.",
            linewidth=2,
            label=f"Mean: {mean:.2f}"
        )

        ax.legend(fontsize=9)

    else:
        ax.text(
            0.5, 0.5, "No finite values",
            ha="center",
            va="center",
            transform=ax.transAxes
        )

    ax.axvline(
        0,
        linestyle="--",
        color="black",
        linewidth=1.2
    )

    ax.set(
        title=title,
        xlabel=x_label,
        ylabel="Denoised events"
    )

    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_over_time(
    catalog, column, title, y_label, output_path
):
    data = catalog[["start_time", column]].copy()

    data[column] = pd.to_numeric(
        data[column], errors="coerce"
    )
    data["start_time"] = parse_mixed_timestamps(
        data["start_time"]
    )

    data = data.dropna(
        subset=["start_time", column]
    )
    data = data[np.isfinite(data[column])]
    data = data.sort_values("start_time")

    fig, ax = plt.subplots(figsize=(12, 5))

    if len(data):
        ax.scatter(
            data["start_time"],
            data[column],
            s=15,
            alpha=0.55
        )
    else:
        ax.text(
            0.5, 0.5, "No finite values",
            ha="center",
            va="center",
            transform=ax.transAxes
        )

    ax.set(
        title=title,
        xlabel="Event time",
        ylabel=y_label
    )

    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# %% Main
def main():
    cfg, _args = get_cfg(__doc__)

    output_dir = store.comparison_dir(cfg)
    catalog_path = output_dir / "catalog.csv"

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Comparison catalog not found: {catalog_path}\n"
            "Run 06_build_comparison_catalog.py first."
        )

    catalog = pd.read_csv(catalog_path)

    if catalog.empty:
        raise RuntimeError(
            f"Comparison catalog is empty: {catalog_path}"
        )

    required_columns = [
        "start_time",
        "delta_snr_db",
        "sdr_db",
        "max_correlation"
    ]

    missing = [
        column for column in required_columns
        if column not in catalog.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required catalog columns: {missing}"
        )

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Histograms
    histograms = [
        (
            "delta_snr_db",
            "SNR difference: denoised vs filtered",
            "ΔSNR (denoised − filtered, dB)",
            "snr_delta_histogram.png"
        ),
        (
            "sdr_db",
            "SDR: denoised relative to bandpass filtered",
            "SDR (dB)",
            "sdr_histogram.png"
        ),
        (
            "max_correlation",
            "Peak CC: denoised vs bandpass filtered",
            "Peak cross-correlation (-1 to +1)",
            "cc_histogram.png"
        )
    ]

    for column, title, label, filename in histograms:
        plot_histogram(
            catalog[column],
            title,
            label,
            figure_dir / filename
        )

    # Metrics through time
    plot_metric_over_time(
        catalog,
        "sdr_db",
        "SDR through the comparison period",
        "SDR (dB)",
        figure_dir / "sdr_over_time.png"
    )

    plot_metric_over_time(
        catalog,
        "max_correlation",
        "Peak CC through the comparison period",
        "Peak cross-correlation (-1 to +1)",
        figure_dir / "cc_over_time.png"
    )

    print("\nComparison plots complete.")
    print(f"Events in catalog: {len(catalog)}")
    print(f"Figures saved in: {figure_dir}")


if __name__ == "__main__":
    main()
