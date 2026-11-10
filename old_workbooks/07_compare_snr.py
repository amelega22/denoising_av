
"""use waveform traces to compare snr via snr difference histogram

inputs:
- raw and denoised traces from denoise_then_filter
- bndpass-filtered traces from filter_then_denoise
- saved detections from both pipelines

"""

        cfg.data.primary_station
        if cfg.data.primary_station in denoised_traces
        else sorted(denoised_traces)[0]
    )
    bandpass_primary = (
        cfg.data.primary_station
        if cfg.data.primary_station in bandpass_traces
        else sorted(bandpass_traces)[0]
    )

    denoised_intervals = triggers_to_intervals(
        denoised_traces[denoised_primary],
        denoised_detections["triggers_denoised"],
    )
    bandpass_intervals = triggers_to_intervals(
        bandpass_traces[bandpass_primary],
        bandpass_detections["triggers_filtered"],
    )

    matches = match_events(
        bandpass_intervals,
        denoised_intervals,
    )

    print(f"Denoised events: {len(denoised_intervals)}")
    print(f"Bandpass events: {len(bandpass_intervals)}")
    print(f"Matched events: {len(matches)}")
    print(f"Pre-event noise window: {cfg.trigger.lta_seconds} seconds")

    if not matches:
        raise RuntimeError(
            "No overlapping denoised and bandpass events were found."
        )

    catalog = build_comparison_catalog(
        cfg,
        raw_traces,
        bandpass_traces,
        denoised_traces,
        matches,
    )

    output_dir = store.comparison_dir(cfg)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = output_dir / "catalog.csv"
    compact = bool(cfg.output.compact_csv_floats)
    if compact:
        catalog.to_csv(catalog_path, index=False, float_format="%.2g")
    else:
        catalog.to_csv(catalog_path, index=False)

    snr_figure_path = figure_dir / "snr_delta_histogram.png"
    plot_snr_differences(catalog, snr_figure_path)

    sdr_figure_path = figure_dir / "sdr_histograms.png"
    plot_sdr_distributions(catalog, sdr_figure_path)

    print(f"\nSaved comparison catalog: {catalog_path}")
    print(f"Saved SNR histogram: {snr_figure_path}")
    print(f"Saved SDR histograms: {sdr_figure_path}")

    print("\nMedian ΔSNR relative to raw:")
    for kind in (
        "bandpass_filtered",
        "denoised_highpass_filtered",
    ):
        values = catalog.loc[
            catalog["detection_type"] == kind,
            "delta_snr_db",
        ].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            print(f"  {kind}: {np.median(values):.2f} dB")
        else:
            print(f"  {kind}: no finite values")

    print("\nMedian SDR relative to raw:")
    for kind in (
        "bandpass_filtered",
        "denoised_highpass_filtered",
    ):
        values = catalog.loc[
            catalog["detection_type"] == kind,
            "sdr_db",
        ].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            print(f"  {kind}: {np.median(values):.2f} dB")
        else:
            print(f"  {kind}: no finite values")


if __name__ == "__main__":
    main()
