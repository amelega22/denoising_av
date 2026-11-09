"""compare SNR between the two denoised datasets(filter -> denoise  vs  denoise -> filter)

needs the catalogs from both pipelines

run with: --python scripts/05_snr_plots.py

output:  outputs/<run_id>/comparison/:
- snr_summary.csv =  mean / median / std of per-event SNR per pipeline and trace type
- snr_paired.csv = denoised events detected in BOTH pipelines, with SNR in each
- figures/snr_distributions.png = SNR histograms (denoised) + boxplots (filtered & denoised)
- figures/snr_whole_period.png = whole-period SNR bars
- figures/snr_paired.png = matched-event scatter vs 1:1 and SNR-difference histogram
"""
import matplotlib.pyplot as plt

from _common import get_cfg

from shishaldin import plotting as P
from shishaldin import snr, store


def main():
    cfg, _ = get_cfg(__doc__)
    a, b = store.PIPELINES
    labels = snr.pipeline_labels(cfg)

    catalogs = snr.load_catalogs(cfg)
    whole = snr.load_whole_period_snr(cfg)
    out = store.comparison_dir(cfg)

    summary = snr.summarize(catalogs)
    summary.to_csv(out / "snr_summary.csv", index=False)
    print("PER-EVENT SNR (dB)")
    print(summary.round(2).to_string(index=False))

    print("\nWHOLE-PERIOD SNR (dB)")
    for pipe in (a, b):
        print(f"  {labels[pipe]}: filtered {whole[pipe]['filtered']['snr_db']:.2f}, "
              f"denoised {whole[pipe]['denoised']['snr_db']:.2f}")

    paired, a_only, b_only = snr.match_denoised(catalogs[a], catalogs[b])
    paired.to_csv(out / "snr_paired.csv", index=False)
    d = paired["diff_db"].dropna()
    print(f"\nDENOISED EVENTS DETECTED IN BOTH: {len(paired)}  "
          f"({a_only} only in {a}, {b_only} only in {b})")
    if len(d):
        print(f"  SNR difference (denoise->filter minus filter->denoise): "
              f"median {d.median():+.2f} dB, mean {d.mean():+.2f} dB, "
              f"higher in denoise->filter for {(d > 0).mean() * 100:.0f}% of events")

    figs = {
        "snr_distributions": P.plot_snr_distributions(catalogs, labels),
        "snr_whole_period": P.plot_whole_period_snr(whole, labels),
        "snr_paired": P.plot_snr_paired(paired, labels[a], labels[b], a_only, b_only),
    }
    for name, fig in figs.items():
        fig.savefig(out / "figures" / f"{name}.png", dpi=150)
        plt.close(fig)
    print(f"\nsaved comparison to {out}")


if __name__ == "__main__":
    main()
