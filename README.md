# Denoising Alaska Volcano Observatory Seismic Data

This repository is used to denoise and analyze Alaska Volcano Observatory
(AVO) volcano seismic data.

## Installation

Install [Mamba](https://mamba.readthedocs.io/), then create the project
environment from `environment.yml`:

```bash
mamba env create -f environment.yml
mamba activate denoising_av
```

If the environment already exists, update it after changes to
`environment.yml`:

```bash
mamba env update -f environment.yml --prune
mamba activate denoising_av
```

Conda can also be used by replacing `mamba` with `conda` in these commands.

## Running the workflow

Update the necessary parameters in `config.yaml`, then validate the
configuration:

```bash
python scripts/validate_config.py
```

Instrument correction defaults to `data.instrument_correction: remove_sensitivity`:
divide by the scalar instrument sensitivity, retaining the frequency-dependent
response. Set it to `remove_response` to deconvolve the full response instead.
Both processing entry points use this setting.

Run the complete processing workflow from the repository root:

```bash
python scripts/run_all.py
```

Weekly files are the canonical waveform outputs. The workflow stores one shared
filtered trace per station/week plus one denoised trace per pipeline; it does not
store processed raw waveforms or duplicate full-period traces by default. Gap
metadata is also shared between pipelines. This layout keeps runs resumable
without multiplying storage for long time periods.

To additionally create the legacy-style full-period raw, filtered, and denoised
traces and waveform figures, use:

```bash
python scripts/run_all.py --stitch-full
```

Stage 2 stores compact trigger and coincidence arrays by default. Full STA/LTA
characteristic functions can be retained for diagnostics with `--save-cft`, but
they can be very large for week-long traces:

```bash
python scripts/run_all.py --save-cft
```

The download cache remains enabled so interrupted or repeated runs do not need
to fetch the source waveform data again.

The first processing run may download seismic data and the pretrained
SeisBench denoising model, so internet access is required.

For the two-hour denoising investigation (2025-11-01 00:05–02:05 UTC), run:

```bash
python scripts/1_process_data.py --config config.diagnostic.yaml --diagnostics
```

This runs both original pipelines without changing their filters. Results go
to `outputs_diagnostic/<run_id>/`. Under `diagnostics/week_01/<pipeline>/`, each
station segment has intermediate MiniSEED waveforms, `stage_spectra.png`, and
`checks.json`: input, SeisBench preprocessing (including automatic resampling),
model output before the final filter, and final output. Spectra use a common
time interval within the requested window, excluding 30 seconds at each segment
edge. The intermediate waveforms retain the processing padding. Reports include sample
rates, sample counts, timing shifts, RMS, band powers, and peaks near the model's
frequency grid. These peaks are diagnostic measurements, not automatic proof
of artifacts. Invalid samples stop processing; coverage changes are warnings.

To run detection and catalogs afterward with the same configuration:

```bash
python scripts/run_all.py --config config.diagnostic.yaml --start-at 2
```

To repeat the same two-hour test using only the scalar instrument sensitivity:

```bash
python scripts/1_process_data.py --config config.sensitivity.yaml --diagnostics
```

This saves a separate run under `outputs_diagnostic/sensitivity/`. The setting
`data.instrument_correction: remove_sensitivity` divides by the overall
sensitivity without deconvolving the frequency-dependent instrument response.
The baseline `config.diagnostic.yaml` explicitly retains `remove_response` for
comparison; the main configuration defaults to `remove_sensitivity`.
Sensitivity-only runs require native
velocity units for the current waveform labels.
