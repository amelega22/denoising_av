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

Run the complete processing workflow from the repository root:

```bash
python scripts/run_all.py
```

The first processing run may download seismic data and the pretrained
SeisBench denoising model, so internet access is required.
