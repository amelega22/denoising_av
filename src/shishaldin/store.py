"""save, load, and organize waveform data, detection results, catalogs, figures, and run information for each processing pipeline"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import platform
import sys
from pathlib import Path
from typing import Dict, Literal
import numpy as np
from obspy import Stream, Trace, read

from .config import Config

PIPELINES: tuple[Literal["filter_then_denoise"], Literal["denoise_then_filter"]] = ("filter_then_denoise", "denoise_then_filter")
_PACKAGES = ["numpy", "scipy", "pandas", "matplotlib", "obspy", "seisbench", "torch"]

def stage_dir(cfg: Config, pipeline: str) -> Path:
    d = cfg.run_dir / pipeline
    d.mkdir(parents=True, exist_ok=True)
    return d

def comparison_dir(cfg: Config) -> Path:
    d = cfg.run_dir / "comparison"
    d.mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(exist_ok=True)
    return d

def has_traces(cfg: Config, pipeline: str) -> bool:
    d = cfg.run_dir / pipeline / "traces"

    available = [
        stn for stn in cfg.data.stations
        if all(
            (d / f"{kind}.{stn}.mseed").exists()
            for kind in ("filtered", "denoised")
        )
    ]

    return len(available) >= cfg.coincidence_sum

def has_detections(cfg: Config, pipeline: str) -> bool:
    return (cfg.run_dir / pipeline / "detections.npz").exists()

def has_catalog(cfg: Config, pipeline: str) -> bool:
    return (cfg.run_dir / pipeline / "catalog.csv").exists()

def figure_dir(cfg: Config, pipeline: str) -> Path:
    d = stage_dir(cfg, pipeline) / "figures"
    d.mkdir(exist_ok=True)
    return d

#stuff for traces
def save_traces(cfg: Config, pipeline: str, kind: str, traces: Dict[str, Trace]) -> None:
    d = stage_dir(cfg, pipeline) / "traces"
    d.mkdir(exist_ok=True)
    for stn, tr in traces.items():
        Stream([tr]).write(str(d / f"{kind}.{stn}.mseed"), format="MSEED")

def load_traces(cfg: Config, pipeline: str, kind: str) -> Dict[str, Trace]:
    d = stage_dir(cfg, pipeline) / "traces"
    out = {}

    for stn in cfg.data.stations:
        path = d / f"{kind}.{stn}.mseed"
        if not path.exists():
            continue
        st = read(str(path))
        if len(st) != 1:
            raise ValueError(f"{path}: expected 1 trace, found {len(st)}")
        out[stn] = st[0]
    if not out:
        raise FileNotFoundError(f"No {kind} traces found for pipeline '{pipeline}'")
    return out

def save_detections(cfg: Config, pipeline: str, **arrays) -> None:
    np.savez(stage_dir(cfg, pipeline) / "detections.npz", **arrays)

def load_detections(cfg: Config, pipeline: str) -> Dict[str, np.ndarray]:
    path = stage_dir(cfg, pipeline) / "detections.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the detection script first.")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}

def trigger_params(cfg: Config) -> dict:
    """What a detection depends on besides the traces; used to catch a catalog
    being built from detections made with different settings."""
    return {**asdict(cfg.trigger), "coincidence_sum": cfg.coincidence_sum,
            "stations": list(cfg.data.stations), "primary_station": cfg.data.primary_station}

# save information in json
def save_json(cfg: Config, pipeline: str, name: str, payload: dict) -> Path:
    path = stage_dir(cfg, pipeline) / name
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    return path

def load_json(cfg: Config, pipeline: str, name: str) -> dict:
    with open(stage_dir(cfg, pipeline) / name) as fh:
        return json.load(fh)

def provenance() -> dict:
    versions = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": versions,
    }
