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

def weekly_dir(cfg: Config, pipeline: str, week_number: int) -> Path:
    d = stage_dir(cfg, pipeline) / "weekly" / f"week_{week_number:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def shared_weekly_dir(cfg: Config, week_number: int) -> Path:
    """Folder for weekly artifacts that are identical in both pipelines."""
    d = cfg.run_dir / "weekly" / f"week_{week_number:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def weekly_trace_dir(
    cfg: Config, pipeline: str, week_number: int, kind: str
) -> Path:
    """Return the canonical folder for a weekly waveform type.

    Raw and filtered waveforms are shared by both pipelines. Denoised
    waveforms remain pipeline-specific.
    """
    if kind in {"raw", "filtered"}:
        base = shared_weekly_dir(cfg, week_number)
    elif kind == "denoised":
        base = weekly_dir(cfg, pipeline, week_number)
    else:
        raise ValueError(f"Unsupported weekly trace kind: {kind!r}")

    d = base / "traces"
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_weekly_traces(
    cfg: Config,
    pipeline: str,
    week_number: int,
    kind: str,
    traces: Dict[str, Trace],
) -> None:
    """Replace one complete weekly trace set in its canonical location."""
    d = weekly_trace_dir(cfg, pipeline, week_number, kind)

    # A rerun may have fewer usable/configured stations. Remove files from the
    # previous attempt so downstream stages cannot silently consume stale data.
    for old_path in d.glob(f"{kind}.*.mseed"):
        old_path.unlink()

    for station, trace in traces.items():
        Stream([trace]).write(
            str(d / f"{kind}.{station}.mseed"), format="MSEED"
        )

def load_weekly_traces(
    cfg: Config, pipeline: str, week_number: int, kind: str
) -> Dict[str, Trace]:
    """Load weekly traces, with fallback support for the legacy layout."""
    canonical = weekly_trace_dir(cfg, pipeline, week_number, kind)
    paths = sorted(canonical.glob(f"{kind}.*.mseed"))

    if not paths and kind in {"raw", "filtered"}:
        legacy = weekly_dir(cfg, pipeline, week_number) / "traces"
        paths = sorted(legacy.glob(f"{kind}.*.mseed"))

    configured = set(cfg.data.stations)
    traces = {}
    for path in paths:
        station = path.name[len(kind) + 1:-len(".mseed")]
        if station not in configured:
            continue
        stream = read(str(path))
        if len(stream) != 1:
            raise RuntimeError(f"Expected one trace in {path}, found {len(stream)}")
        traces[station] = stream[0]

    return traces

def save_weekly_gaps(cfg: Config, week_number: int, gaps_by_station: dict) -> Path:
    path = shared_weekly_dir(cfg, week_number) / "gaps.json"
    payload = {
        station: [
            {"start": str(start), "end": str(end)} for start, end in gaps
        ]
        for station, gaps in gaps_by_station.items()
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path

def weekly_gaps_path(cfg: Config, pipeline: str, week_number: int) -> Path:
    """Find shared gap metadata, falling back to a legacy pipeline copy."""
    canonical = shared_weekly_dir(cfg, week_number) / "gaps.json"
    if canonical.exists():
        return canonical
    return weekly_dir(cfg, pipeline, week_number) / "gaps.json"

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
