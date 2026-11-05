"""Run configuration.

One YAML file is the single source of truth for a run. It is loaded into
typed dataclasses (typos in keys raise an error instead of being ignored),
and a copy of the *effective* config is written next to every output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from obspy import UTCDateTime


@dataclass
class DataConfig:
    source: str = "IRIS"
    network: str = "AV"
    stations: List[str] = field(default_factory=lambda: ["SSLS", "SSLN"])
    primary_station: str = "SSLS"
    location: str = ""
    channel: str = "BH*"
    component: str = "BHZ"
    start_time: str = "2025-11-01T00:02:00"
    stop_time: str = "2025-11-01T00:12:00"
    edge_trim_seconds: float = 5.0
    cache_dir: str = "cache"


@dataclass
class BandpassConfig:
    """Used by scripts/01_filter_then_denoise.py"""
    freqmin: float = 0.1
    freqmax: float = 8.0
    corners: int = 2
    zerophase: bool = True


@dataclass
class HighpassConfig:
    """Used by scripts/02_denoise_then_filter.py"""
    freq: float = 0.1
    corners: int = 2
    zerophase: bool = True


@dataclass
class FilterConfig:
    bandpass: BandpassConfig = field(default_factory=BandpassConfig)
    highpass: HighpassConfig = field(default_factory=HighpassConfig)

    def __post_init__(self):
        # allow plain dicts (as read from YAML), rejecting unknown keys
        if isinstance(self.bandpass, dict):
            self.bandpass = _build(BandpassConfig, self.bandpass, "filter.bandpass")
        if isinstance(self.highpass, dict):
            self.highpass = _build(HighpassConfig, self.highpass, "filter.highpass")


@dataclass
class DenoiseConfig:
    pretrained: str = "original"


@dataclass
class TriggerConfig:
    sta_seconds: float = 1.5
    lta_seconds: float = 15.0
    thr_on: float = 2.0
    thr_off: float = 1.0
    startup_seconds: float = 60.0
    coincidence_sum: Optional[int] = None


@dataclass
class OutputConfig:
    dir: str = "outputs"
    compact_csv_floats: bool = True


@dataclass
class PlotConfig:
    spectrogram_window_seconds: float = 5.0
    spectrogram_overlap: float = 0.5
    spectrogram_ref: float = 1.0
    spectrogram_ylim: List[float] = field(default_factory=lambda: [0, 10])
    colormap: str = "viridis"
    waveform_ylim: List[float] = field(default_factory=lambda: [-1e-5, 1e-5])
    waveform_yticks: List[float] = field(default_factory=lambda: [-5e-6, 0.0, 5e-6])
    explainer_event_idx: Optional[int] = 1
    explainer_pad_before: float = 30.0
    explainer_pad_after: float = 20.0

    def __post_init__(self):
        # PyYAML reads "1e-5" (no dot) as a string; coerce to be safe.
        self.spectrogram_ylim = [float(x) for x in self.spectrogram_ylim]
        self.waveform_ylim = [float(x) for x in self.waveform_ylim]
        self.waveform_yticks = [float(x) for x in self.waveform_yticks]


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    # Directory that relative paths (cache, outputs) are resolved against.
    # Set by load_config(); not written to config_used.yaml.
    base_dir: str = "."

    def __post_init__(self):
        self.validate()

    # ---- derived values -------------------------------------------------
    @property
    def start(self) -> UTCDateTime:
        return UTCDateTime(str(self.data.start_time))

    @property
    def stop(self) -> UTCDateTime:
        return UTCDateTime(str(self.data.stop_time))

    @property
    def coincidence_sum(self) -> int:
        """Stations that must trigger together; defaults to all stations."""
        n = self.trigger.coincidence_sum
        return len(self.data.stations) if n is None else int(n)

    @property
    def run_id(self) -> str:
        """Deterministic name for a run: same inputs -> same folder."""
        fmt = "%Y%m%dT%H%M%S"
        return f"{self.data.network}_{self.start.strftime(fmt)}_{self.stop.strftime(fmt)}"

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else Path(self.base_dir) / p

    @property
    def run_dir(self) -> Path:
        return self.resolve(self.output.dir) / self.run_id

    # ---- checks / io ----------------------------------------------------
    def validate(self) -> None:
        d, t = self.data, self.trigger
        if d.primary_station not in d.stations:
            raise ValueError(f"primary_station {d.primary_station!r} not in stations {d.stations}")
        if self.start >= self.stop:
            raise ValueError(f"start_time {self.start} must be before stop_time {self.stop}")
        bp, hp = self.filter.bandpass, self.filter.highpass
        if not (0 < bp.freqmin < bp.freqmax):
            raise ValueError("filter.bandpass needs 0 < freqmin < freqmax")
        if hp.freq <= 0:
            raise ValueError("filter.highpass.freq must be > 0")
        if not (1 <= self.coincidence_sum <= len(d.stations)):
            raise ValueError("coincidence_sum must be between 1 and the number of stations")
        if t.thr_off >= t.thr_on:
            raise ValueError("trigger.thr_off should be lower than trigger.thr_on")

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out.pop("base_dir", None)
        out["data"]["start_time"] = str(self.data.start_time)
        out["data"]["stop_time"] = str(self.data.stop_time)
        return out

    def save(self, path: Path) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def _build(cls, values: Optional[Dict[str, Any]], section: str):
    values = values or {}
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) in '{section}': {sorted(unknown)}. Allowed: {sorted(allowed)}")
    return cls(**values)


def load_config(path, **overrides) -> Config:
    """Load a YAML config.

    overrides: dotted keys, e.g. load_config("config.yaml", **{"data.start_time": "..."}).
    """
    path = Path(path).resolve()
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    for dotted, value in overrides.items():
        if value is None:
            continue
        section, key = dotted.split(".", 1)
        raw.setdefault(section, {})[key] = value

    unknown = set(raw) - {"data", "filter", "denoise", "trigger", "output", "plot"}
    if unknown:
        raise ValueError(f"Unknown top-level section(s): {sorted(unknown)}")

    return Config(
        data=_build(DataConfig, raw.get("data"), "data"),
        filter=_build(FilterConfig, raw.get("filter"), "filter"),
        denoise=_build(DenoiseConfig, raw.get("denoise"), "denoise"),
        trigger=_build(TriggerConfig, raw.get("trigger"), "trigger"),
        output=_build(OutputConfig, raw.get("output"), "output"),
        plot=_build(PlotConfig, raw.get("plot"), "plot"),
        base_dir=str(path.parent),
    )
