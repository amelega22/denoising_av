"""Tests for compact weekly waveform storage."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from shishaldin import store


class WeeklyStorageTests(unittest.TestCase):
    def make_cfg(self, root):
        return SimpleNamespace(
            run_dir=Path(root) / "run",
            data=SimpleNamespace(stations=["AAA", "BBB"]),
        )

    def make_trace(self, station, value):
        trace = Trace(
            np.full(20, value, dtype=np.float32),
            header={"station": station, "channel": "BHZ", "sampling_rate": 10},
        )
        trace.stats.starttime = UTCDateTime("2025-01-01")
        return trace

    def test_shared_and_pipeline_specific_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self.make_cfg(folder)
            filtered = {"AAA": self.make_trace("AAA", 1)}
            denoised = {"AAA": self.make_trace("AAA", 2)}

            store.save_weekly_traces(
                cfg, "filter_then_denoise", 1, "filtered", filtered
            )
            for pipeline in store.PIPELINES:
                store.save_weekly_traces(
                    cfg, pipeline, 1, "denoised", denoised
                )

            shared = cfg.run_dir / "weekly/week_01/traces/filtered.AAA.mseed"
            self.assertTrue(shared.is_file())
            self.assertFalse(
                (cfg.run_dir / "filter_then_denoise/weekly/week_01/traces/filtered.AAA.mseed").exists()
            )
            for pipeline in store.PIPELINES:
                loaded = store.load_weekly_traces(
                    cfg, pipeline, 1, "filtered"
                )
                self.assertEqual(set(loaded), {"AAA"})
                self.assertTrue(
                    (cfg.run_dir / pipeline / "weekly/week_01/traces/denoised.AAA.mseed").is_file()
                )

    def test_replacing_trace_set_removes_stale_station_files(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self.make_cfg(folder)
            store.save_weekly_traces(
                cfg, store.PIPELINES[0], 1, "filtered",
                {"AAA": self.make_trace("AAA", 1)},
            )
            store.save_weekly_traces(
                cfg, store.PIPELINES[0], 1, "filtered",
                {"BBB": self.make_trace("BBB", 2)},
            )
            loaded = store.load_weekly_traces(
                cfg, store.PIPELINES[0], 1, "filtered"
            )
            self.assertEqual(set(loaded), {"BBB"})

    def test_legacy_shared_trace_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self.make_cfg(folder)
            legacy = store.weekly_dir(
                cfg, "filter_then_denoise", 1
            ) / "traces"
            legacy.mkdir(parents=True, exist_ok=True)
            Stream([self.make_trace("AAA", 1)]).write(
                str(legacy / "filtered.AAA.mseed"), format="MSEED"
            )

            loaded = store.load_weekly_traces(
                cfg, "filter_then_denoise", 1, "filtered"
            )
            self.assertEqual(set(loaded), {"AAA"})

    def test_characteristic_functions_are_opt_in(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "stage2", root / "scripts/2_sta_lta.py"
        )
        stage2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage2)

        result = {
            "AAA": {
                "cft": np.arange(20, dtype=float),
                "triggers": np.array([[2, 4]]),
                "coincident_flags": np.array([True]),
                "intervals": [],
            }
        }

        with tempfile.TemporaryDirectory() as folder:
            week_dir = Path(folder)
            stage2.save_weekly_detections(
                week_dir,
                np.arange(20), np.arange(20),
                np.array([[2, 4]]), np.array([[3, 5]]),
                result, result, [], [], {},
            )
            with np.load(week_dir / "detections.npz") as data:
                self.assertNotIn("cft_filtered_primary", data.files)
                self.assertNotIn("cft_filtered_AAA", data.files)
                self.assertIn("triggers_filtered_AAA", data.files)


if __name__ == "__main__":
    unittest.main()
