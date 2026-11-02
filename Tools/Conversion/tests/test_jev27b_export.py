from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Tools.Conversion.adapters.jev27b import export, monitor


class AssetTypeTests(unittest.TestCase):
    def test_asset_types_parse_into_dtypes_and_shapes_with_dynamic_dimensions(self):
        times = export.TIMES
        self.assertEqual(export.described(f"NDArray (BFloat16, 16{times}1{times}4{times}?{times}256)"),
                         {"dtype": "bfloat16", "shape": [16, 1, 4, None, 256]})
        self.assertEqual(export.described(f"NDArray (Int32, 1{times}?)"), {"dtype": "int32", "shape": [1, None]})
        with self.assertRaises(ValueError):
            export.described("Tensor<f16>")


class MonitorTests(unittest.TestCase):
    def sampled(self, swaps, frees):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        watcher = monitor.Monitor(Path(folder.name) / "monitor.tsv", [folder.name], [], lambda reason: None)
        verdicts = []
        with mock.patch.object(monitor, "command_output", return_value="1\n"), \
                mock.patch.object(monitor, "swap_used", side_effect=swaps), \
                mock.patch.object(monitor, "free_percent", side_effect=frees):
            for _ in swaps:
                watcher.sample()
                verdicts.append(watcher.verdict())
        return watcher, verdicts

    def test_swap_must_rise_above_the_limit_three_times_in_a_row(self):
        high = int(monitor.SWAP_LIMIT_BYTES)
        _, verdicts = self.sampled([high + 1, high + 2, high + 3, high + 4], [50] * 4)
        self.assertEqual([value is not None for value in verdicts], [False, False, False, True])
        _, steady = self.sampled([high + 1, high + 2, high + 2, high + 3, high + 4], [50] * 5)
        self.assertEqual([value is not None for value in steady], [False] * 5)

    def test_low_free_memory_stops_after_three_samples_and_records_extremes(self):
        _, verdicts = self.sampled([1, 1, 1, 1], [7, 7, 9, 5])
        self.assertEqual([value is not None for value in verdicts], [False, False, False, False])
        watcher, verdicts = self.sampled([1, 1, 1], [7, 6, 5])
        self.assertIn("memory free percentage", verdicts[-1])
        self.assertEqual(watcher.lowest["free_percent"], 5)
        self.assertEqual(watcher.peak["rss_bytes"], 1024)


if __name__ == "__main__":
    unittest.main()
