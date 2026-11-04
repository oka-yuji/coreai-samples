import copy
import unittest

from Tools.Conversion.adapters.jev27b.summarize_comparison import LENGTHS, ORDER, summarize


def fixture():
    inputs = [{"tokens": value, "tokens_sha256": str(value)} for value in LENGTHS]
    return {
        "plan": {"sequence": ORDER, "warmup_rows_per_process": 4, "measured_rounds_per_process": 3,
                 "processes_per_condition": 2, "measured_n_per_length": 6, "inputs": inputs},
        "runs": [{
            "condition": condition, "run_id": f"process-{process}", "completed": True, "guard_stop": None,
            "rows": [{"token_count": value, "tokens_sha256": str(value),
                      "thermal_state_at_start": "nominal",
                      "elapsed_seconds": 999 if round_index == 0 else 10 * round_index + length_index + process}
                     for round_index in range(4) for length_index, value in enumerate(LENGTHS)],
            "memory": {"max_rss_bytes": 200 + process, "kernel_footprint_peak_bytes": 100 + process},
        } for process, condition in enumerate(ORDER)],
    }


class ComparisonSummaryTests(unittest.TestCase):
    def test_warmups_are_excluded_and_balanced_processes_are_combined(self):
        result = summarize(fixture())
        self.assertEqual(result["A"]["latency_seconds"]["100"],
                         {"n": 6, "median": 23.5, "min": 10, "max": 37, "p95": None})
        self.assertEqual(result["A"]["mixed_round_seconds"],
                         {"n": 6, "median": 100, "min": 46, "max": 154, "p95": None})
        self.assertEqual(result["A"]["memory"]["kernel_footprint_peak_bytes"]["max"], 107)
        self.assertEqual(result["A"]["memory"]["max_rss_bytes"]["max"], 207)

    def test_missing_reordered_failed_or_non_nominal_observations_are_rejected(self):
        changes = [
            lambda value: value["runs"].pop(),
            lambda value: value["runs"].append(value["runs"].pop(0)),
            lambda value: value["runs"].__setitem__(7, copy.deepcopy(value["runs"][0])),
            lambda value: value["runs"][0].update(completed=False),
            lambda value: value["runs"][0].update(guard_stop="swap"),
            lambda value: value["runs"][0]["rows"].pop(),
            lambda value: value["runs"][0]["rows"][0].update(thermal_state_at_start="serious"),
            lambda value: value["runs"][0]["rows"][0].update(tokens_sha256="changed"),
            lambda value: value["plan"].update(measured_n_per_length=3),
        ]
        for change in changes:
            document = copy.deepcopy(fixture())
            change(document)
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize(document)

    def test_nonfinite_and_invalid_observations_are_rejected_including_warmups(self):
        for invalid in [float("nan"), float("inf"), 0, -1, True, "1"]:
            for index in [0, 4]:
                document = fixture()
                document["runs"][0]["rows"][index]["elapsed_seconds"] = invalid
                with self.subTest(invalid=invalid, index=index), self.assertRaises(ValueError):
                    summarize(document)


if __name__ == "__main__":
    unittest.main()
