import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

from Tools.Conversion.adapters.gemma4_31b import reference, summary
from Tools.Conversion.adapters.gemma4_31b.compare import compare
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import PipelineError

RECIPE = Path(__file__).resolve().parents[1] / "recipes/gemma4_31b"
MONITOR = "Tools.Conversion.adapters.jev27b.monitor"
CASES = ["ja_question", "ja_system", "ja_two_turns", "en_question", "en_system", "code_python", "code_swift", "json_ja",
         "json_en", "ja_long", "one_word", "raw_text"]


class FixtureTests(unittest.TestCase):
    def test_the_recipe_fixtures_hold_twelve_cases_in_the_fixed_order(self):
        cases = reference.load_fixtures(RECIPE / "fixtures.json")
        self.assertEqual([case["id"] for case in cases], CASES)
        self.assertEqual({case["id"]: case["max_new_tokens"] for case in cases if case["max_new_tokens"] != 64},
                         {"code_python": 128, "code_swift": 128})
        self.assertEqual([case["id"] for case in cases if case["kind"] == "raw"], ["raw_text"])
        roles = {case["id"]: [message["role"] for message in case.get("messages", [])] for case in cases}
        self.assertEqual((roles["ja_system"], roles["ja_two_turns"]), (["system", "user"], ["user", "assistant", "user"]))
        long = cases[CASES.index("ja_long")]["messages"]
        self.assertEqual((len(long), long[0]["content"].count("\n\n")), (1, 30))

    def test_numbered_paragraphs_follow_the_fixed_rule(self):
        spec = {"count": 3, "template": "P{n} {item} {color}", "values": {"item": ["a", "b"], "color": ["x", "y", "z"]},
                "separator": "|", "ask": 2, "question": "Q{n}"}
        self.assertEqual(reference.paragraphs(spec), "P1 a x|P2 b y|P3 a z|Q2")
        with self.assertRaises(ValueError):
            reference.paragraphs({**spec, "ask": 4})

    def test_malformed_fixtures_are_rejected(self):
        original = json.loads((RECIPE / "fixtures.json").read_text())
        mutations = [lambda d: d["cases"][1].update(id="ja_question"), lambda d: d["cases"][0].update(kind="vision"),
                     lambda d: d["cases"][0].update(max_new_tokens=0), lambda d: d["cases"][0].update(extra=True),
                     lambda d: d["cases"][2]["messages"].pop(),
                     lambda d: d["cases"][1]["messages"].reverse(),
                     lambda d: d["cases"][0].update(paragraphs=d["cases"][9]["paragraphs"]),
                     lambda d: d["cases"][9]["paragraphs"].update(ask=31), lambda d: d["cases"][11].update(text=""),
                     lambda d: d.update(version=2)]
        with tempfile.TemporaryDirectory() as folder:
            for index, mutate in enumerate(mutations):
                data = copy.deepcopy(original)
                mutate(data)
                path = Path(folder) / f"{index}.json"
                path.write_text(json.dumps(data))
                with self.subTest(index=index), self.assertRaises(ValueError):
                    reference.load_fixtures(path)


@unittest.skipIf(np is None, "NumPy is required for the record checks")
class RecordTests(unittest.TestCase):
    def test_the_scoring_positions_run_from_the_last_input_to_the_second_last_output(self):
        sequence = np.arange(8)
        self.assertEqual(sequence[reference.scoring_slice(5, 3)].tolist(), [4, 5, 6])
        self.assertEqual(sequence[reference.scoring_slice(1, 1)].tolist(), [0])

    def test_forced_statistics_count_matches_mismatches_and_non_finite_values(self):
        logits = np.array([[0, 5, 1], [2, 0, 1], [0, 1, 3]], dtype=np.float32)
        record = reference.forced_statistics(logits, [1, 0, 1])
        self.assertEqual((record["positions"], record["argmax_matches"], record["non_finite"]), (3, 2, 0))
        mismatch = record["mismatches"][0]
        weights = [math.exp(value - 3) for value in (0, 1, 3)]
        self.assertEqual((mismatch["position"], mismatch["greedy"], mismatch["top2"]), (2, 1, [2, 1]))
        self.assertAlmostEqual(mismatch["probability_difference"], (weights[2] - weights[1]) / sum(weights))
        logits[0, 0], logits[2, 1] = np.nan, np.inf
        record = reference.forced_statistics(logits, [1, 0, 1])
        self.assertEqual(record["non_finite"], 2)
        self.assertNotIn("probability_difference", record["mismatches"][-1])
        json.dumps(record, allow_nan=False)

    def test_step_statistics_report_ties_and_non_maximal_choices(self):
        steps = np.array([[1, 1, 0], [0, 2, 1]], dtype=np.float32)
        record = reference.step_statistics(steps, [1, 2])
        self.assertEqual((record["chosen_is_max"], record["top_ties"], record["non_finite"]), (False, 1, 0))
        self.assertEqual(record["steps"][0], {"top2": [0, 1], "top2_logits": [1.0, 1.0]})
        self.assertTrue(reference.step_statistics(steps, [0, 1])["chosen_is_max"])

    def test_reruns_report_the_first_differing_step(self):
        steps = np.array([[3, 1, 0], [0, 1, 2], [1, 0, 0]], dtype=np.float32)
        self.assertIsNone(reference.first_difference([0, 2, 0], [0, 2, 0], steps, steps))
        difference = reference.first_difference([0, 2, 0], [0, 1, 0], steps, steps)
        self.assertEqual((difference["position"], difference["first"]["top2"]), (1, [2, 1]))
        self.assertEqual(reference.first_difference([0, 2], [0, 2, 0], steps, steps)["position"], 2)

    def test_stop_reasons_distinguish_stop_tokens_and_the_limit(self):
        self.assertEqual(reference.stop_reason([5, 106], [1, 106, 50], 64), "eos")
        self.assertEqual(reference.stop_reason([5, 6], [1, 106, 50], 2), "length")
        self.assertEqual(reference.stop_reason([5], [1, 106, 50], 2), "other")


class MonitorTests(unittest.TestCase):
    def monitor(self, folder, reasons, swaps, free=50):
        patches = [mock.patch(f"{MONITOR}.swap_used", side_effect=swaps), mock.patch(f"{MONITOR}.free_percent", return_value=free),
                   mock.patch(f"{MONITOR}.command_output", return_value="1\n")]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return reference.ReferenceMonitor(Path(folder) / "monitor.tsv", [folder], [], reasons.append)

    def test_a_swap_increase_above_the_limit_stops_the_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            monitor = self.monitor(folder, [], [4_000_000_000, 6_000_000_000, 6_000_000_001])
            verdicts = []
            for _ in range(3):
                monitor.sample()
                verdicts.append(monitor.verdict())
            self.assertEqual(verdicts[:2], [None, None])
            self.assertTrue(verdicts[2].startswith("swap rose by 2000000001 B"))

    def test_low_free_memory_still_stops_the_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            monitor = self.monitor(folder, [], [10**9] * 3, free=5)
            for _ in range(3):
                monitor.sample()
            self.assertTrue(monitor.verdict().startswith("memory free percentage below"))

    def test_the_monitor_records_the_reason_and_interrupts_the_main_thread_once(self):
        reasons = []
        with tempfile.TemporaryDirectory() as folder, mock.patch(f"{MONITOR}.INTERVAL_S", 0.01), \
                mock.patch(f"{MONITOR}._thread") as thread:
            monitor = self.monitor(folder, reasons, [10**9] + [4 * 10**9] * 100)
            monitor.start()
            deadline = time.monotonic() + 5
            while not reasons and time.monotonic() < deadline:
                time.sleep(0.01)
            summary_record = monitor.finish()
        self.assertTrue(reasons and reasons[0].startswith("swap rose"))
        thread.interrupt_main.assert_called_once()
        self.assertEqual(summary_record["aborts"], 1)

    def test_an_aborted_run_fails_without_a_reference_record(self):
        class Fake:
            def __init__(self, path, volumes, patterns, on_abort):
                self.on_abort = on_abort

            def start(self):
                self.on_abort("swap rose by 2000000001 B, above 2000000000 B")

            def finish(self):
                return {"samples": 1, "aborts": 1}

        def interrupted(*arguments):
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as folder:
            prepare, output = Path(folder) / "prepare", Path(folder) / "out"
            prepare.mkdir()
            output.mkdir()
            (prepare / "ready.json").write_text(json.dumps({"ready": True, "contract_sha256": sha256(RECIPE / "model.json")}))
            argv = ["reference", "--model", str(RECIPE / "model.json"), "--snapshot", folder, "--fixtures",
                    str(RECIPE / "fixtures.json"), "--prepare", str(prepare), "--output", str(output)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(reference, "ReferenceMonitor", Fake), \
                    mock.patch.object(reference, "reference", interrupted):
                self.assertEqual(reference.main(), 1)
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual((failure["passed"], failure["abort"][:9]), (False, "swap rose"))
            self.assertFalse((output / "reference.json").exists())


def write_run(folder, outputs, logits):
    folder = Path(folder)
    (folder / "cases").mkdir(parents=True)
    common = {"model": {"id": "owner/model"}, "dtype": {"weights": "bfloat16"}, "device": "mps", "fixtures_sha256": "f",
              "environment": {"python": "3.11"}}
    (folder / "reference.json").write_text(json.dumps({**common, "cases": [{"id": "case"}]}))
    (folder / "cases/case.tokens.json").write_text(json.dumps({"input_tokens": [2, 7], "output_tokens": outputs}))
    np.save(folder / "cases/case.logits.npy", np.asarray(logits, dtype=np.float32))


@unittest.skipIf(np is None, "NumPy is required for the comparison")
class CompareTests(unittest.TestCase):
    def test_identical_runs_pass_and_a_changed_output_is_located(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second, third = (Path(folder) / name for name in ("a", "b", "c"))
            write_run(first, [4, 5, 1], [[0, 2], [1, 0], [3, 0]])
            write_run(second, [4, 5, 1], [[0, 2], [1, 0], [3, 0]])
            write_run(third, [4, 6, 1], [[0, 2.5], [1, 0.25], [9, 9]])
            same = compare(first, second)
            self.assertEqual((same["passed"], same["max_abs_logit_difference"]), (True, 0.0))
            changed = compare(first, third)["cases"][0]
            self.assertEqual((changed["output_tokens_match"], changed["first_difference"], changed["compared_positions"],
                              changed["max_abs_logit_difference"]), (False, 1, 2, 0.5))


def reference_record():
    memory = {"max_rss_bytes": 10, "footprint_bytes": 5, "lifetime_peak_footprint_bytes": 12,
              "mps_driver_allocated_bytes": 8, "mps_current_allocated_bytes": 7}
    case = {"id": "raw_text", "kind": "raw", "bos": "explicit", "input_tokens": 3, "output_tokens": 2, "max_new_tokens": 2,
            "stop_reason": "length", "stop_token": None, "scoring_positions": 2, "non_finite": {"forced": 0, "steps": 0},
            "greedy_chosen_is_max": True, "step_top_ties": 0, "forced_argmax_matches": 1,
            "forced_mismatches": [{"position": 1, "greedy": 4, "top2": [3, 4], "probability_difference": 0.01}],
            "step_vs_forced_max_abs_logit": 0.125,
            "timing": {"greedy_s": 1.0, "first_token_s": 0.5, "tokens_per_s": 2.0, "decode_tokens_per_s": None, "forced_s": 0.2},
            "rerun": {"tokens_match": True, "greedy_s": 1.0, "first_difference": None},
            "files": {"tokens": {"bytes": 1, "sha256": "a" * 64},
                      "logits": {"bytes": 2, "sha256": "b" * 64, "dtype": "float32", "shape": [2, 262144]}}}
    return {"schema": reference.SCHEMA, "passed": True, "checks": {"finite": True},
            "model": {"id": "google/gemma-4-31B-it", "revision": "c" * 40, "contract_sha256": "d" * 64,
                      "weights_sha256": {"model-00001-of-00002.safetensors": "e" * 64}},
            "environment": {"python": "3.11.9", "os_build": "26A428", "packages": {"torch": "2.15.0"}},
            "dtype": {"weights": "bfloat16"}, "device": "mps",
            "loading": {"class": "Gemma4ForConditionalGeneration", "attn_implementation": "sdpa", "parameters": 9,
                        "parameter_dtypes": ["torch.bfloat16"], "parameter_devices": ["mps:0"],
                        "info": {"missing_keys": []}, "tied_embeddings": True, "after_load": memory},
            "fixtures_sha256": "f" * 64, "cases": [case],
            "resources": {"cases": [memory], "final": memory, "mps_recommended_max_bytes": 100,
                          "swap": {"first_bytes": 1, "peak_bytes": 1, "increase_bytes": 0, "limit_increase_bytes": 2},
                          "monitor": {"samples": 3, "interval_s": 5, "aborts": 0, "peak": {"rss_bytes": 11},
                                      "lowest": {"free_percent": 40, "free_bytes": {"volume/folder": 1}}}},
            "timing": {"load_s": 1.0, "total_s": 2.0}}


class SummaryTests(unittest.TestCase):
    REPORT = {"version": 1, "status": "completed", "verification": "not_configured",
              "stages": [{"role": "reference", "status": "executed", "execute_s": 1.0}]}

    def test_the_projection_keeps_numbers_and_drops_local_paths(self):
        result = summary.project([self.REPORT], [reference_record()])
        text = json.dumps(result, allow_nan=False)
        self.assertNotIn("volume/folder", text)
        run = result["runs"][0]["reference"]
        self.assertEqual((run["resources"]["max_rss_bytes"], run["resources"]["lifetime_peak_footprint_bytes"],
                          run["cases"][0]["forced_mismatches"]), (10, 12, [{"position": 1, "probability_difference": 0.01}]))

    def test_free_text_in_a_shareable_field_is_rejected(self):
        for change in (lambda r: r["environment"].update(os_build="folder/name"),
                       lambda r: r["cases"][0].update(stop_reason="free text"), lambda r: r.update(passed="yes")):
            record = reference_record()
            change(record)
            with self.subTest(change=change), self.assertRaises(PipelineError):
                summary.project([self.REPORT], [record])


if __name__ == "__main__":
    unittest.main()
