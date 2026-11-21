import contextlib
import copy
import io
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

from Tools.Conversion.adapters.gemma4_31b import reference, summary
from Tools.Conversion.adapters.gemma4_31b.compare import compare
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import PipelineError

RECIPE = Path(__file__).resolve().parents[1] / "recipes/gemma4_31b"
MONITOR = "Tools.Conversion.adapters.jev27b.monitor"
CASES = ["ja_question", "ja_system", "ja_two_turns", "en_question", "en_system", "code_python", "code_swift", "json_ja",
         "json_en", "ja_long", "one_word", "raw_text"]
VOCABULARY = 8
MEMORY = {"max_rss_bytes": 10, "footprint_bytes": 5, "lifetime_peak_footprint_bytes": 12,
          "mps_driver_allocated_bytes": 8, "mps_current_allocated_bytes": 7}
OBSERVED = {"samples": 3, "interval_s": 5, "aborts": 0, "failure": None, "unreadable_samples": 0,
            "first": {"swap_used_bytes": 5}, "peak": {"swap_used_bytes": 5, "rss_bytes": 11},
            "lowest": {"free_percent": 40, "free_bytes": {"volume/folder": 1}}}


class FakeTokenizer:
    def __init__(self, chat, text, rendered=None):
        self.chat, self.text, self.rendered = chat, text, chat if rendered is None else rendered

    def apply_chat_template(self, messages, tokenize, return_dict=False, **options):
        return {"input_ids": self.chat} if tokenize else "rendered"

    def __call__(self, text, add_special_tokens):
        return {"input_ids": self.rendered if text == "rendered" else self.text}


class FakeModel:
    def __init__(self, outputs, last_row=-1.0, rerun_nan=False):
        self.outputs, self.last_row, self.rerun_nan, self.calls = outputs, last_row, rerun_nan, 0

    def generate(self, input_ids, attention_mask, generation_config, stopping_criteria):
        self.calls += 1
        rows = []
        for token in self.outputs:
            row = torch.full((1, VOCABULARY), -1.0)
            row[0, token] = 2.0
            if self.rerun_nan and self.calls == 2 and not rows:
                row[0, 0] = math.nan
            rows.append(row)
            stopping_criteria(input_ids, row)
        return SimpleNamespace(sequences=torch.tensor([input_ids[0].tolist() + self.outputs]), logits=tuple(rows))

    def __call__(self, input_ids, attention_mask, use_cache):
        tokens = input_ids[0].tolist()
        logits = torch.full((1, len(tokens), VOCABULARY), -1.0)
        for position, token in enumerate(tokens[1:]):
            logits[0, position, token] = 2.0
        logits[0, -1, 0] = self.last_row
        return SimpleNamespace(logits=logits)


class ReadyMonitor:
    def __init__(self, on_abort, delay):
        self.on_abort, self.delay, self.ready = on_abort, delay, threading.Event()

    def start(self):
        if self.delay is not None:
            threading.Timer(self.delay, self.ready.set).start()

    def finish(self):
        return {"samples": 1, "aborts": 0}


def write_loaded_files(folder):
    snapshot = Path(folder) / "model"
    snapshot.mkdir()
    files = {}
    for name in (*reference.LOADED_FILES, "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "README.md"):
        (snapshot / name).write_bytes(name.encode())
        files[name] = {"bytes": len(name), "sha256": sha256(snapshot / name)}
    return snapshot, {"files": files}


def run_main(folder, monitor, run):
    prepare, output = Path(folder) / "prepare", Path(folder) / "out"
    prepare.mkdir()
    output.mkdir()
    (prepare / "ready.json").write_text(json.dumps({"ready": True, "contract_sha256": sha256(RECIPE / "model.json")}))
    argv = ["reference", "--model", str(RECIPE / "model.json"), "--snapshot", str(folder), "--fixtures",
            str(RECIPE / "fixtures.json"), "--prepare", str(prepare), "--output", str(output)]
    with mock.patch.object(sys, "argv", argv), mock.patch.object(reference, "ReferenceMonitor", monitor), \
            mock.patch.object(reference, "reference", run):
        return reference.main(), output


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


class RenderTests(unittest.TestCase):
    CHAT = {"id": "chat", "kind": "chat", "messages": [{"role": "user", "content": "x"}]}
    RAW = {"id": "raw", "kind": "raw", "text": "x"}

    def test_inputs_start_with_exactly_one_bos(self):
        self.assertEqual(reference.render(FakeTokenizer([2, 5, 6], []), self.CHAT, 2), ([2, 5, 6], "template"))
        self.assertEqual(reference.render(FakeTokenizer([], [5, 6]), self.RAW, 2), ([2, 5, 6], "explicit"))
        rejected = [(FakeTokenizer([5, 6], []), self.CHAT), (FakeTokenizer([2, 2, 6], []), self.CHAT),
                    (FakeTokenizer([2, 6, 2], []), self.CHAT), (FakeTokenizer([], [5, 2]), self.RAW),
                    (FakeTokenizer([2, 5], [], rendered=[2, 6]), self.CHAT)]
        for index, (tokenizer, case) in enumerate(rejected):
            with self.subTest(index=index), self.assertRaises(ValueError):
                reference.render(tokenizer, case, 2)


@unittest.skipIf(np is None, "NumPy is required for the record checks")
class RecordTests(unittest.TestCase):
    def test_the_scoring_positions_run_from_the_last_input_to_the_second_last_output(self):
        sequence = np.arange(8)
        self.assertEqual(sequence[reference.scoring_slice(5, 3)].tolist(), [4, 5, 6])
        self.assertEqual(sequence[reference.scoring_slice(1, 1)].tolist(), [0])

    def test_forced_statistics_count_matches_mismatches_and_ties(self):
        logits = np.array([[0, 5, 1], [2, 0, 1], [0, 1, 3], [4, 4, 0]], dtype=np.float32)
        record = reference.forced_statistics(logits, [1, 0, 1, 0])
        self.assertEqual((record["positions"], record["argmax_matches"], record["top_ties"]), (4, 3, 1))
        mismatch = record["mismatches"][0]
        weights = [math.exp(value - 3) for value in (0, 1, 3)]
        self.assertEqual((mismatch["position"], mismatch["greedy"], mismatch["top2"]), (2, 1, [2, 1]))
        self.assertAlmostEqual(mismatch["probability_difference"], (weights[2] - weights[1]) / sum(weights))
        logits[0, 0], logits[2, 1] = np.nan, np.inf
        self.assertEqual(reference.non_finite(logits), 2)
        record = reference.forced_statistics(logits, [1, 0, 1, 0])
        self.assertEqual([entry["position"] for entry in record["mismatches"]], [0])
        self.assertNotIn("probability_difference", record["mismatches"][0])
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


@unittest.skipIf(np is None or torch is None, "NumPy and PyTorch are required for the case loop")
class RunCasesTests(unittest.TestCase):
    CASE = {"id": "raw", "kind": "raw", "text": "x", "max_new_tokens": 4}

    def test_every_forced_position_is_saved_and_the_output_rows_are_scored(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            records, _ = reference.run_cases(FakeModel([5, 6, 1]), FakeTokenizer([], [7, 3]), [self.CASE], [1], 2, "cpu",
                                             Path(folder))
            record = records[0]
            saved = np.load(Path(folder) / "cases/raw.logits.npy")
            tokens = json.loads((Path(folder) / "cases/raw.tokens.json").read_text())
        self.assertEqual(tokens["input_tokens"] + tokens["output_tokens"], [2, 7, 3, 5, 6, 1])
        self.assertEqual((saved.shape, saved.dtype.name), ((6, VOCABULARY), "float32"))
        self.assertEqual(saved[2:5].argmax(-1).tolist(), [5, 6, 1])
        self.assertEqual((record["saved_positions"], record["first_scoring_row"], record["scoring_positions"],
                          record["forced_argmax_matches"], record["stop_reason"], record["non_finite"]),
                         (6, 2, 3, 3, "eos", {"forced": 0, "steps": 0}))
        self.assertTrue(record["rerun"]["tokens_match"])

    def test_non_finite_values_are_counted_over_every_saved_row(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            records, _ = reference.run_cases(FakeModel([5, 6, 1], last_row=math.nan), FakeTokenizer([], [7, 3]),
                                             [self.CASE], [1], 2, "cpu", Path(folder))
        self.assertEqual((records[0]["non_finite"]["forced"], records[0]["forced_argmax_matches"]), (1, 3))

    def test_non_finite_values_in_the_second_greedy_run_are_counted(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            records, _ = reference.run_cases(FakeModel([5, 6, 1], rerun_nan=True), FakeTokenizer([], [7, 3]),
                                             [self.CASE], [1], 2, "cpu", Path(folder))
        record = records[0]
        self.assertEqual((record["non_finite"], record["rerun"]["tokens_match"], record["rerun"]["non_finite"]),
                         ({"forced": 0, "steps": 0}, True, 1))
        self.assertFalse(reference.evaluate(loading(), records, "mps")["finite"])


def loading(**changes):
    return {"info": {"missing_keys": []}, "parameter_dtypes": ["torch.bfloat16"], "parameter_devices": ["mps:0"],
            "training": False, **changes}


def case_record(non_finite=0, rerun=True, rerun_non_finite=0):
    return {"non_finite": {"forced": non_finite, "steps": 0}, "greedy_chosen_is_max": True,
            "rerun": {"tokens_match": rerun, "non_finite": rerun_non_finite}}


def checked_record(checks):
    return {"checks": dict(checks), "loading": {"after_load": dict(MEMORY)}, "resources": {"cases": [dict(MEMORY)]}}


class CheckTests(unittest.TestCase):
    def test_non_finite_values_or_a_different_rerun_fail_the_reference(self):
        clean = reference.evaluate(loading(), [case_record()], "mps")
        self.assertTrue(reference.finalize(checked_record(clean), copy.deepcopy(OBSERVED), dict(MEMORY))["passed"])
        for record, key in ((case_record(non_finite=1), "finite"), (case_record(rerun_non_finite=1), "finite"),
                            (case_record(rerun=False), "rerun_tokens_match")):
            checks = reference.evaluate(loading(), [case_record(), record], "mps")
            with self.subTest(key=key):
                self.assertFalse(checks[key])
                self.assertFalse(reference.finalize(checked_record(checks), copy.deepcopy(OBSERVED), dict(MEMORY))["passed"])
        self.assertFalse(reference.evaluate(loading(info={"missing_keys": ["x"]}), [case_record()], "mps")["loading_clean"])

    def test_a_stopped_or_unreadable_monitor_or_missing_resources_fail_the_reference(self):
        changes = [("observed", {"aborts": 1}), ("observed", {"failure": "monitor stopped: TimeoutExpired"}),
                   ("observed", {"unreadable_samples": 1}), ("observed", {"samples": 0}), ("observed", {"first": None}),
                   ("final", {"lifetime_peak_footprint_bytes": None}), ("final", {"mps_driver_allocated_bytes": None}),
                   ("final", {"max_rss_bytes": 0})]
        for target, change in changes:
            observed, final = copy.deepcopy(OBSERVED), dict(MEMORY)
            (observed if target == "observed" else final).update(change)
            with self.subTest(change=change):
                record = reference.finalize(checked_record({"finite": True}), observed, final)
                self.assertFalse(record["passed"])

    @unittest.skipIf(torch is None, "PyTorch is required to read the memory record")
    def test_a_missing_footprint_fails_the_reference(self):
        with mock.patch.object(reference, "footprint", return_value=(None, None)):
            final = {**reference.memory("cpu"), "mps_driver_allocated_bytes": 8}
        self.assertIsNone(final["lifetime_peak_footprint_bytes"])
        self.assertFalse(reference.finalize(checked_record({"finite": True}), copy.deepcopy(OBSERVED), final)["passed"])


class InputTests(unittest.TestCase):
    def test_the_files_to_load_are_hashed_against_the_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot, contract = write_loaded_files(folder)
            verified = reference.verify_inputs(snapshot, contract)
        self.assertEqual(set(verified), set(contract["files"]) - {"README.md"})
        self.assertEqual(verified["model-00002-of-00002.safetensors"], contract["files"]["model-00002-of-00002.safetensors"])

    def test_a_changed_resized_or_linked_file_to_load_is_rejected(self):
        def relink(snapshot):
            copy_path = snapshot.parent / "copy"
            copy_path.write_bytes((snapshot / "config.json").read_bytes())
            (snapshot / "config.json").unlink()
            (snapshot / "config.json").symlink_to(copy_path)

        changes = {"content": lambda snapshot: (snapshot / "chat_template.jinja").write_bytes(b"x" * len("chat_template.jinja")),
                   "size": lambda snapshot: (snapshot / "tokenizer.json").write_bytes(b"another size"), "link": relink}
        for label, change in changes.items():
            with tempfile.TemporaryDirectory() as folder, self.subTest(label=label):
                snapshot, contract = write_loaded_files(folder)
                change(snapshot)
                with self.assertRaises(ValueError):
                    reference.verify_inputs(snapshot, contract)

    @unittest.skipIf(torch is None, "PyTorch is required to call the reference step")
    def test_a_snapshot_changed_after_prepare_stops_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot, contract = write_loaded_files(folder)
            args = SimpleNamespace(snapshot=snapshot, fixtures=RECIPE / "fixtures.json", output=Path(folder))
            with mock.patch.object(reference, "runtime_tokenizer", side_effect=RuntimeError("tokenizer")) as tokenizer, \
                    mock.patch.object(reference, "load_model") as load:
                with self.assertRaises(RuntimeError):
                    reference.reference(args, contract, {})
                shard = snapshot / "model-00002-of-00002.safetensors"
                shard.write_bytes(b"y" * shard.stat().st_size)
                with self.assertRaises(ValueError):
                    reference.reference(args, contract, {})
            self.assertEqual(tokenizer.call_count, 1)
            load.assert_not_called()


class MonitorTests(unittest.TestCase):
    def monitor(self, folder, reasons, swaps, free=50):
        patches = [mock.patch(f"{MONITOR}.swap_used", side_effect=swaps), mock.patch(f"{MONITOR}.free_percent", return_value=free),
                   mock.patch(f"{MONITOR}.command_output", return_value="1\n")]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return reference.ReferenceMonitor(Path(folder) / "monitor.tsv", [folder], [], reasons.append)

    def wait(self, monitor, reasons):
        monitor.start()
        deadline = time.monotonic() + 5
        while not reasons and time.monotonic() < deadline:
            time.sleep(0.01)
        return monitor.finish()

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
            observed = self.wait(self.monitor(folder, reasons, itertools.chain([10**9], itertools.repeat(4 * 10**9))), reasons)
        self.assertTrue(reasons and reasons[0].startswith("swap rose"))
        thread.interrupt_main.assert_called_once()
        self.assertEqual(observed["aborts"], 1)

    def test_a_sampling_error_stops_the_stage_instead_of_ending_the_monitor_silently(self):
        calls, reasons = [], []

        def swap():
            calls.append(1)
            if len(calls) > 1:
                raise subprocess.TimeoutExpired(["sysctl"], 30)
            return 10**9

        with tempfile.TemporaryDirectory() as folder, mock.patch(f"{MONITOR}.INTERVAL_S", 0.01), \
                mock.patch(f"{MONITOR}._thread") as thread, mock.patch.object(reference, "schedule_kill") as kill:
            observed = self.wait(self.monitor(folder, reasons, swap), reasons)
        self.assertTrue(reasons and reasons[0].startswith("monitor stopped: TimeoutExpired"))
        kill.assert_called_once()
        self.assertAlmostEqual(kill.call_args.args[0], 120, delta=1)
        thread.interrupt_main.assert_called_once()
        self.assertEqual((observed["aborts"], observed["failure"] is None), (1, False))
        self.assertFalse(reference.finalize(checked_record({"finite": True}), observed, dict(MEMORY))["passed"])

    def test_unreadable_memory_pressure_output_stops_the_stage(self):
        outputs = {"ps": "1\n", "sysctl": "total = 6144.00M  used = 1024.00M  free = 5120.00M  (encrypted)\n",
                   "memory_pressure": "unexpected output\n"}
        reasons = []
        with tempfile.TemporaryDirectory() as folder, mock.patch(f"{MONITOR}.INTERVAL_S", 0.01), \
                mock.patch(f"{MONITOR}._thread") as thread, \
                mock.patch(f"{MONITOR}.command_output", side_effect=lambda *command: outputs[command[0]]):
            monitor = reference.ReferenceMonitor(Path(folder) / "monitor.tsv", [folder], [], reasons.append)
            observed = self.wait(monitor, reasons)
        self.assertFalse(monitor.ready.is_set())
        self.assertEqual(reasons[:1], [reference.UNREADABLE])
        thread.interrupt_main.assert_called_once()
        self.assertGreaterEqual(observed["unreadable_samples"], 1)
        self.assertFalse(reference.finalize(checked_record({"finite": True}), observed, dict(MEMORY))["passed"])

    def test_an_unreadable_sample_still_kills_a_process_that_does_not_stop(self):
        reasons = []
        with tempfile.TemporaryDirectory() as folder, mock.patch(f"{MONITOR}.INTERVAL_S", 0.01), \
                mock.patch(f"{MONITOR}.KILL_AFTER_S", 0.05), mock.patch(f"{MONITOR}._thread"), mock.patch("os.kill") as kill:
            monitor = self.monitor(folder, reasons, itertools.repeat(10**9), free=None)
            monitor.start()
            deadline = time.monotonic() + 5
            while not kill.called and time.monotonic() < deadline:
                time.sleep(0.01)
            monitor.finish()
        self.assertEqual(reasons[:1], [reference.UNREADABLE])
        self.assertEqual(kill.call_args.args[1], reference.signal.SIGKILL)

    def test_an_aborted_run_fails_without_a_reference_record(self):
        def monitor(path, volumes, patterns, on_abort):
            fake = ReadyMonitor(on_abort, delay=None)
            fake.ready.set()
            on_abort("swap rose by 2000000001 B, above 2000000000 B")
            return fake

        def interrupted(*arguments):
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as folder:
            code, output = run_main(folder, monitor, interrupted)
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual((code, failure["passed"], failure["abort"][:9]), (1, False, "swap rose"))
            self.assertFalse((output / "reference.json").exists())

    def test_the_model_is_not_loaded_before_the_first_readable_sample(self):
        seen = {}

        def monitor(path, volumes, patterns, on_abort):
            seen["monitor"] = ReadyMonitor(on_abort, delay=0.1)
            return seen["monitor"]

        def interrupted(*arguments):
            seen["ready"] = seen["monitor"].ready.is_set()
            seen["monitor"].on_abort("stopped by the test")
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as folder:
            code, _ = run_main(folder, monitor, interrupted)
        self.assertEqual((code, seen["ready"]), (1, True))

    def test_no_readable_first_sample_fails_without_loading(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(reference, "FIRST_SAMPLE_TIMEOUT_S", 0.05):
            started = mock.Mock()
            code, output = run_main(folder, lambda path, volumes, patterns, on_abort: ReadyMonitor(on_abort, None), started)
            failure = json.loads((output / "failure.json").read_text())
        started.assert_not_called()
        self.assertEqual(code, 1)
        self.assertTrue(failure["abort"].startswith("no readable monitor sample"))


ROWS = [[0, 2], [1, 0], [3, 0], [0, 1], [2, 2]]


def write_run(folder, inputs, outputs, logits, passed=True, dtype="float32", cases=("case",)):
    folder = Path(folder)
    (folder / "cases").mkdir(parents=True)
    common = {"model": {"id": "owner/model"}, "dtype": {"weights": "bfloat16"}, "device": "mps", "fixtures_sha256": "f",
              "environment": {"python": "3.11"}}
    (folder / "reference.json").write_text(json.dumps({**common, "passed": passed, "cases": [{"id": name} for name in cases]}))
    (folder / "cases/case.tokens.json").write_text(json.dumps({"input_tokens": inputs, "output_tokens": outputs}))
    np.save(folder / "cases/case.logits.npy", np.asarray(logits, dtype=dtype))


@unittest.skipIf(np is None, "NumPy is required for the comparison")
class CompareTests(unittest.TestCase):
    def test_identical_runs_pass_and_a_changed_output_is_located(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second, third = (Path(folder) / name for name in ("a", "b", "c"))
            write_run(first, [2, 7], [4, 5, 1], ROWS)
            write_run(second, [2, 7], [4, 5, 1], ROWS)
            write_run(third, [2, 7], [4, 6, 1], [[0, 2.5], [1, 0.25], [3, 0.5], [9, 9], [9, 9]])
            same = compare(first, second, vocabulary=2)
            self.assertEqual((same["passed"], same["max_abs_logit_difference"], same["cases"][0]["compared_positions"]),
                             (True, 0.0, 5))
            changed = compare(first, third, vocabulary=2)["cases"][0]
            self.assertEqual((changed["output_tokens_match"], changed["first_difference"], changed["compared_positions"],
                              changed["max_abs_logit_difference"]), (False, 3, 3, 0.5))

    def test_runs_with_different_input_tokens_do_not_match(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "a", Path(folder) / "b"
            write_run(first, [2, 7], [4, 5, 1], ROWS)
            write_run(second, [2, 8], [4, 5, 1], ROWS)
            result = compare(first, second, vocabulary=2)
        case = result["cases"][0]
        self.assertFalse(result["passed"])
        self.assertEqual((case["input_tokens_match"], case["output_tokens_match"], case["first_difference"],
                          case["compared_positions"]), (False, True, 1, 1))

    def test_missing_rows_another_dtype_a_failed_reference_or_a_missing_case_never_pass(self):
        variants = {"missing rows": {"logits": ROWS[:3]}, "dtype": {"dtype": "float64"}, "failed reference": {"passed": False},
                    "missing case": {"cases": ()}, "extra case": {"cases": ("case", "other")}}
        for label, changes in variants.items():
            with tempfile.TemporaryDirectory() as folder, self.subTest(label=label):
                first, second = Path(folder) / "a", Path(folder) / "b"
                write_run(first, [2, 7], [4, 5, 1], ROWS)
                write_run(second, [2, 7], [4, 5, 1], changes.pop("logits", ROWS), **changes)
                result = compare(first, second, vocabulary=2)
                self.assertFalse(result["passed"])
                self.assertTrue(result["problems"])


def reference_record():
    case = {"id": "raw_text", "kind": "raw", "bos": "explicit", "input_tokens": 3, "output_tokens": 2, "max_new_tokens": 2,
            "stop_reason": "length", "stop_token": None, "saved_positions": 5, "first_scoring_row": 2,
            "scoring_positions": 2, "non_finite": {"forced": 0, "steps": 0}, "greedy_chosen_is_max": True,
            "step_top_ties": 0, "forced_argmax_matches": 1, "forced_top_ties": 1,
            "forced_mismatches": [{"position": 1, "greedy": 4, "top2": [3, 4], "probability_difference": 0.01}],
            "step_vs_forced_max_abs_logit": 0.125,
            "timing": {"greedy_s": 1.0, "first_token_s": 0.5, "tokens_per_s": 2.0, "decode_tokens_per_s": None, "forced_s": 0.2},
            "rerun": {"tokens_match": True, "greedy_s": 1.0, "non_finite": 0, "first_difference": None},
            "files": {"tokens": {"bytes": 1, "sha256": "a" * 64},
                      "logits": {"bytes": 2, "sha256": "b" * 64, "dtype": "float32", "shape": [5, 262144]}}}
    return {"schema": reference.SCHEMA, "passed": True, "checks": {"finite": True},
            "model": {"id": "google/gemma-4-31B-it", "revision": "c" * 40, "contract_sha256": "d" * 64,
                      "weights_sha256": {"model-00001-of-00002.safetensors": "e" * 64}},
            "environment": {"python": "3.11.9", "os_build": "26A428", "packages": {"torch": "2.15.0"}},
            "dtype": {"weights": "bfloat16"}, "device": "mps",
            "loading": {"class": "Gemma4ForConditionalGeneration", "attn_implementation": "sdpa", "parameters": 9,
                        "parameter_dtypes": ["torch.bfloat16"], "parameter_devices": ["mps:0"],
                        "info": {"missing_keys": []}, "tied_embeddings": True, "after_load": dict(MEMORY)},
            "fixtures_sha256": "f" * 64, "cases": [case],
            "resources": {"cases": [dict(MEMORY)], "final": dict(MEMORY), "mps_recommended_max_bytes": 100,
                          "swap": {"first_bytes": 1, "peak_bytes": 1, "increase_bytes": 0, "limit_increase_bytes": 2},
                          "monitor": copy.deepcopy(OBSERVED)},
            "timing": {"load_s": 1.0, "total_s": 2.0}}


class SummaryTests(unittest.TestCase):
    REPORT = {"version": 1, "status": "completed", "verification": "not_configured",
              "stages": [{"role": "reference", "status": "executed", "execute_s": 1.0}]}

    def test_the_projection_keeps_numbers_and_drops_local_paths(self):
        result = summary.project([self.REPORT], [reference_record()])
        text = json.dumps(result, allow_nan=False)
        self.assertNotIn("volume/folder", text)
        run = result["runs"][0]["reference"]
        case = run["cases"][0]
        self.assertEqual((run["resources"]["max_rss_bytes"], run["resources"]["lifetime_peak_footprint_bytes"],
                          case["forced_mismatches"]), (10, 12, [{"position": 1, "probability_difference": 0.01}]))
        self.assertEqual((case["saved_positions"], case["first_scoring_row"], case["forced_top_ties"],
                          run["resources"]["monitor"]["failed"], run["resources"]["monitor"]["unreadable_samples"]),
                         (5, 2, 1, False, 0))

    def test_the_rerun_projection_counts_problems_without_their_text(self):
        case = {"id": "case", "problems": ["case: logits dtype float64"], "input_tokens_match": True,
                "output_tokens_match": True, "first_difference": None, "compared_positions": 0,
                "max_abs_logit_difference": None, "argmax_changes": 0}
        comparison = {"schema": "gemma4-31b-rerun/1", "passed": False, "problems": case["problems"], "vocabulary": 2,
                      "max_abs_logit_difference": None, "cases": [case]}
        result = summary.project([self.REPORT], [reference_record()], comparison)
        self.assertEqual((result["rerun"]["passed"], result["rerun"]["problems"], result["rerun"]["cases"][0]["problems"],
                          result["rerun"]["max_abs_logit_difference"]), (False, 1, 1, None))
        self.assertNotIn("float64", json.dumps(result))

    def test_free_text_in_a_shareable_field_is_rejected(self):
        for change in (lambda r: r["environment"].update(os_build="folder/name"),
                       lambda r: r["cases"][0].update(stop_reason="free text"), lambda r: r.update(passed="yes")):
            record = reference_record()
            change(record)
            with self.subTest(change=change), self.assertRaises(PipelineError):
                summary.project([self.REPORT], [record])


if __name__ == "__main__":
    unittest.main()
