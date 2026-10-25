import json
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Tools.OCREval import reference


class RunPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.set = self.root / "set"
        self.set.mkdir()
        self.out = self.root / "out"
        self.logs = self.root / "logs"
        self.lock = self.root / "locks" / "heavy-job"
        samples = {
            "A": {"id": "A", "pipelines": ["text"], "size_px": [1000, 2000]},
            "B": {"id": "B", "pipelines": ["table", "text"], "table_bbox": [1, 2, 3, 4], "size_px": [1000, 2000]},
            "C": {"id": "C", "pipelines": ["two_stage"], "size_px": [1000, 2000]},
            "D": {"id": "D", "pipelines": ["text"], "crop_px": [5, 6, 7, 8], "image": "page.png", "size_px": [10, 10]},
            "E": {"id": "E", "pipelines": ["table"], "crop_px": [9, 8, 7, 6], "image": "page.png", "size_px": [10, 10]},
        }
        for sid, gt in samples.items():
            (self.set / f"{sid}.json").write_text(json.dumps(gt))
            (self.set / gt.get("image", f"{sid}.png")).write_bytes(b"")
        self.calls = []
        self.environment = mock.Mock(return_value={"os": {"build": "test"}})
        self.busy = []
        self.failure = {}

    def tearDown(self):
        self.tmp.cleanup()

    def fake_child(self, set_dir, out, logs, sid, prompt_id, name, crop=None, resize=None):
        self.calls.append((sid, prompt_id, name, crop, resize))
        if prompt_id == "layout":
            (out / "outputs").mkdir(parents=True, exist_ok=True)
            (out / "outputs" / f"{name}.txt").write_text(
                "<box:100 050 500 100><label:title><up>\n<box:100 200 480 900><label:text><up>\n")
        if sid in self.failure:
            raise self.failure[sid]

    def run_plan(self, only=None):
        with mock.patch.dict(os.environ, {"COREAI_HEAVY_JOB_LOCK": str(self.lock)}), \
                mock.patch.object(reference, "child", self.fake_child), \
                mock.patch.object(reference, "environment", self.environment), \
                mock.patch.object(reference, "model_processes", lambda: self.busy), \
                mock.patch.object(reference.signal, "signal"):
            return reference.run_plan(self.set, self.out, self.logs, only, "313-test", "test-session", 0)

    def record(self):
        return json.loads((self.out / "invocations.jsonl").read_text().splitlines()[-1])

    def released(self):
        return list(self.lock.parent.glob("heavy-job-done-313-test-*"))

    def test_runs_each_pipeline_in_order_and_releases_the_lock(self):
        self.failure = {"A": RuntimeError("A exited with 1")}
        failed = self.run_plan()
        self.assertEqual(failed, [{"sample": "A", "error": "A exited with 1"}])
        self.assertEqual(self.calls, [
            ("A", "text", "A-text-bf16", None, None),
            ("B", "table", "B-table-bf16", [1, 2, 3, 4], None),
            ("B", "text", "B-text-bf16", None, None),
            ("C", "layout", "C-layout-bf16", None, 1036),
            ("C", "text", "C-block01-text-bf16", [100, 100, 500, 200], None),
            ("C", "text", "C-block02-text-bf16", [100, 400, 480, 1800], None),
            ("D", "text", "D-text-bf16", [5, 6, 7, 8], None),
            ("E", "table", "E-table-bf16", [9, 8, 7, 6], None),
        ])
        self.assertFalse(self.lock.exists())
        self.assertEqual(len(self.released()), 1)
        self.assertIn("session: test-session", (self.released()[0] / "owner.txt").read_text())
        record = self.record()
        self.assertEqual(record["samples"], ["A", "B", "C", "D", "E"])
        self.assertEqual(record["completed"], ["B", "C", "D", "E"])
        self.assertIsNone(record["aborted"])
        self.assertTrue(record["lock_released"])
        self.assertEqual(record["os"], {"build": "test"})

    def test_a_timed_out_run_is_recorded_and_the_next_sample_runs(self):
        self.failure = {"A": subprocess.TimeoutExpired("A-text-bf16", 900)}
        failed = self.run_plan(["A", "D"])
        self.assertEqual([f["sample"] for f in failed], ["A"])
        self.assertIn("timed out", failed[0]["error"])
        self.assertEqual([c[0] for c in self.calls], ["A", "D"])
        self.assertEqual(self.record()["completed"], ["D"])
        self.assertEqual(len(self.released()), 1)

    def test_an_interrupt_releases_the_lock_and_marks_the_record(self):
        self.failure = {"B": KeyboardInterrupt()}
        with self.assertRaises(KeyboardInterrupt):
            self.run_plan(["A", "B", "D"])
        self.assertEqual([c[0] for c in self.calls], ["A", "B"])
        self.assertFalse(self.lock.exists())
        self.assertEqual(len(self.released()), 1)
        record = self.record()
        self.assertEqual((record["completed"], record["aborted"]), (["A"], "KeyboardInterrupt"))

    def test_a_changed_owner_keeps_the_lock(self):
        def rewrite(*args, **kwargs):
            (self.lock / "owner.txt").write_text("someone else")
        self.fake_child = rewrite
        with self.assertRaisesRegex(RuntimeError, "lock owner changed"):
            self.run_plan(["A"])
        self.assertEqual((self.lock / "owner.txt").read_text(), "someone else")
        self.assertEqual(self.released(), [])
        self.assertFalse(self.record()["lock_released"])

    def test_busy_lock_is_never_taken(self):
        self.lock.mkdir(parents=True)
        (self.lock / "owner.txt").write_text("someone else")
        with self.assertRaises(TimeoutError):
            self.run_plan()
        self.assertEqual((self.lock / "owner.txt").read_text(), "someone else")
        self.assertEqual(self.calls, [])

    def test_a_running_model_process_blocks_the_lock(self):
        self.busy = ["123 coreai-chat"]
        with self.assertRaises(TimeoutError):
            self.run_plan()
        self.assertFalse(self.lock.exists())
        self.assertEqual(self.calls, [])

    def test_a_failing_process_check_is_an_error(self):
        broken = subprocess.CompletedProcess(["pgrep"], 2, stdout="", stderr="pgrep: bad pattern")
        with mock.patch.object(reference.subprocess, "run", return_value=broken):
            with self.assertRaisesRegex(RuntimeError, "bad pattern"):
                reference.model_processes()

    def test_inputs_are_checked_before_the_environment_and_the_lock(self):
        with self.assertRaisesRegex(ValueError, "unknown samples"):
            self.run_plan(["Z"])
        (self.set / "A.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "run `python -m Tools.OCREval make`"):
            self.run_plan(["A"])
        for path in self.set.iterdir():
            path.unlink()
        with self.assertRaisesRegex(ValueError, "no ground truth"):
            self.run_plan()
        self.environment.assert_not_called()
        self.assertFalse(self.lock.parent.exists())
        self.assertEqual(self.calls, [])


class ChildTests(unittest.TestCase):
    def test_a_hung_run_is_killed_and_reported(self):
        class Hung:
            pid = 4242

            def __init__(self, *args, **kwargs):
                self.waits = 0

            def wait(self, timeout=None):
                self.waits += 1
                if self.waits == 1:
                    raise subprocess.TimeoutExpired("run", timeout)
                return -9

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(reference.subprocess, "Popen", Hung), \
                mock.patch.object(reference.os, "killpg") as kill:
            with self.assertRaisesRegex(RuntimeError, "A-text-bf16 timed out after 900s"):
                reference.child(Path(tmp), Path(tmp), Path(tmp), "A", "text", "A-text-bf16")
        kill.assert_called_once_with(4242, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
