import json
import os
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
        self.lock = self.root / "locks" / "heavy-job"
        samples = {
            "A": {"id": "A", "pipelines": ["text"], "size_px": [1000, 2000]},
            "B": {"id": "B", "pipelines": ["table", "text"], "table_bbox": [1, 2, 3, 4], "size_px": [1000, 2000]},
            "C": {"id": "C", "pipelines": ["two_stage"], "size_px": [1000, 2000]},
        }
        for sid, gt in samples.items():
            (self.set / f"{sid}.json").write_text(json.dumps(gt))
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def fake_child(self, set_dir, out, logs, sid, prompt_id, name, crop=None, resize=None):
        self.calls.append((sid, prompt_id, name, crop, resize))
        if prompt_id == "layout":
            (out / "outputs").mkdir(parents=True, exist_ok=True)
            (out / "outputs" / f"{name}.txt").write_text(
                "<box:100 050 500 100><label:title><up>\n<box:100 200 480 900><label:text><up>\n")
        if sid == "A":
            raise RuntimeError("A exited with 1")

    def run_plan(self, **kw):
        with mock.patch.dict(os.environ, {"COREAI_HEAVY_JOB_LOCK": str(self.lock)}), \
                mock.patch.object(reference, "child", self.fake_child), \
                mock.patch.object(reference, "environment", lambda: {"os": {"build": "test"}}), \
                mock.patch.object(reference, "model_processes", lambda: []):
            return reference.run_plan(self.set, self.out, self.root / "logs", None, "313-test", "test-session", 0, **kw)

    def test_runs_each_pipeline_in_order_and_releases_the_lock(self):
        failed = self.run_plan()
        self.assertEqual(failed, [{"sample": "A", "error": "A exited with 1"}])
        self.assertEqual(self.calls, [
            ("A", "text", "A-text-bf16", None, None),
            ("B", "table", "B-table-bf16", [1, 2, 3, 4], None),
            ("B", "text", "B-text-bf16", None, None),
            ("C", "layout", "C-layout-bf16", None, 1036),
            ("C", "text", "C-block01-text-bf16", [100, 100, 500, 200], None),
            ("C", "text", "C-block02-text-bf16", [100, 400, 480, 1800], None),
        ])
        self.assertFalse(self.lock.exists())
        done = list(self.lock.parent.glob("heavy-job-done-313-test-*"))
        self.assertEqual(len(done), 1)
        self.assertIn("session: test-session", (done[0] / "owner.txt").read_text())
        record = json.loads((self.out / "invocations.jsonl").read_text())
        self.assertEqual(record["samples"], ["A", "B", "C"])
        self.assertEqual(record["os"], {"build": "test"})

    def test_busy_lock_is_never_taken(self):
        self.lock.mkdir(parents=True)
        (self.lock / "owner.txt").write_text("someone else")
        with self.assertRaises(TimeoutError):
            self.run_plan()
        self.assertEqual((self.lock / "owner.txt").read_text(), "someone else")
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
