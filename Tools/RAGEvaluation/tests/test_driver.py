import json
from pathlib import Path
import tempfile
import unittest

from Tools.RAGEvaluation.__main__ import initial_result, run_cases, summarize
from Tools.Conversion.storage import ProcessCleanupError


class DriverTests(unittest.TestCase):
    def plan(self):
        return {"cases": ["one", "two", "three"], "manifestHash": "hash", "strategy": "flat",
                "mode": "work", "manifest": "manifest.json", "settings": {"deadlineSeconds": 1}}

    def test_missing_and_interrupted_results_remain_in_denominator(self):
        with tempfile.TemporaryDirectory() as root:
            plan = self.plan()
            for index, status in enumerate(["completed", "running"]):
                directory = Path(root) / f"case-{index:03}"
                directory.mkdir()
                (directory / "result.json").write_text(json.dumps(initial_result(plan, plan["cases"][index], status, "")))
                (Path(root) / f"case-{index:03}.control.json").write_text(json.dumps({
                    "caseID": plan["cases"][index], "driverState": "settled" if status == "completed" else "attempted"}))
            result = summarize(plan, root)
            self.assertEqual(result["requested"], 3)
            self.assertEqual(result["counts"]["completed"], 1)
            self.assertEqual(result["counts"]["running"], 1)
            self.assertEqual(result["counts"]["notRun"], 1)
            self.assertEqual(result["factualSupport"], "ungraded")

    def test_wrong_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "case-000"
            directory.mkdir()
            row = initial_result(self.plan(), "one", "completed", "")
            row["mode"] = "chat"
            (directory / "result.json").write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                summarize(self.plan(), root)

    def test_missing_control_does_not_promote_unconfirmed_native_completion(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "case-000"
            directory.mkdir()
            (directory / "result.json").write_text(json.dumps(initial_result(self.plan(), "one", "completed", "")))
            result = summarize(self.plan(), root)
            self.assertEqual(result["counts"]["completed"], 0)
            self.assertEqual(result["counts"]["running"], 1)
            self.assertEqual(result["rows"][0]["nativeStatus"], "completed")

    def test_child_failure_is_recorded_and_does_not_drop_later_cases(self):
        def fail(*args):
            raise RuntimeError("fixture child failed")
        with tempfile.TemporaryDirectory() as root:
            run_cases(self.plan(), root, Path("app"), Path("model"), executor=fail)
            result = summarize(self.plan(), root)
            self.assertEqual(result["counts"]["failed"], 3)
            self.assertEqual(result["requested"], 3)

    def test_interrupt_keeps_unexecuted_cases(self):
        def interrupt(*args):
            raise KeyboardInterrupt("fixture stop")
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(KeyboardInterrupt):
                run_cases(self.plan(), root, Path("app"), Path("model"), executor=interrupt)
            result = summarize(self.plan(), root)
            self.assertEqual(result["counts"]["cancelled"], 1)
            self.assertEqual(result["counts"]["notRun"], 2)

    def test_success_without_terminal_record_is_a_failure(self):
        with tempfile.TemporaryDirectory() as root:
            run_cases(self.plan(), root, Path("app"), Path("model"), executor=lambda *args: None)
            self.assertEqual(summarize(self.plan(), root)["counts"]["failed"], 3)


    def test_native_deadline_cancellation_is_not_rewritten_as_failure(self):
        plan = self.plan()
        def cancel(command, *args):
            directory = Path(command[command.index("--output") + 1])
            directory.mkdir()
            case_id = command[command.index("--case") + 1]
            (directory / "result.json").write_text(json.dumps(initial_result(plan, case_id, "cancelled", "deadline")))
            raise RuntimeError("exit 1")
        with tempfile.TemporaryDirectory() as root:
            run_cases(plan, root, Path("app"), Path("model"), executor=cancel)
            counts = summarize(plan, root)["counts"]
            self.assertEqual(counts["cancelled"], 3)
            self.assertEqual(counts["failed"], 0)

    def test_unconfirmed_cleanup_is_neither_completed_nor_unattempted(self):
        for completed in [False, True]:
            plan = self.plan()
            def uncertain(command, *args):
                if completed:
                    directory = Path(command[command.index("--output") + 1])
                    directory.mkdir()
                    row = initial_result(plan, "one", "completed", "")
                    (directory / "result.json").write_text(json.dumps(row))
                raise ProcessCleanupError(RuntimeError("stop"), 123, [RuntimeError("alive")], None)
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ProcessCleanupError):
                    run_cases(plan, root, Path("app"), Path("model"), executor=uncertain)
                counts = summarize(plan, root)["counts"]
                self.assertEqual(counts["running"], 1)
                self.assertEqual(counts["completed"], 0)
                self.assertEqual(counts["notRun"], 2)


if __name__ == "__main__":
    unittest.main()
