import contextlib
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import venv

from Tools.Conversion.recipe import load_recipe, preflight
from Tools.Conversion.runner import run
from Tools.Conversion import runner
from Tools.Conversion.storage import PipelineError, directory_lock, read_json, write_json


WORKER = """
import json, pathlib, sys, time
source, out, counter, mode = map(str, sys.argv[1:])
out = pathlib.Path(out)
counter = pathlib.Path(counter)
counter.write_text(counter.read_text() + 'run\\n' if counter.exists() else 'run\\n')
value = json.loads(pathlib.Path(source).read_text())['value']
(out / 'result.json').write_text(json.dumps({'value': value, 'passed': value > 0}))
if mode == 'fail':
    sys.exit(17)
if mode == 'slow':
    time.sleep(60)
if mode == 'mutate':
    pathlib.Path(source).write_text(json.dumps({'value': value + 1}))
if mode == 'symlink':
    (out / 'result.json').unlink()
    (out / 'result.json').symlink_to(source)
"""


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.worker = self.root / "worker.py"
        self.worker.write_text(WORKER)
        self.source = self.root / "source.json"
        write_json(self.source, {"value": 1})
        self.work = self.root / "cache"
        self.recipe_path = self.root / "recipe.json"
        self.counter = self.root / "count"
        self.data = {
            "version": 1, "name": "test_recipe",
            "variables": {"python": sys.executable, "source": str(self.source)},
            "steps": [{"id": "reference", "role": "reference",
                       "command": ["$python", str(self.worker), "$source", "$out", str(self.counter), "ok"],
                       "inputs": [str(self.worker), "$source"], "outputs": ["result.json"],
                       "checks": [{"file": "result.json", "key": ["passed"], "equals": True}]}],
        }

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        write_json(self.recipe_path, self.data)
        return load_recipe(self.recipe_path, {}, self.root)

    def execute(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return read_json(run(self.load(), self.work, **kwargs))

    def count(self):
        return len(self.counter.read_text().splitlines()) if self.counter.exists() else 0

    def add_verify(self):
        self.data["steps"].append({
            "id": "verify", "role": "verify", "depends": ["reference"],
            "command": ["$python", str(self.worker), "$reference/result.json", "$out", str(self.counter), "ok"],
            "inputs": [str(self.worker)], "outputs": ["result.json"],
            "checks": [{"file": "result.json", "key": ["value"], "min": 1}],
        })

    def test_successful_outputs_are_reused_without_launching_worker(self):
        self.add_verify()
        first, second = self.execute(), self.execute()
        self.assertEqual(self.count(), 2)
        self.assertEqual([r["status"] for r in second["stages"]], ["reused", "reused"])
        self.assertEqual(second["verification"], "passed")
        self.assertEqual(second["stages"][0]["origin"], first["run_id"])
        self.assertEqual(first["stages"][0]["key"], second["stages"][0]["key"])

    def test_input_change_invalidates_reference_and_dependents(self):
        self.add_verify()
        self.execute()
        write_json(self.source, {"value": 2})
        changed = self.execute()
        self.assertEqual([s["status"] for s in changed["stages"]], ["executed", "executed"])
        self.assertEqual(self.count(), 4)

    def test_gate_change_does_not_recompute_reference(self):
        self.add_verify()
        self.execute()
        self.data["steps"][1]["checks"][0]["min"] = 0
        changed = self.execute()
        self.assertEqual([s["status"] for s in changed["stages"]], ["reused", "executed"])
        self.assertEqual(self.count(), 3)

    def test_corrupt_output_is_not_accepted_and_invalidates_changed_dependency(self):
        self.add_verify()
        first = self.execute()
        result = Path(first["stages"][0]["artifacts"]) / "result.json"
        result.write_text('{"value":999,"passed":true}')
        repaired = self.execute()
        self.assertEqual(repaired["stages"][0]["status"], "executed")
        self.assertEqual(read_json(Path(repaired["stages"][0]["artifacts"]) / "result.json")["value"], 1)
        self.assertEqual(len(list((self.work / "quarantine").iterdir())), 1)

    def test_same_size_same_mtime_input_modification_is_detected(self):
        first = self.execute()
        old = self.source.stat()
        self.source.write_text(self.source.read_text().replace("1", "2"))
        os.utime(self.source, ns=(old.st_atime_ns, old.st_mtime_ns))
        second = self.execute()
        self.assertNotEqual(first["stages"][0]["key"], second["stages"][0]["key"])

    def test_undeclared_artifact_files_are_also_checked(self):
        self.worker.write_text(WORKER + "\n(out / 'model.bin').write_bytes(b'original')\n")
        first = self.execute()
        (Path(first["stages"][0]["artifacts"]) / "model.bin").write_bytes(b"modified")
        second = self.execute()
        self.assertEqual(second["stages"][0]["status"], "executed")
        self.assertEqual((Path(second["stages"][0]["artifacts"]) / "model.bin").read_bytes(), b"original")

    def test_relative_executable_is_rejected_before_different_cwd_can_resolve_it(self):
        self.data["steps"][0]["command"][0] = "./worker"
        with self.assertRaisesRegex(PipelineError, "absolute executable"):
            preflight(self.load())
        self.assertEqual(self.count(), 0)

    def test_code_and_environment_changes_invalidate_cache(self):
        first = self.execute()
        self.worker.write_text(WORKER + "\nunused = 1\n")
        second = self.execute()
        self.data["environment"] = {"PIPELINE_TEST_MODE": "changed"}
        third = self.execute()
        self.assertEqual(len({x["stages"][0]["key"] for x in (first, second, third)}), 3)

    def test_nonzero_exit_and_partial_output_are_never_reused(self):
        self.data["steps"][0]["command"][-1] = "fail"
        for _ in range(2):
            with self.assertRaises(PipelineError):
                self.execute()
        self.assertEqual(self.count(), 2)
        self.assertFalse((self.work / "objects").exists())
        reports = list((self.work / "runs").glob("*/report.json"))
        self.assertEqual([read_json(p)["status"] for p in reports], ["failed", "failed"])

    def test_verification_failure_does_not_publish_manifest(self):
        write_json(self.source, {"value": -1})
        with self.assertRaisesRegex(PipelineError, "Verification failed"):
            self.execute()
        self.assertFalse((self.work / "objects").exists())

    def test_missing_required_output_and_symlinks_fail(self):
        self.data["steps"][0]["outputs"].append("missing.json")
        with self.assertRaises(PipelineError):
            self.execute()
        self.data["steps"][0]["outputs"].remove("missing.json")
        self.data["steps"][0]["command"][-1] = "symlink"
        with self.assertRaises(PipelineError):
            self.execute()
        self.assertFalse((self.work / "objects").exists())

    def test_input_changed_during_execution_is_not_cached(self):
        self.data["steps"][0]["command"][-1] = "mutate"
        with self.assertRaisesRegex(PipelineError, "Inputs or toolchain changed"):
            self.execute()
        self.assertFalse((self.work / "objects").exists())

    def test_timeout_leaves_no_complete_manifest_and_releases_locks(self):
        self.data["steps"][0].update(timeout_s=0.1, heavy=True)
        self.data["steps"][0]["command"][-1] = "slow"
        lock = self.root / "heavy-job"
        with self.assertRaises(subprocess.TimeoutExpired):
            self.execute(lock_dir=lock)
        self.assertFalse(lock.exists())
        self.assertFalse(list((self.work / "locks").iterdir()))
        self.assertFalse((self.work / "objects").exists())

    def test_existing_shared_lock_is_not_stolen(self):
        lock = self.root / "heavy-job"
        lock.mkdir()
        (lock / "owner.txt").write_text("another session")
        self.data["steps"][0]["heavy"] = True
        with self.assertRaisesRegex(PipelineError, "Busy lock"):
            self.execute(lock_dir=lock)
        self.assertEqual((lock / "owner.txt").read_text(), "another session")
        self.assertEqual(self.count(), 0)

    def test_failed_later_stage_resumes_successful_reference(self):
        self.add_verify()
        self.data["steps"][1]["checks"][0]["min"] = 2
        with self.assertRaises(PipelineError):
            self.execute()
        self.data["steps"][1]["checks"][0]["min"] = 1
        resumed = self.execute()
        self.assertEqual([s["status"] for s in resumed["stages"]], ["reused", "executed"])

    def test_invalid_recipes_fail_before_execution(self):
        original = copy.deepcopy(self.data)
        mutations = [
            lambda d: d.update(unknown=True),
            lambda d: d["steps"][0].update(depends=["future"]),
            lambda d: d["steps"][0].update(outputs=["../escape"]),
            lambda d: d["steps"][0].update(outputs=["/absolute"]),
            lambda d: d["steps"][0].update(command=["$missing"]),
            lambda d: d["steps"][0].update(timeout_s=0),
            lambda d: d["steps"][0].update(role="verify", checks=[]),
            lambda d: d["steps"][0].update(role="benchmark", cache=True),
            lambda d: d["variables"].update(out="reserved"),
            lambda d: d["variables"].update(a="$b", b="$a"),
        ]
        for mutation in mutations:
            self.data = copy.deepcopy(original)
            mutation(self.data)
            with self.subTest(recipe=self.data), self.assertRaises(PipelineError):
                self.load()
        self.assertEqual(self.count(), 0)

    def test_missing_input_is_rejected_in_preflight(self):
        self.source.unlink()
        with self.assertRaises(PipelineError):
            preflight(self.load())
        self.assertFalse(self.work.exists())

    def test_dangling_nested_input_prevents_all_stages(self):
        self.add_verify()
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "missing").symlink_to(self.root / "not_present")
        self.data["steps"][1]["inputs"].append(str(nested))
        with self.assertRaisesRegex(PipelineError, "dangling link"):
            self.execute()
        self.assertEqual(self.count(), 0)

    def test_empty_later_input_prevents_all_stages(self):
        self.add_verify()
        empty = self.root / "empty"
        empty.mkdir()
        self.data["steps"][1]["inputs"].append(str(empty))
        with self.assertRaisesRegex(PipelineError, "Empty source"):
            self.execute()
        self.assertEqual(self.count(), 0)

    def test_corrupt_manifest_is_a_cache_miss(self):
        for corruption in ([], {"version": 1}):
            first = self.execute()
            manifest = Path(first["stages"][0]["artifacts"]).parent / "manifest.json"
            write_json(manifest, corruption)
            self.assertEqual(self.execute()["stages"][0]["status"], "executed")

    def test_owner_write_failure_releases_own_empty_lock(self):
        lock = self.root / "empty_lock"
        with mock.patch("Tools.Conversion.storage.write_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError), directory_lock(lock, {"job": "test"}):
                pass
        self.assertFalse(lock.exists())

    def test_signal_immediately_after_mkdir_releases_own_lock(self):
        lock = self.root / "new_lock"
        mkdir = Path.mkdir

        def make(path, *args, **kwargs):
            result = mkdir(path, *args, **kwargs)
            if path == lock:
                os.kill(os.getpid(), signal.SIGTERM)
            return result

        def stop(_signum, _frame):
            raise KeyboardInterrupt("cancel during lock creation")

        previous = signal.signal(signal.SIGTERM, stop)
        try:
            with mock.patch.object(Path, "mkdir", make):
                with self.assertRaises(KeyboardInterrupt), directory_lock(lock, {"job": "test"}):
                    pass
            self.assertFalse(lock.exists())
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_venv_python_keeps_its_environment(self):
        environment = self.root / "venv"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        self.data["variables"]["python"] = str(environment / "bin/python")
        self.worker.write_text(WORKER + "\n(out / 'prefix.txt').write_text(sys.prefix)\n")
        self.data["steps"][0]["outputs"].append("prefix.txt")
        result = self.execute()
        self.assertEqual((Path(result["stages"][0]["artifacts"]) / "prefix.txt").read_text(), str(environment))

    def test_failed_verify_is_not_reported_as_not_run(self):
        self.add_verify()
        self.data["steps"][1]["checks"][0]["min"] = 2
        with self.assertRaises(PipelineError):
            self.execute()
        report = read_json(next((self.work / "runs").glob("*/report.json")))
        self.assertEqual(report["verification"], "failed")

    def test_successful_verification_survives_later_benchmark_failure(self):
        self.add_verify()
        benchmark = copy.deepcopy(self.data["steps"][0])
        benchmark.update(id="timing", role="benchmark", checks=[])
        benchmark["command"][-1] = "fail"
        self.data["steps"].append(benchmark)
        with self.assertRaises(PipelineError):
            self.execute()
        report = read_json(next((self.work / "runs").glob("*/report.json")))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["verification"], "passed")

    def test_refresh_preserves_previous_report_artifacts(self):
        first = self.execute()
        second = self.execute(refresh=True)
        self.assertNotEqual(first["stages"][0]["artifacts"], second["stages"][0]["artifacts"])
        self.assertTrue(Path(first["stages"][0]["artifacts"]).exists())
        self.assertTrue(Path(second["stages"][0]["artifacts"]).exists())
        self.assertFalse((Path(second["stages"][0]["artifacts"]).parent / "scratch").exists())

    def test_signal_during_popen_is_deferred_until_child_is_owned(self):
        children = []
        popen = subprocess.Popen

        def spawn(*args, **kwargs):
            child = popen(*args, **kwargs)
            children.append(child)
            os.kill(os.getpid(), signal.SIGTERM)
            return child

        def stop(_signum, _frame):
            raise KeyboardInterrupt("termination")

        previous = signal.signal(signal.SIGTERM, stop)
        try:
            with mock.patch("Tools.Conversion.runner.subprocess.Popen", side_effect=spawn):
                with self.assertRaises(KeyboardInterrupt):
                    runner.execute([sys.executable, "-c", "import time;time.sleep(60)"],
                                   str(self.root), dict(os.environ), self.root / "spawn.log", 30)
            self.assertIsNotNone(children[0].poll())
        finally:
            signal.signal(signal.SIGTERM, previous)
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    def test_plan_is_read_only_and_does_not_start_worker(self):
        self.load()
        result = subprocess.run(
            [sys.executable, "-m", "Tools.Conversion", "plan", str(self.recipe_path)],
            cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["recipe"], "test_recipe")
        self.assertFalse(self.work.exists())
        self.assertEqual(self.count(), 0)

    def test_benchmark_always_runs_and_reference_only_is_not_verified(self):
        self.data["steps"][0]["role"] = "benchmark"
        first, second = self.execute(), self.execute()
        self.assertEqual(second["stages"][0]["status"], "executed")
        self.assertEqual(second["verification"], "not_configured")
        self.assertEqual(self.count(), 2)

    def test_cancel_kills_descendants_and_second_run_refuses_active_cache_lock(self):
        pidfile = self.root / "child.pid"
        self.worker.write_text(WORKER + """
import subprocess, signal
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
pathlib.Path(sys.argv[3] + '.pid').write_text(str(child.pid))
time.sleep(60)
""")
        self.load()
        command = [sys.executable, "-m", "Tools.Conversion", "run", str(self.recipe_path),
                   "--work-dir", str(self.work)]
        cwd = Path(__file__).resolve().parents[3]
        with (self.root / "parent.log").open("w") as log:
            first = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=log)
            try:
                pidfile = Path(str(self.counter) + ".pid")
                deadline = time.monotonic() + 10
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(pidfile.exists())
                second = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=10)
                self.assertNotEqual(second.returncode, 0)
                self.assertIn("Busy lock", second.stderr)
                first.terminate()
                time.sleep(0.65)
                first.terminate()
                self.assertEqual(first.wait(timeout=10), 130)
                child = int(pidfile.read_text())
                deadline = time.monotonic() + 3
                state = ""
                while time.monotonic() < deadline:
                    observed = subprocess.run(["ps", "-o", "stat=", "-p", str(child)],
                                              capture_output=True, text=True)
                    state = observed.stdout.strip()
                    if not state or state.startswith("Z"):
                        break
                    time.sleep(0.05)
                self.assertTrue(not state or state.startswith("Z"), state)
                self.assertFalse((self.work / "objects").exists())
            finally:
                if first.poll() is None:
                    first.kill()
                    first.wait()


if __name__ == "__main__":
    unittest.main()
