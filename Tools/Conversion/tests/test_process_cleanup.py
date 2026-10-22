import errno
import inspect
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from Tools.Conversion import runner
from Tools.Conversion.storage import PipelineError, ProcessCleanupError, blocked_signals, directory_lock, read_json


class ProcessCleanupTests(unittest.TestCase):
    def test_interruption_at_cleanup_entry_never_releases_an_unconfirmed_group(self):
        program = (
            "import os,pathlib,subprocess,sys; "
            "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)']); "
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); os._exit(int(sys.argv[2]))"
        )
        for code in (0, 7):
            for forced in (False, True):
                with self.subTest(code=code, forced=forced), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    pidfile, lock = root / "descendant.pid", root / "lock"
                    groups = []
                    original = runner.cleanup_process

                    def interrupted(child, primary, state):
                        groups.append(child.pid)
                        if forced:
                            raise KeyboardInterrupt("interrupted before cleanup")
                        runner.os.kill(runner.os.getpid(), signal.SIGTERM)
                        return original(child, primary, state)

                    try:
                        with patch.object(runner, "cleanup_process", side_effect=interrupted):
                            with self.assertRaises(BaseException) as caught:
                                with directory_lock(lock, {}):
                                    runner.execute([sys.executable, "-c", program, str(pidfile), str(code)], root,
                                                   dict(runner.os.environ), root / "process.log", 3)
                        descendant = int(pidfile.read_text())
                        status = subprocess.run(["/bin/ps", "-o", "stat=", "-p", str(descendant)],
                                                capture_output=True, text=True, timeout=2).stdout.strip()
                        if forced:
                            self.assertIsInstance(caught.exception, ProcessCleanupError)
                            self.assertTrue(lock.exists())
                            self.assertFalse(read_json(lock / "owner.txt")["cleanup"]["confirmed"])
                        else:
                            self.assertIsInstance(caught.exception, KeyboardInterrupt if code == 0 else PipelineError)
                            self.assertFalse(isinstance(caught.exception, ProcessCleanupError))
                            self.assertFalse(lock.exists())
                            self.assertTrue(not status or status.startswith("Z"), status)
                    finally:
                        for group in groups:
                            try:
                                runner.os.killpg(group, signal.SIGKILL)
                            except (ProcessLookupError, PermissionError):
                                pass
                        if pidfile.exists():
                            descendant = int(pidfile.read_text())
                            deadline = time.monotonic() + 3
                            while True:
                                status = subprocess.run(["/bin/ps", "-o", "stat=", "-p", str(descendant)],
                                                        capture_output=True, text=True, timeout=2).stdout.strip()
                                if not status or status.startswith("Z"):
                                    break
                                if time.monotonic() > deadline:
                                    self.fail("The test did not stop its own descendant")
                                time.sleep(0.01)

    def execute(self, folder, code=0):
        runner.execute([sys.executable, "-c", f"raise SystemExit({code})"], folder,
                       dict(runner.os.environ), Path(folder) / "process.log", 10)

    def test_a_completed_process_is_not_failed_by_a_zombie_only_group(self):
        with tempfile.TemporaryDirectory() as folder:
            def observe(command, **kwargs):
                self.assertEqual(command, ["/bin/ps", "-ax", "-o", "pgid=,stat="])
                return subprocess.CompletedProcess(command, 0, f"{group[0]} Z\n0 Ss\n", "")

            group = []

            def refused(pid, sig):
                group[:] = [pid]
                raise PermissionError(errno.EPERM, "Operation not permitted")

            with patch.object(runner.os, "killpg", side_effect=refused), patch.object(runner.subprocess, "run", side_effect=observe):
                self.execute(folder)

    def test_live_or_unobservable_groups_still_fail_cleanup(self):
        for observation in ("live", "empty", "malformed", "failed"):
            with self.subTest(observation=observation), tempfile.TemporaryDirectory() as folder:
                group = []

                def refused(pid, sig):
                    group[:] = [pid]
                    raise PermissionError(errno.EPERM, "Operation not permitted")

                def observe(command, **kwargs):
                    if observation == "failed":
                        raise subprocess.TimeoutExpired(command, 5)
                    output = f"{group[0]} S\n" if observation == "live" else "" if observation == "empty" else "invalid"
                    return subprocess.CompletedProcess(command, 0, output, "")

                with patch.object(runner.os, "killpg", side_effect=refused), patch.object(runner.subprocess, "run", side_effect=observe):
                    with self.assertRaises(ProcessCleanupError) as caught:
                        self.execute(folder)
                    self.assertFalse(caught.exception.details["confirmed"])

    def test_cleanup_failure_preserves_the_original_command_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(runner.os, "killpg", side_effect=PermissionError(errno.EPERM, "permission denied")), patch.object(
                    runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("ps", 5)):
                with self.assertRaisesRegex(ProcessCleanupError, "Command exited 7") as caught:
                    self.execute(folder, 7)
            self.assertIsInstance(caught.exception.primary, PipelineError)
            self.assertEqual(caught.exception.details["exit_code"], 7)
            self.assertTrue(caught.exception.details["errors"])
            self.assertIn(f"process group {caught.exception.details['process_group']}", str(caught.exception))
            self.assertIn("child exit 7", str(caught.exception))

    def test_nested_locks_survive_repeated_interruptions_and_retention_record_failure(self):
        from Tools.Conversion import storage

        def stop(signum, frame):
            raise KeyboardInterrupt("repeated cancellation")

        previous = signal.signal(signal.SIGTERM, stop)
        try:
            for write_fails in (False, True):
                with self.subTest(write_fails=write_fails), tempfile.TemporaryDirectory() as folder:
                    locks = [Path(folder) / name for name in ("shared", "work", "stage")]
                    failure = ProcessCleanupError(KeyboardInterrupt("initial cancellation"), 123,
                                                  [PermissionError(errno.EPERM, "permission denied")], None)
                    original = storage.write_json

                    def write(path, value):
                        if write_fails and value.get("retained"):
                            raise OSError(errno.ENOSPC, "no space for retained owner record")
                        return original(path, value)

                    with patch.object(storage, "write_json", side_effect=write):
                        with self.assertRaises(ProcessCleanupError) as caught:
                            with directory_lock(locks[0], {}), directory_lock(locks[1], {}), directory_lock(locks[2], {}):
                                original_tokens = {lock: read_json(lock / "owner.txt")["token"] for lock in locks}
                                with blocked_signals():
                                    runner.os.kill(runner.os.getpid(), signal.SIGTERM)
                                    raise failure
                    self.assertIs(caught.exception, failure)
                    self.assertIsInstance(failure.primary, KeyboardInterrupt)
                    self.assertTrue(any("repeated cancellation" in note for note in failure.__notes__))
                    for lock in locks:
                        self.assertTrue(lock.is_dir())
                        owner = read_json(lock / "owner.txt")
                        self.assertEqual(owner["token"], original_tokens[lock])
                        if owner.get("retained"):
                            self.assertTrue(owner["retained"])
                            self.assertEqual(owner["cleanup"]["process_group"], 123)
                        else:
                            self.assertTrue(any(f"Could not record retained lock {lock}:" in note for note in failure.__notes__))
                        with self.assertRaisesRegex(PipelineError, "Busy lock"):
                            with directory_lock(lock, {}):
                                self.fail("An unconfirmed cleanup released its lock")
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_interruption_after_os_mask_change_preserves_each_lock_and_restores_the_mask(self):
        from Tools.Conversion import storage

        native_mask = signal.pthread_sigmask
        previous = native_mask(signal.SIG_BLOCK, set())
        ready = injected = False

        def interrupt(how, mask):
            nonlocal injected
            result = native_mask(how, mask)
            if ready and not injected and how == signal.SIG_BLOCK and set(mask) == {signal.SIGINT, signal.SIGTERM}:
                injected = True
                raise KeyboardInterrupt("interrupted after the native mask change")
            return result

        try:
            with tempfile.TemporaryDirectory() as folder:
                locks = [Path(folder) / name for name in ("shared", "work", "stage")]
                failure = ProcessCleanupError(PipelineError("pending cleanup"), 123, [PermissionError()], None)
                with patch.object(storage.signal, "pthread_sigmask", side_effect=interrupt):
                    with self.assertRaises(ProcessCleanupError) as caught:
                        with directory_lock(locks[0], {}), directory_lock(locks[1], {}), directory_lock(locks[2], {}):
                            tokens = {lock: read_json(lock / "owner.txt")["token"] for lock in locks}
                            ready = True
                            raise failure
                self.assertTrue(injected)
                self.assertIs(caught.exception, failure)
                self.assertEqual(native_mask(signal.SIG_BLOCK, set()), previous)
                for lock in locks:
                    owner = read_json(lock / "owner.txt")
                    self.assertEqual(owner["token"], tokens[lock])
                    self.assertTrue(owner.get("retained") or any(
                        f"Could not record retained lock {lock}:" in note for note in failure.__notes__))
                    with self.assertRaisesRegex(PipelineError, "Busy lock"):
                        with directory_lock(lock, {}):
                            self.fail("A cleanup interruption released its lock")
        finally:
            native_mask(signal.SIG_SETMASK, previous)

    def test_cleanup_retention_cannot_overwrite_a_replacement_lock_owner(self):
        from Tools.Conversion.storage import write_json

        for copied_token in (False, True):
            with self.subTest(copied_token=copied_token), tempfile.TemporaryDirectory() as folder:
                lock, previous = Path(folder) / "lock", Path(folder) / "previous-lock"
                failure = ProcessCleanupError(PipelineError("original failure"), 123, [PermissionError()], 7)
                with self.assertRaises(ProcessCleanupError) as caught:
                    with directory_lock(lock, {"job": "original"}):
                        original = read_json(lock / "owner.txt")
                        lock.rename(previous)
                        lock.mkdir()
                        replacement = {"token": original["token"] if copied_token else "another-token", "job": "replacement"}
                        write_json(lock / "owner.txt", replacement)
                        raise failure
                self.assertIs(caught.exception, failure)
                self.assertEqual(read_json(lock / "owner.txt"), replacement)
                self.assertEqual(read_json(previous / "owner.txt"), original)
                self.assertTrue(any("ownership changed" in note for note in failure.__notes__))

    def test_additional_sigterm_cannot_erase_cleanup_during_lock_propagation(self):
        from Tools.Conversion import storage

        source, first = inspect.getsourcelines(storage.directory_lock.__wrapped__)
        propagation = source[:next(index for index, line in enumerate(source) if line.startswith("    finally:"))]
        locations = [first + index for index, line in enumerate(propagation)
                     if line.strip() in ("retained = cleanup_failure(error)", "raise retained")]
        self.assertEqual(len(locations), 2)

        def stop(signum, frame):
            raise KeyboardInterrupt("signal during cleanup propagation")

        previous = signal.signal(signal.SIGTERM, stop)
        try:
            for location in locations:
                with self.subTest(location=location), tempfile.TemporaryDirectory() as folder:
                    locks = [Path(folder) / name for name in ("work", "shared", "stage")]
                    failure = ProcessCleanupError(PipelineError("original command failure"), 123,
                                                  [PermissionError(errno.EPERM, "permission denied")], 7)
                    fired = []

                    def trace(frame, event, argument):
                        if event == "line" and frame.f_code.co_filename == storage.__file__\
                                and frame.f_lineno == location and not fired:
                            fired.append(True)
                            runner.os.kill(runner.os.getpid(), signal.SIGTERM)
                        return trace

                    try:
                        sys.settrace(trace)
                        with self.assertRaises(ProcessCleanupError) as caught:
                            with directory_lock(locks[0], {}), directory_lock(locks[1], {}), directory_lock(locks[2], {}):
                                raise failure
                    finally:
                        sys.settrace(None)
                    self.assertEqual(fired, [True])
                    self.assertIs(caught.exception, failure)
                    self.assertTrue(all(lock.is_dir() for lock in locks))
                    self.assertTrue(all(read_json(lock / "owner.txt")["retained"] for lock in locks))
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_pipeline_records_primary_cancellation_and_retains_all_locks(self):
        from Tools.Conversion.tests import test_pipeline

        fixture = test_pipeline.PipelineTests()
        fixture.setUp()
        try:
            fixture.data["steps"][0]["heavy"] = True
            shared = fixture.root / "shared.lock"
            failure = ProcessCleanupError(KeyboardInterrupt("initial cancellation"), 123,
                                          [PermissionError(errno.EPERM, "permission denied")], 0)
            with patch.object(runner, "cleanup_process", side_effect=failure):
                with self.assertRaises(ProcessCleanupError):
                    fixture.execute(lock_dir=shared)
            reports = list((fixture.work / "runs").glob("*/report.json"))
            self.assertEqual(len(reports), 1)
            report = read_json(reports[0])
            self.assertEqual(report["status"], "cancelled")
            self.assertIn("initial cancellation", report["error"])
            self.assertEqual(report["cleanup"], failure.details)
            self.assertTrue(shared.is_dir())
            self.assertTrue((fixture.work / "run.lock").is_dir())
            self.assertEqual(len(list((fixture.work / "locks").iterdir())), 1)
            self.assertFalse((fixture.work / "objects").exists())
        finally:
            fixture.tearDown()

    def test_terminal_report_failure_cannot_release_locks_after_unknown_cleanup(self):
        from Tools.Conversion.tests import test_pipeline

        fixture = test_pipeline.PipelineTests()
        fixture.setUp()
        try:
            fixture.data["steps"][0]["heavy"] = True
            shared = fixture.root / "shared.lock"
            failure = ProcessCleanupError(PipelineError("initial failure"), 123,
                                          [PermissionError(errno.EPERM, "permission denied")], 7)
            original = runner.save_report

            def save(path, value):
                if "cleanup" in value:
                    raise OSError(errno.ENOSPC, "no space for terminal report")
                original(path, value)

            with patch.object(runner, "cleanup_process", side_effect=failure), patch.object(runner, "save_report", side_effect=save):
                with self.assertRaises(ProcessCleanupError) as caught:
                    fixture.execute(lock_dir=shared)
            self.assertIs(caught.exception, failure)
            self.assertTrue(shared.is_dir())
            self.assertTrue((fixture.work / "run.lock").is_dir())
            self.assertTrue(any("terminal report" in note for note in failure.__notes__))
        finally:
            fixture.tearDown()

    def test_model_fingerprints_and_cached_reuse_remain_under_the_machine_lock(self):
        from Tools.Conversion.tests import test_pipeline

        fixture = test_pipeline.PipelineTests()
        fixture.setUp()
        try:
            fixture.data["steps"][0]["heavy"] = True
            shared = fixture.root / "shared.lock"
            original_host, original_inventory = runner.host_identity, runner.inventory
            observations = []

            def host(*args):
                observations.append(shared.is_dir())
                return original_host(*args)

            def inventory(*args):
                observations.append(shared.is_dir())
                return original_inventory(*args)

            with patch.object(runner, "host_identity", side_effect=host), patch.object(runner, "inventory", side_effect=inventory):
                fixture.execute(lock_dir=shared)
                cached = fixture.execute(lock_dir=shared)
            self.assertTrue(observations and all(observations))
            self.assertEqual(cached["stages"][0]["status"], "reused")
            self.assertFalse(shared.exists())
            with directory_lock(shared, {"owner": "another job"}):
                with patch.object(runner, "host_identity") as probe:
                    with self.assertRaisesRegex(PipelineError, "Busy lock"):
                        fixture.execute(lock_dir=shared)
                    probe.assert_not_called()
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
