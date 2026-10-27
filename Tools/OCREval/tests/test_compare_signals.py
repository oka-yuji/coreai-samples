import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Tools.OCREval import compare
from Tools.OCREval.reference import sha256
from Tools.Conversion import runner

def probe(ignored):
    real_popen=subprocess.Popen
    workers=[]
    prior=signal.signal(signal.SIGHUP, signal.SIG_IGN if ignored else signal.SIG_DFL)
    result={}
    try:
        with tempfile.TemporaryDirectory(prefix="teleocr333-hup-cpu-") as tmp:
            root=Path(tmp);(root/"cli").write_text("fake");(root/"bundle").mkdir()
            plan={"cli":str(root/"cli"),"cli_sha256":sha256(root/"cli"),"sets":{"synthetic":str(root)},"comparisons":[],"variants":[{"id":"A","engine":"coreai","bundle":str(root/"bundle"),"asset_hashes":{}}],"cache_root":"unused"}
            (root/"plan.json").write_text(json.dumps(plan));lock=root/"lock"
            def spawn_then_hup(*args,**kwargs):
                child=real_popen([sys.executable,"-c","import time; time.sleep(" + ("0.01" if ignored else "60") + ")"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
                workers.append(child)
                os.kill(os.getpid(),signal.SIGHUP)
                return child
            with mock.patch.dict(os.environ,{"COREAI_HEAVY_JOB_LOCK":str(lock)}),mock.patch.object(compare.reference,"model_processes",return_value=[]),mock.patch.object(compare.reference,"environment",return_value={"machine":{"cpu":"fake"},"os":{"build":"fake"}}),mock.patch.object(compare.reference,"command",return_value="fake"),mock.patch.object(compare,"prepared_inputs",return_value={"synthetic":{"A":{"image":"unused.png"}}}),mock.patch.object(runner.subprocess,"Popen",side_effect=spawn_then_hup):
                caught=None
                try: compare.run_plan(root/"plan.json",root/"out",performance=False)
                except BaseException as error: caught=f"{type(error).__name__}: {error}"
            result={"ignored_preserved": signal.getsignal(signal.SIGHUP) is signal.SIG_IGN,"scope":"CPU sleeper only; actual SIGHUP sent to this isolated probe after OS child spawn and before Popen handle reaches runner.execute", "caught":caught,"child_count":len(workers),"child_alive_after_run_plan":workers[0].poll() is None,"lock_exists_after_run_plan":lock.exists(),"archives":len(list(root.glob("heavy-job-done-333-*"))),"runs":json.loads((root/"out/runs.json").read_text()),"invocation":json.loads((root/"out/invocation.json").read_text())}
    finally:
        signal.signal(signal.SIGHUP,prior)
        for child in workers:
            if child.poll() is None: os.killpg(child.pid,signal.SIGKILL)
            child.wait(timeout=5)
        result["owned_cpu_child_cleaned_up_by_probe"]=all(child.poll() is not None for child in workers)
    return result


class HangupTests(unittest.TestCase):
    def test_hangup_during_spawn_cleans_the_child_and_preserves_ignored_hangups(self):
        root = Path(__file__).resolve().parents[3]
        for ignored in (False, True):
            with self.subTest(ignored=ignored):
                command = [sys.executable, str(Path(__file__).resolve()), "--probe", str(int(ignored))]
                result = subprocess.run(command, cwd=root, env=os.environ | {"PYTHONPATH": str(root)}, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                row = json.loads(result.stdout.splitlines()[-1])
                self.assertFalse(row["child_alive_after_run_plan"])
                self.assertFalse(row["lock_exists_after_run_plan"])
                self.assertEqual(row["child_count"], 1)
                self.assertEqual(row["archives"], 1)
                self.assertEqual(row["ignored_preserved"], ignored)
                self.assertEqual(len(row["runs"]), 1)
                if not ignored:
                    self.assertIn("KeyboardInterrupt", row["caught"])
                    self.assertEqual(row["runs"][0]["status"], "failed")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        print(json.dumps(probe(sys.argv[-1] == "1")))
    else:
        unittest.main()
