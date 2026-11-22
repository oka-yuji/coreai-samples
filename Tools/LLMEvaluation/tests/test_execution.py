import builtins
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from Tools.LLMEvaluation import execution
from Tools.LLMEvaluation.tests import test_contract

EMPTY = {"setup": "", "test": "", "call": ""}
PASSED = {"correct": True, "failure": None, "rerun": False}


def ran(code):
    return execution.run(execution.program(EMPTY, code))


def alive(token):
    for _ in range(50):
        if subprocess.run(["pgrep", "-f", token], stdout=subprocess.DEVNULL).returncode:
            return False
        time.sleep(0.1)
    return True


class ExtractionTests(unittest.TestCase):
    def test_the_code_follows_the_rules_of_the_contract(self):
        cases = {
            "a python block": ("Intro\n```python\ndef f():\n    return 1\n```\nDone.\n", "def f():\n    return 1\n"),
            "a block without a language": ("Intro\n```\nx = 1\n```\n", "x = 1\n"),
            "no block": ("x = 1\nprint(x)", "x = 1\nprint(x)"),
            "a block that is not closed": ("Intro\n```python\nx = 1\ny = 2", "x = 1\ny = 2"),
            "several blocks": ("```text\nnote\n```\n```python\nfirst\n```\n```python\nsecond\n```", "first\n"),
            "several blocks without python": ("```\nfirst\n```\n```text\nsecond\n```\n", "first\n"),
            "more words after python": ("```python title\nx = 1\n```", "x = 1\n"),
            "another language word": ("```Python\na\n```\n```python3\nb\n```\n```py\nc\n```\n```python\nd\n```", "d\n"),
            "an indented fence": ("  ```python\n  x = 1\n  ```\n", "  ```python\n  x = 1\n  ```\n"),
            "carriage returns": ("```python\r\nx = 1\r\n```\r\n", "x = 1\r\n"),
        }
        for name, (text, code) in cases.items():
            with self.subTest(name):
                self.assertEqual(execution.extract(text), code)


class IsolationTests(unittest.TestCase):
    def test_the_network_writes_outside_and_standard_input_fail_and_writes_inside_succeed(self):
        with tempfile.TemporaryDirectory() as folder, socket.create_server(("127.0.0.1", 0)) as server:
            outside = Path(folder).resolve() / "outside.txt"
            port = server.getsockname()[1]
            programs = {
                "network": f"import socket\n\ntry:\n    socket.create_connection(('127.0.0.1', {port}), timeout=5)\n"
                           "except OSError:\n    pass\nelse:\n    raise AssertionError('connected')\n",
                "write outside": f"try:\n    open({str(outside)!r}, 'w')\nexcept OSError:\n    pass\n"
                                 "else:\n    raise AssertionError('written')\n",
                "standard input": "try:\n    input()\nexcept EOFError:\n    pass\nelse:\n    raise AssertionError('read')\n",
                "write inside": "import os\n\nassert os.listdir() == []\nwith open('inside.txt', 'w') as handle:\n"
                                "    handle.write('kept')\nwith open('inside.txt') as handle:\n"
                                "    assert handle.read() == 'kept'\n",
                "child process": "import subprocess\nimport sys\n\n"
                                 "assert subprocess.run([sys.executable, '-c', 'pass']).returncode == 0\n",
                "interpreter": "import sys\n\nassert sys.flags.isolated and sys.flags.no_site\n"
                               f"assert sys.flags.dont_write_bytecode and sys.executable == {sys.executable!r}\n",
                "not in the scoring process": "import builtins\n\nbuiltins.ran_in_the_scorer = True\n",
            }
            typed, keys = os.pipe()
            os.write(keys, b"typed\n")
            os.close(keys)
            saved = os.dup(0)
            os.dup2(typed, 0)
            os.close(typed)
            try:
                for name, code in programs.items():
                    with self.subTest(name):
                        self.assertEqual(ran(code), PASSED)
            finally:
                os.dup2(saved, 0)
                os.close(saved)
            server.setblocking(False)
            with self.assertRaises(BlockingIOError):
                server.accept()
            self.assertFalse(outside.exists())
            self.assertFalse(hasattr(builtins, "ran_in_the_scorer"))


class FailureTests(unittest.TestCase):
    def test_the_failure_tells_how_the_program_ended_and_a_large_output_does_not_stop_it(self):
        cases = {"an exception": ("raise ValueError\n", "not_finished"),
                 "an early end": ("import sys\n\nsys.exit(0)\n", "not_finished"),
                 "another status after the end": ("import atexit\nimport os\n\natexit.register(os._exit, 2)\n",
                                                  "exit_status")}
        for name, (code, failure) in cases.items():
            with self.subTest(name):
                self.assertEqual(ran(code), {"correct": False, "failure": failure, "rerun": False})
        self.assertEqual(ran("import sys\n\nfor _ in range(200):\n    print('x' * 100000)\n"
                             "    print('x' * 100000, file=sys.stderr)\n"), PASSED)


class TimeLimitTests(unittest.TestCase):
    def test_a_program_over_the_limit_runs_once_more_and_leaves_no_process_or_folder(self):
        token = secrets.token_hex(16)
        self.addCleanup(subprocess.run, ["pkill", "-f", token])
        spawn = ("import subprocess\nimport sys\nimport time\n\n"
                 f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', '{token}'])\n")
        cases = {"over the limit": (spawn + "time.sleep(6)\n",
                                    {"correct": False, "failure": "timeout", "rerun": True}, 2),
                 "within the limit": (spawn, PASSED, 1)}
        for name, (code, expected, attempts) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as folder:
                with mock.patch.object(tempfile, "tempdir", folder), mock.patch.object(execution, "LIMIT", 1.5), \
                        mock.patch.object(execution, "attempt", wraps=execution.attempt) as attempt:
                    self.assertEqual(ran(code), expected)
                self.assertEqual(attempt.call_count, attempts)
                self.assertEqual(os.listdir(folder), [])
                self.assertFalse(alive(token))

    def test_the_result_of_the_item_is_the_result_of_the_rerun(self):
        cases = {"passes on the rerun": (["timeout", "passed"], {"correct": True, "failure": None, "rerun": True}),
                 "fails on the rerun": (["timeout", "not_finished"],
                                        {"correct": False, "failure": "not_finished", "rerun": True})}
        for name, (attempts, expected) in cases.items():
            with self.subTest(name), mock.patch.object(execution, "attempt", side_effect=attempts) as attempt:
                self.assertEqual(ran(""), expected)
                self.assertEqual(attempt.call_count, len(attempts))


class ContractExampleTests(unittest.TestCase):
    def test_every_code_example_of_the_contract_gives_the_same_result(self):
        def scored(scorer, code):
            return execution.run(execution.program(scorer, code))["correct"]

        for name in ("test_code_is_correct_only_when_the_tests_ran_to_the_end",
                     "test_humaneval_program_starts_with_the_problem_prompt"):
            with self.subTest(name), mock.patch.object(test_contract, "scored", scored):
                result = unittest.TestResult()
                test_contract.ContractTests(name).run(result)
                self.assertEqual((result.testsRun, result.failures, result.errors), (1, [], []))


if __name__ == "__main__":
    unittest.main()
