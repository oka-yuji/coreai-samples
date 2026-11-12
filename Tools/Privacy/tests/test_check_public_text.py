import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "check-public-text.py"
REPOSITORY = "octo/demo"
FAKE_GH = """#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

fake = json.loads(Path(__file__).with_name("fake.json").read_text())
marker = Path(__file__).with_name("unblocked")
if fake.get("block_first") and not marker.exists():
    sys.stdin.buffer.read(1)
    marker.touch()
if "output" in fake:
    sys.stdout.write(fake["output"])
    sys.exit(fake["code"])
if sys.argv[1:] == ["api", "graphql", "--input", "-"]:
    nodes = [node for node in re.findall(r'"([^"]*)"', json.load(sys.stdin)["query"]) if node not in fake.get("missing", [])]
    empty = {"userContentEdits": {"pageInfo": {"hasNextPage": False}, "nodes": []}}
    print(json.dumps({"data": {"nodes": [fake["edits"].get(node, empty) for node in nodes]}}))
else:
    print(json.dumps(fake["pages"][" ".join(sys.argv[1:])]))
"""


def absolute(*parts):
    return "/".join(["", *parts])


def endpoint(path):
    return f"api --paginate --slurp repos/{REPOSITORY}/{path}"


def edits(*versions, more=False):
    nodes = [{"diff": diff, "deletedAt": deleted} for diff, deleted in versions]
    return {"userContentEdits": {"pageInfo": {"hasNextPage": more}, "nodes": nodes}}


def issue(number, body):
    return {"number": number, "node_id": f"I{number}", "title": "Notes", "body": body}


def repository(*pages, history=None):
    return {
        "pages": {
            endpoint("issues?state=all&per_page=100"): list(pages),
            endpoint("issues/comments?per_page=100"): [[]],
            endpoint("pulls/comments?per_page=100"): [[]],
        },
        "edits": history or {},
    }


class PublicTextChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="public text checks ")
        self.addCleanup(temporary.cleanup)
        self.bin = Path(temporary.name)
        gh = self.bin / "gh"
        gh.write_text(FAKE_GH, encoding="utf-8")
        gh.chmod(0o755)

    def run_check(self, fake):
        (self.bin / "fake.json").write_text(json.dumps(fake), encoding="utf-8")
        environment = dict(os.environ, PATH=os.pathsep.join([str(self.bin), os.environ.get("PATH", "")]))
        return subprocess.run([sys.executable, "-I", str(SCRIPT), REPOSITORY], capture_output=True, text=True, env=environment)

    def test_paths_in_titles_bodies_comments_reviews_and_edits_are_reported_by_place_and_line_only(self):
        home = absolute("Users", "alice", "work", "run.log")
        volume = absolute("Volumes", "Archive", "models")
        temporary = absolute("private", "var", "folders", "xy", "T", "out")
        linux = absolute("home", "runner", "work")
        session = "-".join(["", "Users", "alice", "Desktop", "app"])
        escaped = "\\/".join(["", "Users", "alice", "model"])
        result = self.run_check({
            "pages": {
                endpoint("issues?state=all&per_page=100"): [
                    [
                        {"number": 1, "node_id": "I1", "title": "Plain", "body": None},
                        {"number": 2, "node_id": "I2", "title": f"Log at {home}", "body": f"intro\n{volume}\n"},
                    ],
                    [{"number": 3, "node_id": "P3", "title": "Fix", "body": f"a\nb\n{temporary}", "pull_request": {}}],
                ],
                endpoint("pulls/3/reviews?per_page=100"): [
                    [{"id": 31, "node_id": "R31", "body": ""}, {"id": 32, "node_id": "R32", "body": f"ok\n{linux}"}],
                ],
                endpoint("issues/comments?per_page=100"): [[{"id": 41, "node_id": "C41", "body": f"cache {session}"}]],
                endpoint("pulls/comments?per_page=100"): [[{"id": 51, "node_id": "D51", "body": f"x\ny\n{escaped}"}]],
            },
            "edits": {
                "I2": edits((f"intro\n{volume}\n", None), (f"draft {home}", None)),
                "R31": edits(("", None), (f"first\n{home}", None)),
                "C41": edits((f"cache {session}", None)),
            },
        })
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "issue #2 title:1",
                "issue #2 body:2",
                "issue #2 body:2 (edit 1)",
                "issue #2 body:1 (edit 2)",
                "pr #3 body:3",
                "comment 41:1",
                "comment 41:1 (edit 1)",
                "review 31:2 (edit 2)",
                "review 32:2",
                "review-comment 51:3",
            ],
        )
        self.assertIn("10 lines contain an absolute local path.", result.stderr)
        for text in ("alice", "Archive", "folders", "run.log", "runner"):
            self.assertNotIn(text, result.stdout + result.stderr)

    def test_deleted_edits_are_not_read_and_a_repository_without_paths_passes(self):
        body = f"Use ~/Library/Caches/app or docs/media/demo.gif and `/Volumes/`.\nFiles: {absolute('Users', 'runner', 'work', 'lib')}\n"
        history = {"I1": edits((body, None), (absolute("Users", "alice", "old"), "2026-11-01T00:00:00Z"))}
        result = self.run_check(repository([issue(1, body)], history=history))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")

    def test_items_after_the_first_hundred_are_read_in_the_next_batch(self):
        body = f"intro\n{absolute('Users', 'alice', 'x')}"
        plain = [issue(number, "plain") for number in range(1, 101)]
        result = self.run_check(repository(plain, [issue(101, body)], history={"I101": edits((body, None))}))
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["issue #101 body:2", "issue #101 body:2 (edit 1)"])
        self.assertIn("current text and edit histories 100/101", result.stderr)
        self.assertIn("current text and edit histories 101/101", result.stderr)

    def test_progress_is_visible_before_a_blocked_api_call_returns(self):
        fake = dict(repository([issue(1, "private payload")]), block_first=True)
        (self.bin / "fake.json").write_text(json.dumps(fake), encoding="utf-8")
        environment = dict(os.environ, PATH=os.pathsep.join([str(self.bin), os.environ.get("PATH", "")]))
        child = subprocess.Popen(
            [sys.executable, "-I", str(SCRIPT), REPOSITORY], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
        )
        try:
            ready, _, _ = select.select([child.stderr], [], [], 5)
            self.assertTrue(ready, "No progress was flushed before the blocked API returned")
            self.assertEqual(child.stderr.readline(), b"Audit progress: issue/PR inventory 0\n")
            self.assertIsNone(child.poll())
            stdout, stderr = child.communicate(b"x", timeout=10)
            self.assertEqual(child.returncode, 0, stderr)
            self.assertEqual(stdout, b"")
            self.assertNotIn(b"private payload", stderr)
            self.assertIn(b"scan complete; finding lines 0", stderr)
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate()

    def test_paths_separated_by_json_and_url_escapes_are_reported(self):
        body = "\n".join([
            "\\u002f".join(["", "Users", "alice", "model"]),
            "\\u002F".join(["", "Volumes", "Archive", "models"]),
            "%2F".join(["", "home", "alice"]),
            "%2f".join(["", "var", "folders", "xy"]),
            "\\/".join(["", "Users", "alice", "model"]),
        ])
        mixed = "a\\u002fb %2Fc\\nd %0A\n\\/opt\\u002Ftools%2fbin\nx " + "%2F".join(["", "Users", "alice", "y"]) + "\n"
        result = self.run_check(repository([issue(1, body), issue(2, mixed)]))
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [f"issue #1 body:{line}" for line in range(1, 6)] + ["issue #2 body:3"])

    def test_closed_progress_pipe_preserves_audit_exit_codes(self):
        cases = [(repository([issue(1, "plain")]), 0),
                 (repository([issue(1, absolute("Users", "alice", "x"))]), 1),
                 ({"output": "", "code": 1}, 2)]
        environment = dict(os.environ, PATH=os.pathsep.join([str(self.bin), os.environ.get("PATH", "")]))
        for fake, expected in cases:
            with self.subTest(expected=expected):
                (self.bin / "fake.json").write_text(json.dumps(fake), encoding="utf-8")
                reader, writer = os.pipe()
                os.close(reader)
                try:
                    result = subprocess.run([sys.executable, "-I", str(SCRIPT), REPOSITORY],
                                            stdout=subprocess.PIPE, stderr=writer, env=environment, timeout=10)
                finally:
                    os.close(writer)
                self.assertEqual(result.returncode, expected)
                self.assertEqual(result.stdout, b"issue #1 body:1\n" if expected == 1 else b"")

    def test_api_failures_exit_with_two_instead_of_reporting_no_paths(self):
        path = absolute("Users", "alice", "x")
        plain = [issue(number, "plain") for number in range(1, 101)]
        cases = {
            "gh fails": ({"output": "", "code": 1}, "CalledProcessError: Command "),
            "invalid JSON": ({"output": "not json", "code": 0}, "JSONDecodeError: Expecting value"),
            "more edits than one page": (repository([issue(1, path)], history={"I1": edits(("", None), more=True)}), None),
            "fewer edit histories than requested": (dict(repository([issue(1, path)]), missing=["I1"]), None),
            "fewer edit histories in the second batch": (dict(repository(plain, [issue(101, path)]), missing=["I101"]), None),
        }
        for name, (fake, cause) in cases.items():
            with self.subTest(name):
                result = self.run_check(fake)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(result.stdout, "")
                messages = result.stderr.splitlines()
                self.assertTrue(messages[-1].startswith("Could not read"), messages)
                if cause:
                    self.assertTrue(messages[-2].startswith(cause), messages)
                self.assertNotIn("scan complete", result.stderr)


if __name__ == "__main__":
    unittest.main()
