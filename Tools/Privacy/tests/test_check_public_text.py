import json
import os
from pathlib import Path
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
if "output" in fake:
    sys.stdout.write(fake["output"])
    sys.exit(fake["code"])
if sys.argv[1:] == ["api", "graphql", "--input", "-"]:
    nodes = re.findall(r'"([^"]*)"', json.load(sys.stdin)["query"])[:fake.get("limit")]
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


def repository(body, history=None):
    return {
        "pages": {
            endpoint("issues?state=all&per_page=100"): [[{"number": 1, "node_id": "I1", "title": "Notes", "body": body}]],
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
        result = self.run_check(repository(body, history))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")

    def test_paths_separated_by_json_and_url_escapes_are_reported(self):
        body = "\n".join([
            "\\u002f".join(["", "Users", "alice", "model"]),
            "\\u002F".join(["", "Volumes", "Archive", "models"]),
            "%2F".join(["", "home", "alice"]),
            "%2f".join(["", "var", "folders", "xy"]),
            "\\/".join(["", "Users", "alice", "model"]),
        ])
        result = self.run_check(repository(body))
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [f"issue #1 body:{line}" for line in range(1, 6)])

    def test_api_failures_exit_with_two_instead_of_reporting_no_paths(self):
        cases = {
            "gh fails": {"output": "", "code": 1},
            "invalid JSON": {"output": "not json", "code": 0},
            "more edits than one page": repository(absolute("Users", "alice", "x"), {"I1": edits(("", None), more=True)}),
            "fewer edit histories than requested": dict(repository(absolute("Users", "alice", "x")), limit=0),
        }
        for name, fake in cases.items():
            with self.subTest(name):
                result = self.run_check(fake)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn("Could not read", result.stderr)


if __name__ == "__main__":
    unittest.main()
