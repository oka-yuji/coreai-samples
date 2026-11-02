import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1]


def absolute(*parts):
    return "/".join(["", *parts])


class CommitChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="commit checks ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "work"
        self.remote = Path(temporary.name) / "remote.git"
        self.environment = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", os.defpath)]),
        }
        self.root.mkdir()
        self.git("init", "-q", "-b", "main")
        for name in ("check-local-paths.py", "hooks/pre-push"):
            target = self.root / "Tools/Privacy" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(TOOLS / name, target)
        self.commit("chore: add the checks")

    def git(self, *args, check=True):
        result = subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false", "-C", str(self.root), *args],
            capture_output=True,
            text=True,
            env=self.environment,
        )
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def commit(self, message="chore: change"):
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD").stdout.strip()

    def publish(self):
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")

    def remote_branch(self, name):
        return self.git("ls-remote", "origin", f"refs/heads/{name}").stdout.split("\t")[0]

    def run_check(self, *args):
        return subprocess.run(
            [sys.executable, "-I", str(self.root / "Tools/Privacy/check-local-paths.py"), "--commits", *args],
            capture_output=True,
            text=True,
            env=self.environment,
        )

    def assertNoMatchedText(self, result):
        for text in ("alice", "Archive"):
            self.assertNotIn(text, result.stdout + result.stderr)

    def test_path_added_and_later_removed_is_reported_with_the_commit_that_added_it(self):
        self.publish()
        self.write("docs/notes.md", f"intro\n{absolute('Users', 'alice', 'work', 'run.log')}\n")
        added = self.commit()
        self.write("docs/other.md", "other\n")
        self.commit()
        self.write("docs/notes.md", "intro\n")
        self.commit()
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.splitlines(), [f"docs/notes.md:2 ({added[:7]})"])
        self.assertNoMatchedText(result)

    def test_path_in_a_commit_message_is_reported_by_commit_and_message_line(self):
        self.publish()
        self.write("docs/notes.md", "intro\n")
        commit = self.commit(f"chore: add notes\n\nCopied from {absolute('Users', 'alice', 'notes')}\n")
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.splitlines(), [f"{commit[:7]}:3 (message)"])
        self.assertNoMatchedText(result)

    def test_path_in_a_blob_that_a_merge_commit_introduces_is_reported(self):
        self.publish()
        self.git("switch", "-q", "-c", "topic")
        self.write("docs/topic.md", "topic\n")
        self.commit()
        self.git("switch", "-q", "main")
        self.git("merge", "-q", "--no-ff", "--no-commit", "topic")
        self.write("docs/merged.md", f"merged\n{absolute('Volumes', 'Archive', 'models')}\n")
        merge = self.commit("chore: merge topic")
        self.assertEqual(len(self.git("rev-list", "--parents", "-n", "1", "HEAD").stdout.split()), 3)
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.splitlines(), [f"docs/merged.md:2 ({merge[:7]})"])
        self.assertNoMatchedText(result)

    def test_commits_already_in_the_base_are_not_read(self):
        self.write("docs/old.md", f"{absolute('Users', 'alice', 'old')}\n")
        self.commit(f"chore: add old notes\n\nFrom {absolute('Users', 'alice', 'old')}\n")
        self.publish()
        self.write("docs/new.md", "new\n")
        self.commit()
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNoMatchedText(result)

    def test_pre_push_hook_blocks_new_commits_with_paths_and_allows_the_rest(self):
        self.write("docs/old.md", f"{absolute('Users', 'alice', 'old')}\n")
        self.commit()
        subprocess.run(["git", "init", "-q", "--bare", str(self.remote)], env=self.environment, check=True)
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "-q", "origin", "main")
        self.git("config", "core.hooksPath", "Tools/Privacy/hooks")
        self.git("switch", "-q", "-c", "other")
        self.write("docs/other.md", "other\n")
        self.commit()
        self.git("switch", "-q", "-c", "topic", "main")
        self.write("docs/new.md", "new\n")
        clean = self.commit()
        self.git("push", "-q", "origin", "topic")
        self.assertEqual(self.remote_branch("topic"), clean)
        self.write("docs/new.md", f"{absolute('Users', 'alice', 'new')}\n")
        added = self.commit()
        blocked = self.git("push", "-q", "origin", "topic", "other", check=False)
        self.assertNotEqual(blocked.returncode, 0)
        self.assertIn(f"docs/new.md:1 ({added[:7]})", blocked.stdout + blocked.stderr)
        self.assertNoMatchedText(blocked)
        self.assertEqual(self.remote_branch("topic"), clean)
        self.assertEqual(self.remote_branch("other"), "")
        self.git("push", "-q", "origin", "--delete", "topic")
        self.assertEqual(self.remote_branch("topic"), "")

    def test_unknown_revision_exits_with_two_and_reports_nothing(self):
        self.publish()
        result = self.run_check("missing..HEAD")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertNoMatchedText(result)


if __name__ == "__main__":
    unittest.main()
