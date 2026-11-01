from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "check-local-paths.py"


def absolute(*parts):
    return "/".join(["", *parts])


class LocalPathChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="local path checks ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        script = self.root / "Tools/Privacy/check-local-paths.py"
        script.parent.mkdir(parents=True)
        shutil.copy(SCRIPT, script)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")

    def run_check(self):
        subprocess.run(["git", "-C", str(self.root), "add", "-A"], check=True)
        return subprocess.run(
            [sys.executable, "-I", str(self.root / "Tools/Privacy/check-local-paths.py")],
            capture_output=True,
            text=True,
        )

    def test_home_volume_and_temporary_paths_are_reported_by_file_and_line_only(self):
        self.write("docs/notes.md", f"intro\nlog {absolute('Users', 'alice', 'work', 'run.log')}\n")
        self.write("docs/data.json", f'{{\n  "home": "{absolute("Users", "alice")}",\n  "disk": "{absolute("Volumes", "Archive", "models")}"\n}}\n')
        self.write("logs/build.log", f"a\nb\n{absolute('private', 'var', 'folders', 'xy', 'T', 'out')}\n")
        self.write("logs/linux.log", f"{absolute('home', 'alice', 'src')}\n")
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(
            result.stdout.splitlines(),
            ["docs/data.json:2", "docs/data.json:3", "docs/notes.md:2", "logs/build.log:3", "logs/linux.log:1"],
        )
        for text in ("alice", "Archive", "folders", "run.log"):
            self.assertNotIn(text, result.stdout + result.stderr)

    def test_path_inside_a_binary_file_is_reported(self):
        self.write("Assets/blob.bin", b"\x00\x01\n\x89PNG" + absolute("Users", "alice", "x").encode() + b"\x00")
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.splitlines(), ["Assets/blob.bin:2"])

    def test_upstream_ci_build_paths_and_files_without_absolute_paths_pass(self):
        self.write("Vendor/LICENSE", f"Files: {absolute('Users', 'runner', 'work', 'lib')}\n{absolute('home', 'runner', 'work')}\n")
        self.write("docs/guide.md", "Use ~/Library/Caches/app or docs/media/demo.gif and `/Volumes/`.\n")
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")

    def test_untracked_files_are_not_read(self):
        subprocess.run(["git", "-C", str(self.root), "add", "-A"], check=True)
        (self.root / "scratch.txt").write_text(absolute("Users", "alice", "x"), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-I", str(self.root / "Tools/Privacy/check-local-paths.py")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
