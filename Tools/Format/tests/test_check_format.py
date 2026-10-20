import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


FORMAT_DIR = Path(__file__).resolve().parents[1]
VALID_SWIFT = "struct Example {\n    let value: Int\n}\n"
WIRE_PATH = "CoreAISamples/Demos/Agent/AgentPromptBuilder.swift"
WIRE_SOURCE = "@Generable\nstruct WebSearchArguments {\n    var max_results: Int?\n}\n"
WIRE_ERROR = (
    "{path}:3:9: error: [AlwaysUseLowerCamelCase] "
    "rename the variable 'max_results' using lowerCamelCase\n"
)


class FormatChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="format checks ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.root = self.directory / "sample repo"
        self.root.mkdir()
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        self.env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Format Test")
        self.git("config", "user.email", "format@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.hooksPath", os.devnull)
        self.write(".swift-format", '{"version":1,"indentation":{"spaces":4},"lineLength":120}\n')
        self.write(".gitignore", "ignored/\n")
        self.write("Stable.swift", VALID_SWIFT)
        for script in FORMAT_DIR.glob("*.py"):
            self.write("Tools/Format/" + script.name, script.read_text())
        self.git("add", ".")
        self.git("commit", "-qm", "Initial fixture")
        self.base = self.git("rev-parse", "HEAD").stdout.strip()
        self.binary_dir = self.directory / "bin"
        self.binary_dir.mkdir()
        self.log = self.directory / "calls.jsonl"
        formatter = self.binary_dir / "xcrun"
        formatter.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['FORMAT_TEST_LOG'], 'a') as output:\n"
            "    output.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "sys.stderr.write(os.environ.get('FORMAT_TEST_STDERR', '').replace('{path}', sys.argv[-1]))\n"
            "sys.exit(int(os.environ.get('FORMAT_TEST_EXIT', '0')))\n"
        )
        formatter.chmod(0o755)
        self.env.update(
            PATH=str(self.binary_dir) + os.pathsep + os.environ["PATH"],
            FORMAT_TEST_LOG=str(self.log),
        )

    def write(self, name, content=VALID_SWIFT):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def git(self, *args):
        return subprocess.run(
            ["git", *args], cwd=self.root, env=self.env, text=True, capture_output=True, check=True
        )

    def check(self, *args, environment=None):
        return subprocess.run(
            [sys.executable, str(self.root / "Tools/Format/check_format.py"), *args],
            cwd=self.directory,
            env=self.env if environment is None else environment,
            text=True,
            capture_output=True,
        )

    def checked_names(self):
        if not self.log.exists():
            return []
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        for call in calls:
            self.assertEqual(
                call[:-1],
                ["swift-format", "lint", "--strict", "--configuration", str(self.root / ".swift-format")],
            )
        return [str(Path(call[-1]).relative_to(self.root)) for call in calls]

    def test_valid_swift_passes_without_personal_configuration(self):
        spec = importlib.util.spec_from_file_location(
            "format_fixture", self.root / "Tools/Format/check_format.py"
        )
        entrypoint = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entrypoint)
        output, errors = io.StringIO(), io.StringIO()
        with mock.patch("pathlib.Path.home", return_value=self.directory / "no personal config"), \
                mock.patch.object(sys, "argv", ["check_format.py", "--all"]), \
                mock.patch.dict(os.environ, self.env, clear=True), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            result = entrypoint.main()
        self.assertEqual(result, 0, errors.getvalue())
        self.assertEqual(self.checked_names(), ["Stable.swift"])
        self.assertIn("checked 1 Swift files; PASS", output.getvalue())

    def prepare_changed_files(self):
        for name in ("Unstaged.swift", "Deleted.swift", "Rename.swift"):
            self.write(name)
        self.git("add", ".")
        self.git("commit", "-qm", "Base selection fixture")
        base = self.git("rev-parse", "HEAD").stdout.strip()
        self.write("Committed.swift")
        self.git("add", "Committed.swift")
        self.git("commit", "-qm", "Committed change")
        self.write("Staged.swift")
        self.git("add", "Staged.swift")
        self.write("Unstaged.swift", VALID_SWIFT + "\n")
        self.write("New folder/Untracked file.swift")
        self.write("ignored/Ignored.swift")
        self.write("Notes.txt", "Not Swift\n")
        self.git("rm", "Deleted.swift")
        self.git("mv", "Rename.swift", "Renamed file.swift")
        return base

    def test_base_selects_changes_untracked_and_renames(self):
        base = self.prepare_changed_files()
        result = self.check("--base", base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.checked_names(), [
            "Committed.swift", "New folder/Untracked file.swift", "Renamed file.swift",
            "Staged.swift", "Unstaged.swift",
        ])

    def test_all_includes_unchanged_and_untracked_files(self):
        self.prepare_changed_files()
        result = self.check("--all")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.checked_names(), [
            "Committed.swift", "New folder/Untracked file.swift", "Renamed file.swift",
            "Stable.swift", "Staged.swift", "Unstaged.swift",
        ])

    def test_base_uses_merge_base_when_target_has_advanced(self):
        self.git("checkout", "-qb", "target")
        self.write("Target only.swift")
        self.git("add", ".")
        self.git("commit", "-qm", "Target change")
        self.git("checkout", "-q", "main")
        self.write("Branch.swift")
        self.git("add", ".")
        self.git("commit", "-qm", "Branch change")
        result = self.check("--base", "target")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.checked_names(), ["Branch.swift"])

    def test_no_swift_changes_passes_without_invoking_formatter(self):
        result = self.check("--base", self.base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.checked_names(), [])
        self.assertIn("checked 0 Swift files; PASS", result.stdout)

    def test_partial_staging_is_rejected_in_both_scopes(self):
        self.write("Stable.swift", VALID_SWIFT + "\n")
        self.git("add", "Stable.swift")
        self.write("Stable.swift", VALID_SWIFT + "\n\n")
        for scope in (("--all",), ("--base", self.base)):
            with self.subTest(scope=scope):
                result = self.check(*scope)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("both staged and unstaged", result.stderr)
                self.assertEqual(self.checked_names(), [])

    def test_symlinks_are_rejected_without_following_them(self):
        (self.root / "Linked.swift").symlink_to(self.root / "Stable.swift")
        result = self.check("--base", self.base)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Swift symlink", result.stderr)
        self.assertEqual(self.checked_names(), [])

    def test_missing_configuration_is_an_invocation_error(self):
        (self.root / ".swift-format").unlink()
        result = self.check("--all")
        self.assertEqual(result.returncode, 2)
        self.assertIn(".swift-format is missing", result.stderr)
        self.assertEqual(self.checked_names(), [])

    def test_invalid_base_is_an_invocation_error(self):
        result = self.check("--base", "no-such-ref")
        self.assertEqual(result.returncode, 2)
        self.assertIn("format check could not run", result.stderr)
        self.assertEqual(self.checked_names(), [])

    def test_missing_formatter_is_an_invocation_error(self):
        missing_bin = self.directory / "no formatter"
        missing_bin.mkdir()
        (missing_bin / "git").symlink_to(shutil.which("git"))
        environment = dict(self.env, PATH=str(missing_bin))
        result = self.check("--all", environment=environment)
        self.assertEqual(result.returncode, 2)
        self.assertIn("format check could not run", result.stderr)
        self.assertEqual(self.checked_names(), [])

    def test_scope_must_be_unambiguous(self):
        for scope in ((), ("--all", "--base", self.base)):
            with self.subTest(scope=scope):
                result = self.check(*scope)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(self.checked_names(), [])

    def test_lint_failure_preserves_diagnostic_and_status(self):
        self.env.update(FORMAT_TEST_EXIT="1", FORMAT_TEST_STDERR="formatting failed\n")
        result = self.check("--all")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "formatting failed\n")
        self.assertEqual(result.stdout, "swift-format: checked 1 Swift files; FAIL\n")

    def prepare_wire_exception(self, source=WIRE_SOURCE, path=WIRE_PATH, error=WIRE_ERROR):
        (self.root / "Stable.swift").unlink()
        self.write(path, source)
        self.env.update(FORMAT_TEST_EXIT="1", FORMAT_TEST_STDERR=error)

    def test_only_the_fixed_wire_name_diagnostic_is_accepted(self):
        self.prepare_wire_exception()
        result = self.check("--all")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fixed wire-name exception WebSearchArguments.max_results", result.stdout)

    def test_other_types_declarations_files_and_diagnostics_are_rejected(self):
        cases = [
            (WIRE_SOURCE.replace("WebSearchArguments", "OtherArguments"), WIRE_PATH, WIRE_ERROR),
            (WIRE_SOURCE.replace("Int?", "String?"), WIRE_PATH, WIRE_ERROR),
            (WIRE_SOURCE + "var max_results: Int?\n", WIRE_PATH, WIRE_ERROR),
            (WIRE_SOURCE, "Other/AgentPromptBuilder.swift", WIRE_ERROR),
            (WIRE_SOURCE, WIRE_PATH, WIRE_ERROR + "another diagnostic\n"),
            (WIRE_SOURCE, WIRE_PATH, WIRE_ERROR.replace(":3:9:", ":4:9:")),
            (WIRE_SOURCE.replace("@Generable\n", "\n"), WIRE_PATH, WIRE_ERROR),
        ]
        for source, path, error in cases:
            with self.subTest(source=source, path=path, error=error):
                for existing in self.root.rglob("*.swift"):
                    existing.unlink()
                self.write(path, source)
                self.env.update(FORMAT_TEST_EXIT="1", FORMAT_TEST_STDERR=error)
                result = self.check("--all")
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertNotIn("fixed wire-name exception", result.stdout)

    @unittest.skipUnless(shutil.which("xcrun"), "Xcode command-line tools are required")
    def test_real_formatter_accepts_valid_and_rejects_invalid_without_writing(self):
        environment = dict(self.env, PATH=os.environ["PATH"])
        for source, status in ((VALID_SWIFT, 0), ("struct Example{let value:Int}\n", 1)):
            with self.subTest(status=status):
                path = self.write("Stable.swift", source)
                result = self.check("--all", environment=environment)
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertEqual(path.read_text(), source)
                self.assertIn("checked 1 Swift files;", result.stdout)


if __name__ == "__main__":
    unittest.main()
