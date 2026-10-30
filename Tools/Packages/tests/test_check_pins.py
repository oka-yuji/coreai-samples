import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "check-pins.py"


def pin(identity, version, revision):
    return {
        "identity": identity,
        "kind": "remoteSourceControl",
        "location": f"https://example.invalid/{identity}.git",
        "state": {"revision": revision, "version": version},
    }


def remote(identity, version, revision):
    return {
        "basedOn": None,
        "packageRef": {
            "identity": identity,
            "kind": "remoteSourceControl",
            "location": f"https://example.invalid/{identity}.git",
            "name": identity,
        },
        "state": {"checkoutState": {"revision": revision, "version": version}, "name": "sourceControlCheckout"},
        "subpath": identity,
    }


def local(identity):
    return {
        "basedOn": None,
        "packageRef": {"identity": identity, "kind": "fileSystem", "location": f"/example/{identity}", "name": identity},
        "state": {"name": "fileSystem", "path": f"/example/{identity}"},
        "subpath": identity,
    }


KIT_PINS = [pin("alpha", "1.0.0", "a1"), pin("beta", "2.0.0", "b2")]
KIT_SELECTION = [remote("alpha", "1.0.0", "a1"), remote("beta", "2.0.0", "b2")]


class PinChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pin checks ")
        self.addCleanup(temporary.cleanup)
        self.alias = Path(temporary.name)
        self.directory = self.alias.resolve()
        self.root = self.directory / "repo"
        self.home = self.directory / "home"
        self.home.mkdir()
        self.derived = self.directory / "DerivedData"
        self.kit_lock = self.root / "CoreAISamplesKit/Package.resolved"
        self.verify_lock = self.root / "Tools/Verify/Package.resolved"
        self.kit_state = self.root / "CoreAISamplesKit/.build/workspace-state.json"
        self.xcode_state = self.derived / "SourcePackages/workspace-state.json"
        self.verify_state = self.root / "Tools/Verify/.build/workspace-state.json"
        (self.root / "Tools/Packages").mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "Tools/Packages/check-pins.py")
        self.lock(self.kit_lock, KIT_PINS)
        self.lock(self.verify_lock, KIT_PINS + [pin("gamma", "3.0.0", "c3")])
        self.state(self.kit_state, KIT_SELECTION + [local("pnginputkit")])
        self.state(self.xcode_state, KIT_SELECTION + [local("coreaisampleskit"), local("pnginputkit")])
        self.state(self.verify_state, KIT_SELECTION + [remote("gamma", "3.0.0", "c3"), local("coreaisampleskit")])

    def lock(self, path, pins):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"originHash": "0", "pins": pins, "version": 3}, indent=2))

    def state(self, path, dependencies):
        path.parent.mkdir(parents=True, exist_ok=True)
        content = {"object": {"artifacts": [], "dependencies": dependencies, "prebuilts": []}, "version": 7}
        path.write_text(json.dumps(content, indent=2))

    def derived_data(self, name, workspace, fmt=plistlib.FMT_XML):
        folder = self.home / "Library/Developer/Xcode/DerivedData" / name
        folder.mkdir(parents=True)
        (folder / "info.plist").write_bytes(plistlib.dumps({"WorkspacePath": str(workspace)}, fmt=fmt))
        self.state(folder / "SourcePackages/workspace-state.json", KIT_SELECTION)
        return folder

    def check(self, *args):
        return subprocess.run(
            [sys.executable, "-I", str(self.root / "Tools/Packages/check-pins.py"), *args],
            cwd=self.directory,
            env=dict(os.environ, HOME=str(self.home)),
            text=True,
            capture_output=True,
        )

    def test_matching_states_and_locks_pass(self):
        result = self.check("--with-verify", str(self.derived))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("PASS "), 4)
        self.assertEqual(result.stderr, "")

    def test_stale_xcode_checkout_fails_with_identity_and_both_values(self):
        self.state(self.xcode_state, [remote("alpha", "1.0.1", "a2"), remote("beta", "2.0.0", "b2")])
        result = self.check(str(self.derived))
        self.assertEqual(result.returncode, 1)
        self.assertIn(
            f"{self.xcode_state}: alpha: selected version 1.0.1 revision a2, but {self.kit_lock} has version 1.0.0 revision a1",
            result.stderr,
        )
        self.assertIn(f"Then delete {self.derived / 'SourcePackages'} and run xcodebuild again", result.stderr)
        self.assertIn(f"PASS {self.kit_state}", result.stdout)
        self.assertNotIn(": beta:", result.stderr)

    def test_selected_package_without_pin_fails(self):
        self.lock(self.kit_lock, KIT_PINS[:1])
        result = self.check(str(self.derived))
        self.assertEqual(result.returncode, 1)
        for state in (self.kit_state, self.xcode_state):
            self.assertIn(
                f"{state}: beta: selected version 2.0.0 revision b2, but {self.kit_lock} has no pin", result.stderr
            )

    def test_pin_without_selected_package_fails(self):
        extra = pin("delta", "4.0.0", "d4")
        self.lock(self.kit_lock, KIT_PINS + [extra])
        self.lock(self.verify_lock, KIT_PINS + [extra, pin("gamma", "3.0.0", "c3")])
        result = self.check(str(self.derived))
        self.assertEqual(result.returncode, 1)
        for state in (self.kit_state, self.xcode_state):
            self.assertIn(
                f"{state}: delta: selected nothing, but {self.kit_lock} has version 4.0.0 revision d4", result.stderr
            )

    def test_verify_state_is_checked_only_with_the_flag(self):
        self.state(self.verify_state, KIT_SELECTION + [remote("gamma", "3.0.1", "c4")])
        self.assertEqual(self.check(str(self.derived)).returncode, 0)
        result = self.check("--with-verify", str(self.derived))
        self.assertEqual(result.returncode, 1)
        self.assertIn(
            f"{self.verify_state}: gamma: selected version 3.0.1 revision c4, but {self.verify_lock} has version 3.0.0 revision c3",
            result.stderr,
        )

    def test_verify_lock_that_differs_from_kit_lock_fails(self):
        for verify_pins, found in (
            ([pin("alpha", "1.0.0", "a1"), pin("beta", "2.0.1", "b3")], "version 2.0.1 revision b3"),
            ([pin("alpha", "1.0.0", "a1")], "no pin"),
        ):
            with self.subTest(found=found):
                self.lock(self.verify_lock, verify_pins)
                result = self.check(str(self.derived))
                self.assertEqual(result.returncode, 1)
                self.assertIn(
                    f"{self.verify_lock}: beta: {found}, but {self.kit_lock} has version 2.0.0 revision b2", result.stderr
                )

    def test_missing_state_file_fails(self):
        self.xcode_state.unlink()
        result = self.check(str(self.derived))
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"{self.xcode_state}: not found", result.stderr)

    def test_local_packages_are_ignored(self):
        self.state(self.xcode_state, KIT_SELECTION + [local("coreaisampleskit"), local("unpinned-local")])
        result = self.check(str(self.derived))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"PASS {self.xcode_state}: 2 packages match {self.kit_lock}", result.stdout)

    def test_default_lookup_uses_the_only_matching_derived_data(self):
        folder = self.derived_data("CoreAISamples-one", self.alias / "repo/CoreAISamples.xcodeproj", plistlib.FMT_BINARY)
        self.derived_data("Other-two", "/example/Other.xcodeproj")
        (self.home / "Library/Developer/Xcode/DerivedData/ModuleCache.noindex").mkdir()
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"PASS {folder / 'SourcePackages/workspace-state.json'}", result.stdout)

    def test_default_lookup_fails_without_a_match(self):
        self.derived_data("Other-two", "/example/Other.xcodeproj")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"WorkspacePath {self.root / 'CoreAISamples.xcodeproj'}, found 0.", result.stderr)

    def test_default_lookup_fails_with_two_matches(self):
        first = self.derived_data("CoreAISamples-one", self.root / "CoreAISamples.xcodeproj")
        second = self.derived_data("CoreAISamples-two", self.alias / "repo/CoreAISamples.xcodeproj")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("found 2.", result.stderr)
        self.assertIn(f"\n  {first}\n  {second}\n", result.stderr)


if __name__ == "__main__":
    unittest.main()
