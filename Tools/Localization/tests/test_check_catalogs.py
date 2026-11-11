import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "check-catalogs.py"
APP = "CoreAISamples/Resources/Localizable.xcstrings"
KIT = "CoreAISamplesKit/Sources/ReceiptUI/Resources/Localizable.xcstrings"


def unit(value, state="translated"):
    return {"stringUnit": {"state": state, "value": value}}


def japanese(value, state="translated"):
    return {"localizations": {"ja": unit(value, state)}}


class CatalogChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="catalog checks ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.script = self.root / "Tools/Localization/check-catalogs.py"
        self.script.parent.mkdir(parents=True)
        shutil.copy(SCRIPT, self.script)

    def write(self, name, strings, track=True):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"sourceLanguage": "en", "strings": strings, "version": "1.0"}), encoding="utf-8")
        if track:
            subprocess.run(["git", "-C", str(self.root), "add", "--", name], check=True)

    def run_check(self):
        result = subprocess.run([sys.executable, "-I", str(self.script)], capture_output=True, text=True)
        return result.returncode, json.loads(result.stdout)

    def test_same_named_catalogs_in_different_folders_are_reported_by_relative_path(self):
        self.write(APP, {"Hello": japanese("Hello ja")})
        self.write(KIT, {"Bye": {}, "Hello": japanese("Hello ja")})
        self.write(".build/Copy/Localizable.xcstrings", {"Untracked": {}}, track=False)
        code, report = self.run_check()
        self.assertEqual(code, 1)
        self.assertEqual(sorted(report["catalogs"]), [APP, KIT])
        self.assertEqual(report["catalogs"][APP]["keys"], 1)
        self.assertEqual(report["catalogs"][KIT]["keys"], 2)
        self.assertEqual(report["errors"], [f"{KIT}: ja: missing translation: Bye"])

    def test_escaped_percent_signs_are_not_placeholders(self):
        self.write(APP, {"%lld%% done": japanese("%lld ja"), "Rate %%d": japanese("Rate ja")})
        code, report = self.run_check()
        self.assertEqual((code, report["errors"]), (0, []))

    def test_percent_followed_by_a_space_and_a_letter_is_not_a_placeholder(self):
        self.write(APP, {"50% off": japanese("50 ja")})
        code, report = self.run_check()
        self.assertEqual((code, report["errors"]), (0, []))

    def test_existing_findings_are_unchanged(self):
        self.write(APP, {
            "Missing": {},
            "Review": japanese("Review ja", "needs_review"),
            "Empty": japanese(""),
            "Count %lld": japanese("Count ja"),
            "%1$@ of %2$lld": japanese("%2$lld / %1$@"),
            "%lld items": {"localizations": {"ja": {"variations": {"plural": {"other": unit("%lld ja")}}}}},
            "Skipped": {"shouldTranslate": False},
        })
        code, report = self.run_check()
        self.assertEqual(code, 1)
        self.assertEqual(report["errors"], [
            f"{APP}: ja: missing translation: Missing",
            f"{APP}: ja: incomplete translation: Review",
            f"{APP}: ja: incomplete translation: Empty",
            f"{APP}: ja: placeholder mismatch: Count %lld",
        ])
        self.assertEqual(report["catalogs"][APP], {"keys": 7, "translatedKeys": 6, "languages": ["en", "ja"]})


if __name__ == "__main__":
    unittest.main()
