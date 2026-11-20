import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Tools.Conversion.adapters.voxcpm2 import reference_bundle, strip
from Tools.Conversion.recipe import load_recipe
from Tools.Conversion.storage import read_json
from Tools.Conversion.tests.test_voxcpm2_reference_bundle import CONTRACT, Fixture, RECIPE, ROOT, snapshot


class ReferenceBundleGuardTests(unittest.TestCase):
    def fixture(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Fixture(directory.name)

    def test_case_aliases_and_resolved_parents_cannot_enter_an_input(self):
        f = self.fixture()
        aliases = [root.with_name(root.name.upper()) for root in (f.base, f.run)]
        if not all(alias.exists() and alias.samefile(root) for alias, root in zip(aliases, (f.base, f.run))):
            self.skipTest("This volume does not provide the same-inode case aliases")
        targets = [alias / "new" / "nested" for alias in aliases]
        nested = f.base / "subfolder"
        nested.mkdir()
        link = f.directory / "input-link"
        link.symlink_to(nested, target_is_directory=True)
        targets.append(link / "new")
        before = snapshot(f.base, f.run)
        for target in targets:
            with self.subTest(target=str(target)), self.assertRaises(ValueError):
                f.build(output=target)
            self.assertFalse(os.path.lexists(target))
            self.assertEqual(snapshot(f.base, f.run), before)
        self.assertEqual(f.history(), [])
        result = f.build(output=f.directory / "sibling" / "new")
        self.assertTrue(result["passed"])

    def test_strip_one_rejects_a_case_alias_before_loading(self):
        f = self.fixture()
        alias = f.raw.with_name(f.raw.name.upper())
        if not alias.exists() or not alias.samefile(f.raw):
            self.skipTest("This volume does not provide the same-inode case alias")
        before = snapshot(f.run)
        with f.runtime(), self.assertRaises(ValueError):
            strip.strip_one("refenc", "fp32", f.raw, alias / "nested.aimodel")
        self.assertEqual(snapshot(f.run), before)
        self.assertEqual(f.history(), [])

    def test_reselected_errors_are_nonnegative_for_each_comparison(self):
        for kind in ("fixed", "coreai"):
            for value in (-0.1, math.nextafter(0.001, math.inf)):
                with self.subTest(kind=kind, value=value):
                    f = self.fixture()
                    report = read_json(f.run / "report.json")
                    report["summary"][kind]["normalized_max"] = value
                    f.save_report(report)
                    with self.assertRaises(ValueError):
                        f.build()
                    self.assertEqual(f.history(), [])
                    self.assertFalse(f.output.exists())
            for value in (0, -0.0, 0.001):
                with self.subTest(kind=kind, value=value):
                    f = self.fixture()
                    report = read_json(f.run / "report.json")
                    report["summary"][kind]["normalized_max"] = value
                    reference_bundle.eligible(report, read_json(f.selection)["files"], CONTRACT)

    def test_reference_assembly_always_revalidates_inputs(self):
        recipe = load_recipe(
            RECIPE / "reference-bundle-recipe.json",
            {"python": "python3", "base_bundle": "/base", "encoder_run": "/run", "selection": "/selection.json"}, ROOT)
        self.assertIs(recipe["steps"][0]["cache"], False)

    def test_final_check_write_failure_does_not_publish_a_passing_record(self):
        f = self.fixture()
        before = snapshot(f.base, f.run, f.selection)
        original = reference_bundle.write_json

        def fail_final(path, value):
            if Path(path).name == "check.json" and value.get("passed") is True:
                raise OSError("The final check could not be saved")
            original(path, value)

        with mock.patch.object(reference_bundle, "write_json", fail_final), self.assertRaises(OSError):
            f.build()
        self.assertTrue((f.output / "reference_fp32.inventory.json").is_file())
        check = read_json(f.output / "check.json")
        self.assertIs(check["passed"], False)
        self.assertIs(check["complete"], False)
        self.assertEqual(snapshot(f.base, f.run, f.selection), before)


if __name__ == "__main__":
    unittest.main()
