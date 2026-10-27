import copy
import json
import shutil
import subprocess
import unittest

try:
    import numpy
except ImportError as error:
    raise unittest.SkipTest(f"NumPy unavailable: {error}") from error

from Tools.Conversion.adapters.qwen_image21.mobile_compare import quality_template, score_quality
from Tools.Conversion.adapters.qwen_image21.mobile_gallery import validate_ratings


class ImageMobileRatingsTests(unittest.TestCase):
    def cohorts(self):
        return {name: {"summary_file": {"sha256": "a" * 64, "bytes": 400},
                       "identity": {"sigmas": [1.0, 0.25, 0.0], "enabled": False}}
                for name in ("A0", "A1", "B", "C")}

    @unittest.skipUnless(shutil.which("node"), "Node is required for the actual browser JSON number roundtrip")
    def test_browser_export_preserves_numeric_identity_and_unreviewed_scores(self):
        cohorts = self.cohorts()
        original = quality_template(cohorts)
        original["cohorts"]["A0"]["ratings"][0]["axes"]["instruction"] = 2
        result = subprocess.run([shutil.which("node"), "-e",
                                 'process.stdout.write(JSON.stringify(JSON.parse(require("fs").readFileSync(0,"utf8"))))'],
                                input=json.dumps(original), capture_output=True, text=True, check=True)
        restored = json.loads(result.stdout)
        self.assertIs(type(restored["cohorts"]["A0"]["identity"]["identity"]["sigmas"][0]), int)
        before = copy.deepcopy(restored)
        validate_ratings(restored, cohorts)
        quality = score_quality(restored, cohorts)
        self.assertFalse(quality["complete"])
        self.assertFalse(quality["passed"])
        self.assertEqual(restored, before)
        self.assertEqual(restored["cohorts"]["A0"]["ratings"], original["cohorts"]["A0"]["ratings"])

    def test_real_identity_changes_are_rejected_by_both_consumers(self):
        cohorts = self.cohorts()
        for defect in ("bool-number", "float-change", "hash", "large-integer", "nonfinite"):
            with self.subTest(defect=defect):
                expected = copy.deepcopy(cohorts)
                if defect == "large-integer":
                    expected["A0"]["identity"]["counter"] = 9007199254740993
                value = quality_template(expected)
                identity = value["cohorts"]["A0"]["identity"] = copy.deepcopy(expected["A0"])
                if defect == "bool-number":
                    identity["identity"]["sigmas"][0] = True
                elif defect == "float-change":
                    identity["identity"]["sigmas"][0] = 1.00000001
                elif defect == "hash":
                    identity["summary_file"]["sha256"] = "b" * 64
                elif defect == "large-integer":
                    identity["identity"]["counter"] = 9007199254740992
                else:
                    identity["identity"]["sigmas"][0] = float("nan")
                with self.assertRaises(ValueError):
                    validate_ratings(value, expected)
                with self.assertRaises(ValueError):
                    score_quality(value, expected)


if __name__ == "__main__":
    unittest.main()
