import json
from pathlib import Path
import tempfile
import unittest

from Tools.Conversion.adapters.voxcpm2 import text_fixture

BASE = Path(__file__).resolve().parents[3]
RESOURCE = BASE / "CoreAISamplesKit/Tests/VoxCPMCoreTests/Resources/text-normalization.json"


class TextFixtureTests(unittest.TestCase):
    def test_the_whitespace_set_matches_str_isspace_and_the_cases_follow_the_original_rule(self):
        built = text_fixture.build()
        self.assertTrue(built["isspaceMatches"])
        self.assertEqual(built["whitespace"], [code for code in range(0x110000) if chr(code).isspace()])
        self.assertIn(0x3000, built["whitespace"])
        self.assertNotIn(0x200B, built["whitespace"])
        cases = {case["name"]: case["normalized"] for case in built["cases"]}
        self.assertIsNone(cases["empty"])
        self.assertIsNone(cases["whitespace only"])
        self.assertEqual(cases["leading and trailing spaces"], " a b ")
        self.assertEqual(cases["newline runs"], "a b")
        self.assertEqual(cases["zero width space"], "a\u200bb")
        self.assertEqual(cases["combining mark after a space"], "a \u0301b")
        self.assertEqual(cases["supplementary characters around spaces"], "\U0001f600 \U0001f4ac \U00020000")
        self.assertEqual(cases["supplementary characters at both ends"], " \U0001f600 \U0001f600 ")
        for case in built["cases"]:
            if case["normalized"] is not None:
                self.assertNotIn("  ", case["normalized"], case["name"])
                self.assertFalse(any(text_fixture.SPACE.fullmatch(character) and character != " "
                                     for character in case["normalized"]), case["name"])

    def test_the_resource_equals_a_new_build_from_the_recorded_versions(self):
        resource = json.loads(RESOURCE.read_text(encoding="ascii"))
        built = text_fixture.build()
        if (resource["python"], resource["unicode"]) != (built["python"], built["unicode"]):
            self.skipTest("The resource was written by another Python or Unicode version")
        self.assertEqual(resource, built)

    def test_an_existing_output_file_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            text_fixture.write(path)
            first = path.read_bytes()
            with self.assertRaises(FileExistsError):
                text_fixture.write(path)
            self.assertEqual(path.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
