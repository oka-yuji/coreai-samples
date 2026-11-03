import json
from pathlib import Path
import unittest

from Tools.Conversion.adapters.jev27b.fixtures import digest
from Tools.Conversion.adapters.jev27b.swift_fixture import (BIAS, FIELDS, JAPANESE, WEIGHT, WIDTH, head_cases,
                                                            license_note, rejected_rows, synthetic_rows)
from Tools.Conversion.adapters.jev27b.typed_head import render

ROOT = Path(__file__).resolve().parents[3]
FOLDER = ROOT / "CoreAISamplesKit/Tests/CoreAIBackendTests/Fixtures/TypedDecision"
FIXTURES = ROOT / "Tools/Conversion/recipes/jev27b/fixtures"


def load(name, folder=FOLDER):
    return json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))


class SwiftFixtureTests(unittest.TestCase):
    def test_render_rows_hold_the_synthetic_rows_then_the_first_jglue_rows_of_f(self):
        document = load("render")
        rows = document["rows"]
        synthetic = synthetic_rows()
        self.assertEqual([{field: row[field] for field in FIELDS} for row in rows[:len(synthetic)]], synthetic)
        entries = {entry["id"]: entry for entry in load("F", FIXTURES)["rows"]}
        expected = [identifier for task, count in JAPANESE
                    for identifier in [key for key in entries if key.startswith(f"jglue:{task}:")][:count]]
        japanese = rows[len(synthetic):]
        self.assertEqual([row["id"] for row in japanese], expected)
        self.assertTrue(all(digest(row) == entries[row["id"]]["sha256"] for row in japanese))
        self.assertEqual(document["__license__"], license_note(load("jglue", FIXTURES)))
        for row in rows:
            self.assertEqual(row["text"], render(row["kind"], row["state"], row["question"], row["options"]))
            self.assertEqual(row["token_count"], len(row["tokens"]))

    def test_real_head_rows_repeat_their_render_rows_and_the_calibrated_temperatures(self):
        real, head = load("real-head"), load("head")
        rendered = {row["id"]: row for row in load("render")["rows"]}
        self.assertEqual(real["__license__"], license_note(load("jglue", FIXTURES)))
        self.assertEqual(real["temperatures"], head["temperatures"])
        self.assertEqual({row["kind"] for row in real["rows"]}, {"noul", "choice", "score"})
        for row in real["rows"]:
            self.assertEqual([row[field] for field in ("kind", "options", "tokens")],
                             [rendered[row["id"]][field] for field in ("kind", "options", "tokens")])
            self.assertEqual(len(row["hidden"]), 5120)

    def test_rejected_rows_and_the_synthetic_head_match_the_generator(self):
        self.assertEqual(load("reject")["rows"], rejected_rows())
        head = load("head")
        self.assertEqual((head["width"], head["weight"], head["bias"]), (WIDTH, WEIGHT, BIAS))
        self.assertEqual(head["rows"], head_cases(head["temperatures"]))
        self.assertEqual(next(row for row in head["rows"] if row["id"] == "tie")["top"], 1)


if __name__ == "__main__":
    unittest.main()
