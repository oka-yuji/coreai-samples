from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from Tools.Conversion.adapters.jev27b.fixtures import CARD_EXAMPLES, SEED, digest, load_rows
from Tools.Conversion.adapters.jev27b.typed_head import band

FIXTURES = Path(__file__).resolve().parents[1] / "recipes/jev27b/fixtures"
ROW_KEYS = {"id", "source", "family", "parts", "cut", "kind", "n_options", "sha256", "tokens", "band", "has_target"}
OPTION_COUNTS = {"noul": {2}, "score": {6}, "choice": set(range(2, 17))}
EXPECTED = {
    "F": {"rows": 302, "source": {"test_set_30k": 240, "jglue:JNLI": 8, "jglue:JCommonsenseQA": 8, "jglue:JSTS": 8,
                                  "synthetic-long": 30, "boundary": 6, "card-example": 2}},
    "Q": {"rows": 900, "source": {"test_set_30k": 900}, "kind": {"noul": 300, "choice": 300, "score": 300}},
    "Cal": {"rows": 3000, "source": {"calibration": 3000}, "kind": {"noul": 1000, "choice": 1000, "score": 1000}},
    "H": {"rows": 1500, "source": {"test": 1500}, "kind": {"noul": 500, "choice": 500, "score": 500}},
    "J": {"rows": 600, "source": {"jglue:JNLI": 200, "jglue:JCommonsenseQA": 200, "jglue:JSTS": 200}},
}


def load(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def corpus_ids(rows):
    return {row["id"] for row in rows if row["source"] == "test_set_30k"} | {part for row in rows for part in row["parts"] or []}


class FixtureSchemaTests(unittest.TestCase):
    def test_every_set_has_the_row_schema_and_a_matching_summary(self):
        for name, expected in EXPECTED.items():
            with self.subTest(set=name):
                data = load(name)
                self.assertEqual((data["name"], data["version"], data["seed"]), (name, 1, SEED))
                rows = data["rows"]
                self.assertEqual(len({row["id"] for row in rows}), len(rows))
                for row in rows:
                    self.assertEqual(set(row), ROW_KEYS)
                    self.assertRegex(row["sha256"], r"\A[0-9a-f]{64}\Z")
                    self.assertIn(row["n_options"], OPTION_COUNTS[row["kind"]])
                    self.assertIs(type(row["tokens"]), int)
                    self.assertEqual(row["band"], band(row["tokens"]))
                    self.assertIs(type(row["has_target"]), bool)
                summary = {"rows": len(rows), "kind": Counter(row["kind"] for row in rows),
                           "source": Counter(row["source"] for row in rows),
                           "band": Counter(row["band"] or "rejected" for row in rows)}
                self.assertEqual(data["summary"], {key: dict(value) if isinstance(value, Counter) else value
                                                   for key, value in summary.items()})
                self.assertEqual(data["summary"]["rows"], expected["rows"])
                self.assertEqual(data["summary"]["source"], expected["source"])
                if "kind" in expected:
                    self.assertEqual(data["summary"]["kind"], expected["kind"])

    def test_regression_set_covers_kinds_bands_boundaries_and_card_examples(self):
        rows = load("F")["rows"]
        sampled = [row for row in rows if row["source"] == "test_set_30k"]
        self.assertEqual(Counter(row["kind"] for row in sampled), {"noul": 80, "choice": 80, "score": 80})
        for label in ("513-2048", "2049-4096", "4097-8192"):
            synthetic = [row for row in rows if row["source"] == "synthetic-long" and row["id"].split(":")[1] == label]
            self.assertEqual(len(synthetic), 10)
            self.assertEqual({row["band"] for row in synthetic}, {label})
            self.assertEqual({row["kind"] for row in synthetic}, {"noul", "choice", "score"})
            self.assertTrue(all(len(row["parts"]) > 1 and row["cut"] is None and not row["has_target"] for row in synthetic))
        boundary = {row["id"]: row for row in rows if row["source"] == "boundary"}
        self.assertEqual(sorted(row["n_options"] for key, row in boundary.items() if key.startswith("boundary:choice")), [2, 2, 16, 16])
        self.assertEqual((boundary["boundary:tokens8192"]["tokens"], boundary["boundary:tokens8192"]["band"]), (8192, "4097-8192"))
        self.assertEqual((boundary["boundary:tokens8193"]["tokens"], boundary["boundary:tokens8193"]["band"]), (8193, None))
        self.assertTrue(all(row["band"] is not None for row in rows if row["id"] != "boundary:tokens8193"))
        self.assertEqual({row["id"] for row in rows if row["source"] == "card-example"}, set(CARD_EXAMPLES))
        japanese = Counter((row["source"], row["kind"], row["n_options"]) for row in rows if row["source"].startswith("jglue:"))
        self.assertEqual(japanese, {("jglue:JNLI", "choice", 3): 8, ("jglue:JCommonsenseQA", "choice", 5): 8, ("jglue:JSTS", "score", 6): 8})

    def test_sampled_sets_are_balanced_across_families(self):
        for name, source in (("F", "test_set_30k"), ("Q", "test_set_30k")):
            rows = [row for row in load(name)["rows"] if row["source"] == source]
            for kind in ("noul", "choice", "score"):
                counts = Counter(row["family"] for row in rows if row["kind"] == kind).values()
                with self.subTest(set=name, kind=kind):
                    self.assertLessEqual(max(counts) - min(counts), 1)

    def test_quality_and_japanese_sets_do_not_overlap_the_regression_set(self):
        f, q, j = load("F")["rows"], load("Q")["rows"], load("J")["rows"]
        self.assertFalse(corpus_ids(f) & {row["id"] for row in q})
        self.assertFalse({row["sha256"] for row in f} & {row["sha256"] for row in q})
        japanese = [row for row in f if row["source"].startswith("jglue:")]
        self.assertFalse({row["id"] for row in japanese} & {row["id"] for row in j})
        self.assertFalse({row["sha256"] for row in japanese} & {row["sha256"] for row in j})
        self.assertEqual(Counter((row["source"], row["kind"], row["n_options"]) for row in j),
                         {("jglue:JNLI", "choice", 3): 200, ("jglue:JCommonsenseQA", "choice", 5): 200, ("jglue:JSTS", "score", 6): 200})

    def test_fixtures_carry_ids_and_hashes_but_no_text(self):
        for path in FIXTURES.glob("*.json"):
            if path.name == "jglue.json":
                continue
            with self.subTest(path=path.name):
                self.assertIsNone(re.search(r'"(state|question|options|target|sentence1|sentence2)"', path.read_text()))


class FixtureRestoreTests(unittest.TestCase):
    def write_sources(self, folder):
        corpus, jglue = folder / "corpus", folder / "jglue"
        rows = [{"id": "a", "kind": "noul", "state": "first", "question": "q?", "options": ["false", "true"],
                 "target": [0.25, 0.75], "family": "f", "domain": "d", "source": "yuri_v3"},
                {"id": "b", "kind": "choice", "state": "second state", "question": "pick", "options": ["x", "y"],
                 "target": [0.5, 0.5], "family": "f", "domain": "d", "source": "openjev_v2"}]
        corpus.mkdir()
        (corpus / "test_set_30k.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        task = jglue / "jnli.json"
        task.parent.mkdir()
        task.write_text(json.dumps({"sentence_pair_id": "7", "sentence1": "p", "sentence2": "h", "label": "neutral"}) + "\n")
        data = task.read_bytes()
        pins = {"files": {"JNLI": {"path": "jnli.json", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}},
                "jnli_labels": ["entailment", "contradiction", "neutral"], "jnli_options": ["E", "C", "N"]}
        return corpus, jglue, pins

    def fixture(self, folder, pins, entries):
        path = folder / "fixtures/F.json"
        path.parent.mkdir(exist_ok=True)
        (path.parent / "jglue.json").write_text(json.dumps(pins))
        path.write_text(json.dumps({"rows": entries}))
        return path

    def test_rows_are_rebuilt_from_corpus_parts_cuts_jglue_rules_and_card_examples(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            corpus, jglue, pins = self.write_sources(folder)
            expected = [
                {"id": "a", "kind": "noul", "state": "first", "question": "q?", "options": ["false", "true"]},
                {"id": "long", "kind": "noul", "state": "first\n\nsecond", "question": "q?", "options": ["false", "true"]},
                {"id": "jglue:JNLI:7", "kind": "choice", "state": "p", "question": "h", "options": ["E", "C", "N"]},
                {"id": "card-example:sku", **{key: CARD_EXAMPLES["card-example:sku"][key]
                                              for key in ("kind", "state", "question", "options")}}]
            entries = [{"id": "a", "source": "test_set_30k", "parts": None, "cut": None},
                       {"id": "long", "source": "boundary", "parts": ["a", "b"], "cut": 6},
                       {"id": "jglue:JNLI:7", "source": "jglue:JNLI", "parts": None, "cut": None},
                       {"id": "card-example:sku", "source": "card-example", "parts": None, "cut": None}]
            for entry, row in zip(entries, expected):
                entry.update(kind=row["kind"], n_options=len(row["options"]), sha256=digest(row))
            rows = load_rows(self.fixture(folder, pins, entries), corpus, jglue)
            self.assertEqual([{key: row[key] for key in expected[0]} for row in rows], expected)
            self.assertEqual(rows[0]["target"], [0.25, 0.75])
            self.assertEqual(rows[2]["label"], 2)
            self.assertNotIn("target", rows[1])

    def test_restoring_a_changed_row_or_without_jglue_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            corpus, jglue, pins = self.write_sources(folder)
            row = {"kind": "noul", "state": "first", "question": "q?", "options": ["false", "true"]}
            entry = {"id": "a", "source": "test_set_30k", "parts": None, "cut": None, "kind": "noul", "n_options": 2,
                     "sha256": digest({**row, "state": "other"})}
            with self.assertRaises(ValueError):
                load_rows(self.fixture(folder, pins, [entry]), corpus, jglue)
            japanese = {"id": "jglue:JNLI:7", "source": "jglue:JNLI", "parts": None, "cut": None, "kind": "choice",
                        "n_options": 3, "sha256": "0" * 64}
            with self.assertRaises(ValueError):
                load_rows(self.fixture(folder, pins, [japanese]), corpus)


if __name__ == "__main__":
    unittest.main()
