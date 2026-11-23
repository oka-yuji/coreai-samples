import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

from Tools.Conversion.adapters.gemma4_31b import baseline, layer1_compare
from Tools.Conversion.storage import read_json, write_json
from Tools.LLMEvaluation import layer1
from Tools.LLMEvaluation.tests.test_layer1 import (BUNDLE, bounded, digest, fixture, generate, inode, release, replaced,
                                                   save, touched, watched)

RECIPE = Path(__file__).resolve().parents[1] / "recipes/gemma4_31b/model.json"
ROOT = Path(__file__).resolve().parents[3]
VOCABULARY = 16
LAYER1 = BUNDLE["contract"]["layer1"]
LONGEST = ("long-ja-16k-0001", "long-ja-16k-0002", "synth-long-16k-0001", "synth-long-16k-0002")
CHECKS = ("non_finite_values", "max_probability_error", "band_mean_max_probability_error", "significant_top1_mismatches")
LIMITS = CHECKS + ("near_tie_probability_margin",)
FORMATS = {"float16": "<f2", "float32": "<f4"}
SCHEMA = "gemma4-31b-layer1-logits/1"


def row(*leading):
    rest = (1 - sum(leading)) / (VOCABULARY - len(leading))
    return np.log(np.array(list(leading) + [rest] * (VOCABULARY - len(leading)))).astype(np.float32)


def entry(result, name):
    return next(item for item in result["items"] if item["id"] == name)


def passed_checks(result):
    return {name: result["checks"][name]["passed"] for name in CHECKS}


class RunIdentityTests(unittest.TestCase):
    def test_the_forced_input_binds_a_run_as_the_baseline_tool_records_it_in_progress_json(self):
        run = {"id": "c3febd49", "contract": {"version": "G31-QC-v1", "sha256": "a" * 64}, "set": "primary",
               "role": "reference", "asset": {"weights": {"model.safetensors": "b" * 64}}, "inputs": {"count": 2, "sha256": "c" * 64},
               "implementation": {"tool": "baseline", "commit": "d" * 40, "environment": {"python": "3.11.9", "os": "\u00e9"}},
               "call_shape": {"prefill_chunk": 1024}, "order": {"last": ["long-ja-4k"]}, "items": 2,
               "started": "2026-11-22T07:14:47Z", "finished": "2026-11-22T15:40:04Z"}
        self.assertEqual(layer1.run_identity(run), baseline.identity(run))
        self.assertEqual(layer1.run_identity({key: value for key, value in run.items() if key not in baseline.SESSION[1:]}),
                         baseline.identity(run))
        self.assertNotEqual(layer1.run_identity({**run, "id": "other"}), baseline.identity(run))


@unittest.skipIf(np is None, "NumPy is required for the comparison")
class Comparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        contract = read_json(RECIPE)
        contract["config"]["vocab_size"] = VOCABULARY
        cls.contract = root / "model.json"
        write_json(cls.contract, contract)
        cls.inputs, cls.reference = save(fixture(), root / "sources")
        cls.forced = root / "forced"
        layer1.write(BUNDLE, cls.inputs, cls.reference, cls.forced)
        cls.records = [json.loads(line) for line in (cls.forced / "forced.jsonl").read_text().splitlines()]
        cls.forced_sha256 = hashlib.sha256((cls.forced / "forced.jsonl").read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.fresh()

    def fresh(self, records=None):
        self.folder = Path(tempfile.mkdtemp(dir=self.temporary.name))
        generator = np.random.default_rng(697)
        self.baseline, self.candidate = {}, {}
        for record in records or self.records:
            values = generator.normal(size=(record["rows"], VOCABULARY))
            values[np.arange(record["rows"]), generator.integers(VOCABULARY, size=record["rows"])] += 4
            self.baseline[record["id"]] = values.astype(np.float32)
            self.candidate[record["id"]] = (values + generator.normal(scale=1e-3, size=values.shape)).astype(np.float32)

    def side(self, name, arrays, dtype="float32", role="anchor", edit=None, records=None):
        folder = self.folder / name
        folder.mkdir()
        listed = []
        for record in records or self.records:
            if record["id"] not in arrays:
                continue
            array = arrays[record["id"]]
            data = np.ascontiguousarray(array, dtype=FORMATS[dtype]).tobytes()
            (folder / f"{record['id']}.{dtype}").write_bytes(data)
            listed.append({"id": record["id"], "file": f"{record['id']}.{dtype}", "dtype": dtype,
                           "shape": list(array.shape), "sha256": hashlib.sha256(data).hexdigest(),
                           "tokens_sha256": record["tokens_sha256"]})
        record = {"schema": SCHEMA, "role": role, "weights": "synthetic", "forced_sha256": self.forced_sha256,
                  "items": listed}
        if edit is not None:
            edit(record)
        (folder / "logits.json").write_text(json.dumps(record))
        return folder

    def compared(self, candidate_dtype="float32", edit=None, forced=None, after=None, sources=None, records=None):
        output, (inputs, run) = self.folder / "result", sources or (self.inputs, self.reference)
        baseline = self.side("baseline", self.baseline, records=records)
        candidate = self.side("candidate", self.candidate, candidate_dtype, "fp16", edit, records)
        if after is not None:
            after(candidate)
        result = layer1_compare.compare(forced or self.forced, inputs, run, baseline, candidate, output, self.contract)
        self.assertEqual(read_json(output / "layer1.json"), result)
        return result

    def failed(self, result, text):
        self.assertFalse(result["passed"])
        self.assertTrue(any(text in problem for problem in result["problems"]), result["problems"])
        self.assertTrue((self.folder / "result/layer1.json").is_file())


class VerdictTests(Comparison):
    def test_close_logits_pass_and_a_band_without_items_is_reported_as_not_evaluated(self):
        result = self.compared(candidate_dtype="float16")
        self.assertTrue(result["passed"], result["problems"])
        self.assertEqual(result["problems"], [])
        self.assertEqual({band["band"]: band["state"] for band in result["bands"]},
                         {"1-512": "passed", "513-2048": "not_evaluated", "2049-4096": "passed", "4097-16896": "passed"})
        empty = result["bands"][1]
        self.assertEqual((empty["items"], empty["compared"], empty["positions"], empty["mean_max_probability_error"]),
                         (0, 0, 0, None))
        self.assertEqual([band["items"] for band in result["bands"]], [72, 0, 12, 4])
        self.assertEqual([item["id"] for item in result["items"]], [record["id"] for record in self.records])
        self.assertEqual(result["overall"]["positions"], sum(record["rows"] for record in self.records))
        self.assertEqual(result["overall"]["top1_agreement"], 1.0)
        self.assertEqual(set(LAYER1["report_only"]) - set(result["overall"]), set())
        self.assertEqual(set(LAYER1["report_only"]) - set(result["bands"][0]), set())
        self.assertEqual({side: result[side]["role"] for side in ("baseline", "candidate")},
                         {"baseline": "anchor", "candidate": "fp16"})
        self.assertEqual(result["forced"]["sha256"], self.forced_sha256)
        self.assertEqual(result["limits"], {key: LAYER1[key] for key in LIMITS})
        self.assertLess(result["checks"]["max_probability_error"]["value"], LAYER1["max_probability_error"])

    def test_a_near_tie_mismatch_alone_passes(self):
        self.baseline["ja-choice-0001"][0] = row(0.40, 0.395)
        self.candidate["ja-choice-0001"][0] = row(0.395, 0.40)
        result = self.compared()
        self.assertTrue(result["passed"], result["problems"])
        self.assertEqual(result["overall"]["near_tie_count"], 1)
        self.assertEqual(entry(result, "ja-choice-0001")["near_tie_mismatches"], 1)
        self.assertEqual(result["checks"]["significant_top1_mismatches"]["value"], 0)
        positions = result["overall"]["positions"]
        self.assertEqual(result["overall"]["top1_agreement"], (positions - 1) / positions)

    def test_a_top1_mismatch_that_is_not_a_near_tie_fails(self):
        for name, before, after in (("other top two", row(0.30, 0.295, 0.29), row(0.29, 0.295, 0.30)),
                                    ("wide gap", row(0.40, 0.38), row(0.385, 0.395))):
            with self.subTest(name):
                self.fresh()
                self.baseline["ja-reading-0001"][0], self.candidate["ja-reading-0001"][0] = before, after
                result = self.compared()
                self.assertFalse(result["passed"])
                self.assertEqual(result["problems"], [])
                self.assertEqual(passed_checks(result), {**dict.fromkeys(CHECKS, True),
                                                         "significant_top1_mismatches": False})
                self.assertEqual(entry(result, "ja-reading-0001")["significant_mismatches"], 1)

    def test_a_position_whose_probability_error_exceeds_the_limit_fails(self):
        self.baseline["extract-ja-0001"][1] = row(0.6, 0.2)
        self.candidate["extract-ja-0001"][1] = row(0.55, 0.25)
        result = self.compared()
        self.assertFalse(result["passed"])
        self.assertEqual(result["problems"], [])
        self.assertEqual(passed_checks(result), {**dict.fromkeys(CHECKS, True), "max_probability_error": False})
        self.assertAlmostEqual(result["checks"]["max_probability_error"]["value"], 0.05, places=6)
        self.assertAlmostEqual(entry(result, "extract-ja-0001")["max_probability_error"], 0.05, places=6)
        overall = result["overall"]
        self.assertAlmostEqual(overall["kl_max"], 0.6 * np.log(0.6 / 0.55) + 0.2 * np.log(0.2 / 0.25), places=5)
        self.assertAlmostEqual(overall["max_abs_logit_difference"], np.log(0.25 / 0.2), places=5)
        self.assertLess(overall["kl_p99"], overall["kl_max"] / 100)
        self.assertLess(abs(overall["kl_mean"] - overall["kl_max"] / overall["positions"]), overall["kl_p99"])

    def test_a_band_whose_mean_error_exceeds_the_limit_fails_although_every_position_is_within_the_maximum(self):
        for name in LONGEST:
            self.baseline[name][:] = row(0.6, 0.2)
            self.candidate[name][:] = row(0.585, 0.215)
        result = self.compared()
        self.assertFalse(result["passed"])
        self.assertEqual(passed_checks(result), {**dict.fromkeys(CHECKS, True),
                                                 "band_mean_max_probability_error": False})
        self.assertEqual([band["state"] for band in result["bands"]], ["passed", "not_evaluated", "passed", "failed"])
        self.assertAlmostEqual(result["bands"][3]["mean_max_probability_error"], 0.015, places=6)

    def test_non_finite_values_on_either_side_fail(self):
        for side, value in (("baseline", np.nan), ("candidate", np.inf)):
            with self.subTest(side):
                self.fresh()
                getattr(self, side)["code-fix-0001"][0, 3] = value
                result = self.compared()
                self.assertFalse(result["checks"]["non_finite_values"]["passed"])
                self.assertEqual(result["checks"]["non_finite_values"]["value"], 1)
                self.assertEqual(entry(result, "code-fix-0001")["non_finite"],
                                 {"baseline": int(side == "baseline"), "candidate": int(side == "candidate")})
                self.failed(result, "code-fix-0001: non-finite values")
                self.assertIsNotNone(entry(result, "code-fix-0002")["max_probability_error"])

    def test_the_limits_are_read_from_the_contract(self):
        cases = (("max_probability_error", 0.1, "extract-ja-0001", row(0.6, 0.2), row(0.55, 0.25), True),
                 ("near_tie_probability_margin", 0.001, "ja-choice-0001", row(0.40, 0.395), row(0.395, 0.40), False))
        for key, value, name, before, after, passes in cases:
            with self.subTest(key):
                self.fresh()
                self.baseline[name][0], self.candidate[name][0] = before, after
                changed = copy.deepcopy(BUNDLE)
                changed["contract"]["layer1"][key] = value
                with mock.patch.object(layer1_compare.items, "load", return_value=changed):
                    result = self.compared()
                self.assertEqual(result["passed"], passes, result["checks"])
                self.assertEqual(result["limits"][key], value)
        self.fresh()
        for name in LONGEST:
            self.baseline[name][:], self.candidate[name][:] = row(0.6, 0.2), row(0.585, 0.215)
        changed = copy.deepcopy(BUNDLE)
        changed["contract"]["layer1"]["band_mean_max_probability_error"] = 0.02
        with mock.patch.object(layer1_compare.items, "load", return_value=changed):
            self.assertTrue(self.compared()["passed"])


class CorrespondenceTests(Comparison):
    def test_logits_made_from_another_forced_input_fail(self):
        self.failed(self.compared(edit=lambda record: record.update(forced_sha256="0" * 64)),
                    "the candidate logits were made from another forced input")

    def test_a_missing_repeated_or_extra_item_fails_and_the_other_items_are_compared(self):
        def extra(record):
            record["items"].append({**record["items"][0], "id": "ja-choice-0099"})

        def repeated(record):
            record["items"].append(dict(record["items"][0]))

        for name, change, text in (("missing", None, "free-ja-0003: the candidate has no logits for the item"),
                                   ("extra", extra, "the candidate lists ja-choice-0099, which is not a layer 1 item"),
                                   ("repeated", repeated, "the candidate lists ja-choice-0001 twice")):
            with self.subTest(name):
                self.fresh()
                if change is None:
                    del self.candidate["free-ja-0003"]
                result = self.compared(edit=change)
                self.failed(result, text)
                self.assertIsNotNone(entry(result, "free-ja-0004")["max_probability_error"])

    def test_logits_with_other_rows_or_another_vocabulary_fail(self):
        for name, change in (("rows", lambda values: np.concatenate([values, values[:1]])),
                             ("vocabulary", lambda values: values[:, :-1])):
            with self.subTest(name):
                self.fresh()
                values = self.candidate["synth-format-0001"]
                self.candidate["synth-format-0001"] = change(values)
                shape = list(self.candidate["synth-format-0001"].shape)
                result = self.compared()
                self.failed(result, f"synth-format-0001: the candidate logits have the shape {shape}, "
                                    f"not {[len(values), VOCABULARY]}")
                self.assertIsNotNone(entry(result, "synth-format-0002")["max_probability_error"])

    def test_a_file_that_differs_from_its_sha256_or_other_tokens_fail(self):
        def other(key):
            return lambda record: record["items"][0].update({key: "0" * 64})

        for key, text in (("sha256", "the candidate file differs from its SHA-256"),
                          ("tokens_sha256", "the candidate read other tokens")):
            with self.subTest(key):
                self.fresh()
                self.failed(self.compared(edit=other(key)), f"ja-choice-0001: {text}")

    def test_an_unreadable_array_is_recorded_by_the_name_of_its_error(self):
        result = self.compared(edit=lambda record: record["items"][2].update(file="missing.float32"))
        self.failed(result, "ja-choice-0003: the candidate logits are unreadable (FileNotFoundError)")

    def test_a_record_with_a_missing_wrong_or_oversized_field_fails(self):
        def item(key, value):
            return lambda record: record["items"][0].update({key: value})

        cases = (("no role", lambda record: record.pop("role"), "the candidate record is malformed (role)"),
                 ("another schema", lambda record: record.update(schema="other/1"),
                  f"the candidate record is not a {layer1_compare.LOGITS} record"),
                 ("another role", lambda record: record.update(role="teacher"),
                  "the candidate role is not a role of the contract"),
                 ("no item list", lambda record: record.update(items={}), "the candidate record has no item list"),
                 ("a flag in the shape", item("shape", [True, VOCABULARY]),
                  "item 0 of the candidate record is malformed (shape)"),
                 ("a huge shape", item("shape", [2**64, VOCABULARY]), "item 0 of the candidate record is malformed (shape)"),
                 ("another dtype", item("dtype", "bfloat16"), "the candidate dtype 'bfloat16' is not float16 or float32"),
                 ("a path", item("file", "../baseline/ja-choice-0001.float32"),
                  "the candidate file name is not a name in its folder"))
        for name, change, text in cases:
            with self.subTest(name):
                self.fresh()
                self.failed(self.compared(edit=change), text)

    def test_a_record_that_is_not_json_fails(self):
        def broken(record):
            record.clear()
            record["schema"] = float("nan")

        self.failed(self.compared(edit=broken), "the candidate logits.json is unreadable")

    def test_a_forced_input_of_another_contract_or_with_fewer_items_fails(self):
        copied = self.folder / "forced"
        shutil.copytree(self.forced, copied)
        summary = read_json(copied / "forced.json")
        summary["contract"]["version"] = "G31-QC-v0"
        (copied / "forced.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        self.failed(self.compared(forced=copied), "forced.json differs from the forced input made again")
        self.fresh()
        copied = self.folder / "forced"
        shutil.copytree(self.forced, copied)
        lines = (copied / "forced.jsonl").read_text().splitlines(keepends=True)
        (copied / "forced.jsonl").write_text("".join(lines[1:]))
        events, patched = watched()
        with patched:
            result = self.compared(forced=copied)
        self.failed(result, "forced.jsonl differs from the forced input made again")
        self.assertEqual(touched(events, inode(copied / "forced.jsonl")), [])
        self.assertEqual(len(touched(events, inode(copied / "forced.json"))), 2)
        self.assertFalse(any("made from another forced input" in problem for problem in result["problems"]))
        self.assertIsNotNone(entry(result, self.records[0]["id"])["max_probability_error"])

    def forged(self, change):
        copied = self.folder / "forced"
        copied.mkdir()
        records = [copy.deepcopy(record) for record in self.records]
        change(records)
        data = "".join(json.dumps(record) + "\n" for record in records).encode()
        (copied / "forced.jsonl").write_bytes(data)
        summary = read_json(self.forced / "forced.json")
        summary["forced"]["sha256"] = hashlib.sha256(data).hexdigest()
        summary["bands"] = {f"{low}-{high}": sum(low <= record["input_tokens"] <= high for record in records)
                            for low, high in LAYER1["bands"]}
        write_json(copied / "forced.json", summary)
        self.forced_sha256 = summary["forced"]["sha256"]
        return copied, records

    def test_a_forced_summary_that_differs_from_the_one_made_again_fails(self):
        def bands(summary):
            summary["bands"]["1-512"] -= 1
            summary["bands"]["513-2048"] += 1

        cases = (("bands", bands), ("categories", lambda summary: summary["categories"].update({"free-ja": 7})),
                 ("count", lambda summary: summary["forced"].update(count=summary["forced"]["count"] - 1)),
                 ("a float count", lambda summary: summary["forced"].update(count=float(summary["forced"]["count"]))),
                 ("set", lambda summary: summary.update(set="reserve")),
                 ("another run", lambda summary: summary["run"].update(id="run-2")))
        for name, change in cases:
            with self.subTest(name):
                self.fresh()
                copied = self.folder / "forced"
                shutil.copytree(self.forced, copied)
                summary = read_json(copied / "forced.json")
                change(summary)
                (copied / "forced.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
                result = self.compared(forced=copied)
                self.failed(result, "forced.json differs from the forced input made again from the input records and the run")
                self.assertEqual(len(result["problems"]), 1)

    def test_a_consistent_forced_input_with_another_input_length_for_a_long_item_fails(self):
        def shortened(records):
            record = next(record for record in records if record["id"] == "long-ja-4k-0001")
            ids, tail = record["tokens"][:512], record["targets"][:record["rows"] - 1]
            record.update(tokens=ids + tail, input_tokens=512, start=511, input_sha256=digest(ids),
                          tokens_sha256=digest(ids + tail))

        forced, records = self.forged(shortened)
        record = next(record for record in records if record["id"] == "long-ja-4k-0001")
        self.assertEqual(layer1.inconsistencies(record, LAYER1), [])
        self.assertEqual(layer1.band(LAYER1["bands"], record["input_tokens"]), "1-512")
        result = self.compared(forced=forced, records=records)
        for text in ("forced.jsonl differs from the forced input made again from the input records and the run",
                     "forced.json differs from the forced input made again from the input records and the run",
                     "the candidate logits were made from another forced input",
                     "long-ja-4k-0001: the baseline read other tokens"):
            self.failed(result, text)
        self.assertEqual(entry(result, "long-ja-4k-0001")["band"], "2049-4096")

    def test_a_broken_forced_record_fails_as_a_difference_from_the_forced_input_made_again(self):
        def moved(records):
            records[0]["input_tokens"] = 3000

        def changed(records):
            records[0]["rows"], records[1]["input_tokens"] = LAYER1["positions"]["max_per_item"] + 1, 20000

        for name, change in (("moved", moved), ("changed", changed)):
            with self.subTest(name):
                self.fresh()
                forced, _ = self.forged(change)
                result = self.compared(forced=forced)
                self.failed(result, "forced.jsonl differs from the forced input made again from the input records and the run")
                self.failed(result, "the baseline logits were made from another forced input")
                self.assertIsNotNone(entry(result, self.records[0]["id"])["max_probability_error"])
                self.assertEqual([band["items"] for band in result["bands"]], [72, 0, 12, 4])

    def test_a_forced_input_made_from_another_reference_run_fails_and_its_own_sources_pass(self):
        state = fixture()
        generate(state, "ja-reading-0002", [31, 32, 33, 34])
        inputs, run = save(state, Path(self.temporary.name) / "other-sources")
        other = Path(self.temporary.name) / "other-forced"
        layer1.write(BUNDLE, inputs, run, other)
        records = [json.loads(line) for line in (other / "forced.jsonl").read_text().splitlines()]
        for sources, passes in (((self.inputs, self.reference), False), ((inputs, run), True)):
            with self.subTest(passes=passes):
                self.fresh(records)
                self.forced_sha256 = hashlib.sha256((other / "forced.jsonl").read_bytes()).hexdigest()
                result = self.compared(forced=other, sources=sources, records=records)
                self.assertEqual(result["passed"], passes, result["problems"])
                if not passes:
                    self.failed(result, "forced.jsonl differs from the forced input made again from the input records and the run")
        shutil.rmtree(Path(self.temporary.name) / "other-sources")
        shutil.rmtree(other)

    def test_sources_that_fail_their_checks_leave_a_failed_record(self):
        run = self.folder / "run"
        shutil.copytree(self.reference, run)
        progress = read_json(run / "progress.json")
        progress["next_sha256"] = None
        write_json(run / "progress.json", progress)
        result = self.compared(sources=(self.inputs, run))
        self.failed(result, "the forced input cannot be made again from the input records and the run (progress.json does "
                            "not confirm these output records of this finished run: next_sha256)")
        self.assertTrue(all(item["max_probability_error"] is None for item in result["items"]))
        self.assertFalse(any("made from another forced input" in problem for problem in result["problems"]))

    def test_an_array_file_of_another_size_or_kind_is_not_read(self):
        name, rows = self.records[3]["id"], self.records[3]["rows"]
        size, extra = rows * VOCABULARY * 4, 4 * VOCABULARY * 16384

        def bigger(folder):
            with (folder / f"{name}.float32").open("ab") as handle:
                handle.write(bytes(extra))

        def smaller(folder):
            path = folder / f"{name}.float32"
            path.write_bytes(path.read_bytes()[:-4])

        def directory(folder):
            (folder / f"{name}.float32").unlink()
            (folder / f"{name}.float32").mkdir()

        for kind, change, text in (("bigger", bigger, f"the candidate file {name}.float32 holds {size + extra} B, not {size} B"),
                                   ("smaller", smaller, f"the candidate file {name}.float32 holds {size - 4} B, not {size} B"),
                                   ("folder", directory, f"the candidate file {name}.float32 is not a regular file")):
            with self.subTest(kind):
                self.fresh()
                targets = []

                def changed(folder):
                    change(folder)
                    targets.append(inode(folder / f"{name}.float32"))

                events, patched = watched()
                with patched:
                    result = self.compared(after=changed)
                self.failed(result, f"{name}: {text}")
                self.assertEqual(touched(events, targets[0]), [])
                self.assertEqual([kind for _, kind, _ in touched(events, inode(self.folder / "baseline" / f"{name}.float32"))],
                                 ["open", "read"])

    def test_a_file_that_grows_after_its_size_was_checked_fails(self):
        name, real, grown_files = self.records[3]["id"], os.fstat, []

        def grown(folder):
            with (folder / f"{name}.float32").open("ab") as handle:
                handle.write(bytes(4))
            status = os.stat(folder / f"{name}.float32")
            grown_files.append((status.st_dev, status.st_ino))

        def reported(descriptor):
            status = real(descriptor)
            if (status.st_dev, status.st_ino) in grown_files:
                return os.stat_result(tuple(status)[:6] + (status.st_size - 4,) + tuple(status)[7:10])
            return status

        with mock.patch.object(layer1.os, "fstat", side_effect=reported):
            result = self.compared(after=grown)
        self.failed(result, f"{name}: the candidate file {name}.float32 changed while it was read")

    def test_no_array_of_an_item_with_a_problem_in_its_entries_is_read(self):
        def changed(record):
            record["items"][0]["dtype"] = "bfloat16"
            record["items"][1]["tokens_sha256"] = "0" * 64

        events, patched = watched()
        with patched:
            result = self.compared(edit=changed)
        self.failed(result, f"{self.records[0]['id']}: the candidate dtype 'bfloat16' is not float16 or float32")
        self.failed(result, f"{self.records[1]['id']}: the candidate read other tokens")
        for record in self.records[:2]:
            for side in ("baseline", "candidate"):
                self.assertEqual(touched(events, inode(self.folder / side / f"{record['id']}.float32")), [])
        self.assertEqual(len(touched(events, inode(self.folder / "candidate" / f"{self.records[2]['id']}.float32"))), 2)

    def test_the_same_folder_on_both_sides_fails(self):
        folder = self.side("both", self.baseline)
        result = layer1_compare.compare(self.forced, self.inputs, self.reference, folder, folder, self.folder / "result",
                                        self.contract)
        self.failed(result, "the baseline and the candidate are the same folder")


class BoundaryTests(Comparison):
    def test_an_existing_output_or_link_is_refused_before_anything_is_read(self):
        missing = self.folder / "missing"
        for name, make in (("folder", lambda path: path.mkdir()),
                           ("dangling-link", lambda path: path.symlink_to(self.folder / "nowhere"))):
            with self.subTest(name):
                output = self.folder / name
                make(output)
                with mock.patch.object(layer1_compare.items, "load", side_effect=AssertionError("read")):
                    with self.assertRaises(FileExistsError):
                        layer1_compare.compare(missing, missing, missing, missing, missing, output, self.contract)
                self.assertFalse(os.path.exists(output / "layer1.json"))

    def test_a_file_that_is_not_a_regular_file_fails_without_waiting(self):
        name = self.records[2]["id"]

        def source(folder):
            run = folder / "run"
            shutil.copytree(self.reference, run)
            return run / "outputs.jsonl", {"sources": (self.inputs, run)}

        def given(folder):
            copied = folder / "given"
            shutil.copytree(self.forced, copied)
            return copied / "forced.jsonl", {"forced": copied}

        def logits(folder):
            return folder / "candidate" / "logits.json", {}

        def array(folder):
            return folder / "candidate" / f"{name}.float32", {}

        cases = (("fifo", source, "the forced input cannot be made again from the input records and the run "
                                  "(outputs.jsonl is not a regular file)"),
                 ("folder", source, "(outputs.jsonl is not a regular file)"),
                 ("missing", source, "the forced input cannot be made again from the input records and the run "
                                     "(FileNotFoundError)"),
                 ("broken link", source, "(FileNotFoundError)"),
                 ("fifo", given, "forced.jsonl differs from the forced input made again from the input records and the "
                                 "run (forced.jsonl is not a regular file)"),
                 ("fifo", logits, "the candidate logits.json is unreadable (logits.json is not a regular file)"),
                 ("fifo", array, f"{name}: the candidate file {name}.float32 is not a regular file"))
        for kind, where, text in cases:
            with self.subTest(kind=kind, file=where.__name__):
                self.fresh()
                folder, waited_on = self.folder, []
                path, options = (None, {}) if where in (logits, array) else where(folder)

                def change(candidate):
                    target = where(folder)[0]
                    waited_on.append(target)
                    replaced(target, kind)

                if path is not None:
                    replaced(path, kind)
                    waited_on.append(path)
                events, patched = watched()
                with patched:
                    thread, outcome = bounded(lambda: self.compared(after=None if path is not None else change, **options))
                    self.assertFalse(release(waited_on[-1], thread), "the comparison waited on a file")
                self.assertNotIn("error", outcome, outcome.get("error"))
                self.failed(outcome["value"], text)
                if kind in ("fifo", "folder"):
                    self.assertEqual(touched(events, inode(waited_on[-1])), [])

    def test_frozen_files_that_do_not_validate_stop_before_the_output_is_made(self):
        output = self.folder / "result"
        with mock.patch.object(layer1_compare.checks, "problems", return_value=["a changed contract"]):
            with self.assertRaises(ValueError):
                layer1_compare.compare(self.forced, self.inputs, self.reference, self.folder, self.folder, output, self.contract)
        self.assertFalse(os.path.lexists(output))

    def test_an_error_outside_the_items_still_leaves_a_failed_record(self):
        with mock.patch.object(layer1_compare, "forced_input", side_effect=RuntimeError("stopped")):
            result = self.compared()
        self.assertEqual(result["problems"], ["the comparison could not be made (RuntimeError)"])
        self.assertFalse(result["passed"])

    def test_an_error_in_one_item_is_recorded_by_its_name_and_the_other_items_are_compared(self):
        original = layer1_compare.judged
        calls = []

        def judged(*arguments):
            calls.append(1)
            if len(calls) == 1:
                raise MemoryError
            return original(*arguments)

        with mock.patch.object(layer1_compare, "judged", side_effect=judged):
            result = self.compared()
        self.failed(result, "ja-choice-0001: the item could not be compared (MemoryError)")
        self.assertEqual(sum(item["max_probability_error"] is not None for item in result["items"]), 87)

    def test_an_interruption_leaves_a_record_of_an_unfinished_comparison(self):
        output = self.folder / "result"
        with mock.patch.object(layer1_compare, "measured", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                layer1_compare.compare(self.forced, self.inputs, self.reference, self.side("baseline", self.baseline),
                                       self.side("candidate", self.candidate), output, self.contract)
        self.assertEqual(read_json(output / "layer1.json"),
                         {"schema": layer1_compare.SCHEMA, "passed": False, "problems": ["the comparison did not finish"]})

    def test_a_result_that_cannot_be_recorded_leaves_a_minimal_failed_record(self):
        calls = []

        def flaky(path, value):
            calls.append(value)
            if len(calls) == 1:
                raise ValueError("not JSON")
            write_json(path, value)

        with mock.patch.object(layer1_compare, "write_json", side_effect=flaky):
            result = self.compared()
        self.assertEqual(len(calls), 2)
        self.assertEqual(result, {"schema": layer1_compare.SCHEMA, "passed": False,
                                  "problems": ["the comparison could not be recorded (ValueError)"]})

    def test_the_command_exits_with_1_and_leaves_the_record_when_the_comparison_fails(self):
        output = self.folder / "result"
        process = subprocess.run([sys.executable, "-m", "Tools.Conversion.adapters.gemma4_31b.layer1_compare",
                                  "--forced", str(self.forced), "--inputs", str(self.inputs), "--run", str(self.reference),
                                  "--baseline", str(self.side("baseline", self.baseline)),
                                  "--candidate", str(self.side("candidate", self.candidate)), "--output", str(output)],
                                 cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertFalse(json.loads(process.stdout)["passed"])
        self.assertFalse(read_json(output / "layer1.json")["passed"])


if __name__ == "__main__":
    unittest.main()
