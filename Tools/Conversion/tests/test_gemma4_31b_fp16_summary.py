import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from Tools.Conversion.adapters.gemma4_31b import export, fp16_summary, layers, native
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import PipelineError, write_json

SENTINEL = "private-example-sentinel"
SHARD = "model-00001-of-00002.safetensors"
MODEL = {"id": "google/gemma-4-31B-it", "revision": "a" * 40, "contract_sha256": "b" * 64,
         "verified_before_loading": {SHARD: {"bytes": 3, "sha256": "c" * 64}, "config.json": {"bytes": 1, "sha256": "d" * 64}}}
KINDS = ["sliding_attention"] * 5 + ["full_attention"]
STATE = {"dtype": "float16", "shape": [1, 1, 4, None, 512]}


def context_record():
    entry = {"positions": 26, "tokens_sha256": "e" * 64, "rows": list(range(26)), "calls": [[0, 26]], "seconds": 0.5,
             "logits": {"bytes": 10, "sha256": "f" * 64, "dtype": "float32", "shape": [26, 262144], "non_finite": 0}}
    return {"schema": layers.CONTEXT_SCHEMA, "implementation": "original", "layers": 6, "layer_types": KINDS, "model": MODEL,
            "weights": "every checkpoint tensor rounded to float16", "embed_scale": 73.5, "chunk": 1024,
            "versions": {"torch": "2.15.0"}, "inputs": {"short": entry, "context": {**entry, "positions": 16896}},
            "timing": {"total_s": 60.0}, "memory": {"max_rss_bytes": 10, "lifetime_peak_footprint_bytes": None}}


def export_record():
    function = {"inputs": {"input_ids": {"dtype": "int32", "shape": [1, None]}}, "outputs": {}, "states": {"keyCache": STATE}}
    return {"schema": export.SCHEMA, "passed": True, "checks": {"asset": True, "free_disk": True}, "problems": [],
            "model": MODEL, "layers": 6, "layer_types": KINDS, "functions": ["main"], "context": 16896, "precision": "float16",
            "compression_config": None, "upstream_revision": "1" * 40, "versions": {"coreai-core": "1.0.0b3"},
            "weights": {"tensors": 85, "elements": 9, "elements_by_dtype": {"torch.float16": 9}, "quantized_layouts": 0},
            "asset": {"functions": {"main": function}, "storage_types": {"Float16": 13, "Int32": 2},
                      "float16_minus_weights": -12},
            "tokenizer": {"tokenizer.json": "2" * 64},
            "metadata": {"name": "gemma4_31b_l6_fp16_main", "language": {"vocab_size": 262144, "max_context_length": 16896},
                         "source": {"hf_model_id": "google/gemma-4-31B-it", "revision": "a" * 40}, "compression": None},
            "bundle": {"folder": "bundle", "name": "gemma4_31b_l6_fp16_main", "main_hash": "3" * 64, "bytes": 4,
                       "files": {"gemma4_31b_l6_fp16_main.aimodel/main.mlirb": {"bytes": 4, "sha256": "4" * 64}}},
            "disk": {"required_free_bytes": 5, "lowest_free_bytes": 6},
            "timing": {"load_s": 1.0, "total_s": 2.0},
            "resources": {"max_rss_bytes": 7, "lifetime_peak_footprint_bytes": 8,
                          "swap": {"first_bytes": 0, "peak_bytes": 0, "increase_bytes": 0, "limit_increase_bytes": 2},
                          "monitor": {"samples": 3, "interval_s": 5, "aborts": 0, "failure": None, "unreadable_samples": 0,
                                      "peak": {"rss_bytes": 9}, "lowest": {"free_percent": 40,
                                                                           "free_bytes": {f"/{SENTINEL}/out": 6}}}}}


def comparison_record(export_hash, reference_hash):
    entry = {"name": "tail", "input": "context", "tokens": 16896, "start": 16832, "chunk": 1, "rows": 64, "passed": True,
             "problems": [], "checks": {"finite": True}, "max_abs_logit": 0.5, "non_finite": 0, "reference_non_finite": 0,
             "constant_rows": 0, "top1_mismatches": {"near_tie": 1, "outside": 0}, "known": {"rows": 0, "max_abs_logit": None},
             "seconds": 3.0, "output": {"bytes": 1, "sha256": "5" * 64},
             "verifier": {"cached": True, "prefix_function": "prefill", "prefix_calls": 5, "main_calls": 64,
                          "non_finite_count": 0, "load_s": 1.0, "prefix_s": 1.0, "scoring_s": 1.0, "specialize_s": 0.0,
                          "cache_query_error": SENTINEL}}
    first = copy.deepcopy(entry)
    first.update(name="short-c1", input="short", tokens=26, start=0, rows=26)
    del first["verifier"]["prefix_function"]
    return {"schema": native.SCHEMA, "passed": True, "problems": [], "criteria": {"max_abs_logit": 2.0},
            "known_call": {"ends": [97, 112], "shortest": 8}, "reference": {"sha256": reference_hash, "layers": 6},
            "export": {"sha256": export_hash, "layers": 6, "functions": ["main"], "main_hash": "3" * 64},
            "verifier": {"bytes": 1, "sha256": "6" * 64}, "conditions": [first, entry]}


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.folder))
        self.reference = self.folder / "context.json"
        write_json(self.reference, context_record())

    def run_files(self, label, record=None, origin=None, stage_role="export", manifest=None):
        artifacts = self.folder / label / "artifacts"
        write_json(artifacts / "export/export.json", record or export_record())
        data = (artifacts / "export/export.json").read_bytes()
        recorded = {"version": 1, "key": "0" * 64, "status": "completed", "run_id": origin or label, "stage": "export",
                    "outputs": {"required": ["export"], "artifacts": {"tree": {
                        "export": {"directory": True},
                        "export/export.json": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}}}
        if manifest:
            manifest(recorded)
        write_json(self.folder / label / "manifest.json", recorded)
        report = {"version": 1, "run_id": label, "status": "completed", "verification": "not_configured",
                  "stages": [{"id": "prepare", "role": "prepare", "status": "executed", "origin": label},
                             {"id": "export", "role": stage_role, "status": "executed", "origin": origin or label,
                              "artifacts": str(artifacts), "execute_s": 1.0}]}
        write_json(self.folder / label / "report.json", report)
        comparison = comparison_record(sha256(artifacts / "export/export.json"), sha256(self.reference))
        write_json(self.folder / label / "comparison.json", comparison)
        return self.folder / label / "report.json", artifacts / "export/export.json", self.folder / label / "comparison.json"

    def summarize(self, *runs):
        argv = ["fp16_summary", "--reference", str(self.reference)]
        for report, record, comparison in runs:
            argv += ["--report", str(report), "--export", str(record), "--comparison", str(comparison)]
        output = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(io.StringIO()):
            return fp16_summary.main(), output.getvalue()

    def test_the_summary_keeps_numbers_hashes_and_fixed_words_and_drops_paths_and_free_text(self):
        code, output = self.summarize(self.run_files("first"), self.run_files("second"))
        self.assertEqual(code, 0)
        result = json.loads(output)
        self.assertNotIn(SENTINEL, output)
        self.assertNotIn(str(self.folder), output)
        self.assertNotIn("rounded to float16", output)
        asset = result["assets"][0]
        self.assertEqual((asset["export"]["resources"]["swap"]["increase_bytes"], asset["export"]["disk"]["lowest_free_bytes"],
                          asset["export"]["bundle"]["files"], asset["comparison"]["conditions"][1]["max_abs_logit"]),
                         (0, 6, {"gemma4_31b_l6_fp16_main.aimodel/main.mlirb": {"bytes": 4, "sha256": "4" * 64}}, 0.5))
        self.assertEqual([entry["verifier"]["prefix_function"] for entry in asset["comparison"]["conditions"]],
                         [None, "prefill"])
        self.assertEqual(result["reference"]["inputs"]["context"]["rows"], 26)
        self.assertEqual(asset["export"]["asset"]["float16_minus_weights"], -12)
        self.assertEqual({key for entry in asset["comparison"]["conditions"] for key in entry["verifier"]}, set(native.PUBLIC))
        self.assertEqual(asset["pipeline"]["stages"][1]["role"], "export")

    def test_free_text_in_a_shareable_field_is_rejected(self):
        changes = [lambda record: record["bundle"]["files"].update({f"/{SENTINEL}/main.mlirb": {"bytes": 1, "sha256": "4" * 64}}),
                   lambda record: record["bundle"]["files"].update({"../main.mlirb": {"bytes": 1, "sha256": "4" * 64}}),
                   lambda record: record["asset"]["storage_types"].update({"free text": 1}),
                   lambda record: record.update(functions=["main prefill"]),
                   lambda record: record.update(passed="yes"),
                   lambda record: record.update(compression_config={"quantization": SENTINEL})]
        for index, change in enumerate(changes):
            record = export_record()
            change(record)
            with self.subTest(change=index):
                with self.assertRaises(PipelineError):
                    fp16_summary.exported(record)

    def test_each_export_must_come_once_with_the_report_that_wrote_it_and_its_comparison(self):
        first, second = self.run_files("first"), self.run_files("second")
        reused = self.run_files("reused", origin="earlier")
        other = self.run_files("other")
        write_json(other[2], comparison_record("0" * 64, sha256(self.reference)))
        variants = {"paired": ([first, second], 0), "the same export twice": ([first, first], 1),
                    "reports swapped": ([(second[0], first[1], first[2]), (first[0], second[1], second[2])], 1),
                    "a reused export step": ([reused], 1), "a comparison of another export": ([other], 1)}
        for label, (runs, expected) in variants.items():
            with self.subTest(case=label):
                code, output = self.summarize(*runs)
                self.assertEqual((code, output == ""), (expected, expected == 1))
        write_json(self.reference, {**context_record(), "embed_scale": 74.0})
        self.assertEqual(self.summarize(first)[0], 1)

    def test_the_export_record_must_be_the_output_that_the_manifest_of_its_stage_recorded(self):
        def remove(entry):
            del entry["outputs"]["artifacts"]["tree"]["export/export.json"]

        variants = {"another run": lambda entry: entry.update(run_id="other"),
                    "another stage": lambda entry: entry.update(stage="prepare"),
                    "an incomplete stage": lambda entry: entry.update(status="running"),
                    "another version": lambda entry: entry.update(version=2),
                    "no entry for export.json": remove,
                    "another size": lambda entry: entry["outputs"]["artifacts"]["tree"]["export/export.json"].update(bytes=1)}
        self.assertEqual(self.summarize(self.run_files("recorded"))[0], 0)
        for label, change in variants.items():
            with self.subTest(case=label):
                self.assertEqual(self.summarize(self.run_files(label.replace(" ", "-"), manifest=change)), (1, ""))
        run = self.run_files("no-manifest")
        (self.folder / "no-manifest/manifest.json").unlink()
        self.assertEqual(self.summarize(run), (1, ""))
        run = self.run_files("listed-manifest")
        (self.folder / "listed-manifest/manifest.json").write_text("[]")
        self.assertEqual(self.summarize(run), (1, ""))

    def test_an_export_record_replaced_after_the_run_is_refused_even_with_a_comparison_made_from_it(self):
        report, record, comparison = self.run_files("replaced")
        write_json(record, {**export_record(), "context": 16384})
        write_json(comparison, comparison_record(sha256(record), sha256(self.reference)))
        self.assertEqual(self.summarize((report, record, comparison)), (1, ""))

    def test_records_of_another_shape_stop_with_the_fixed_error_and_print_nothing(self):
        cases = {"a comparison that is a list": ("comparison", "[]"), "a report that is a string": ("report", '"text"'),
                 "a comparison whose export is a list": ("comparison", json.dumps({"export": [], "reference": {}})),
                 "a broken comparison": ("comparison", "{")}
        for label, (kind, content) in cases.items():
            report, record, comparison = self.run_files(label.replace(" ", "-"))
            {"report": report, "comparison": comparison}[kind].write_text(content)
            errors = io.StringIO()
            argv = ["fp16_summary", "--reference", str(self.reference), "--report", str(report), "--export", str(record),
                    "--comparison", str(comparison)]
            output = io.StringIO()
            with self.subTest(case=label), mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(output), \
                    contextlib.redirect_stderr(errors):
                self.assertEqual(fp16_summary.main(), 1)
                self.assertEqual(output.getvalue(), "")
                self.assertRegex(errors.getvalue(), r"^Cannot summarize local records \(\w+\)\.\n$")

    def test_a_condition_without_a_field_of_the_verifier_record_stops_with_the_fixed_error_and_prints_nothing(self):
        for key in ("cached", "load_s", "prefix_calls"):
            report, record, comparison = self.run_files(f"lacking-{key}")
            data = json.loads(comparison.read_text())
            del data["conditions"][1]["verifier"][key]
            comparison.write_text(json.dumps(data))
            with self.subTest(key=key):
                self.assertEqual(self.summarize((report, record, comparison)), (1, ""))


if __name__ == "__main__":
    unittest.main()
