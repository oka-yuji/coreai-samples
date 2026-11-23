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

try:
    import numpy as np
except ImportError:
    np = None

from Tools.Conversion.adapters.gemma4_31b import effect, export, fp16_summary, layers, native
from Tools.Conversion.adapters.gemma4_31b.prepare import TOKENIZER_FILES
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import PipelineError, read_json, write_json
from Tools.Conversion.tests.test_gemma4_31b_layers import TAPS, TEXT, write_contract, write_run

SENTINEL = "private-example-sentinel"
SHARD = "model-00001-of-00002.safetensors"
MODEL = {"id": "google/gemma-4-31B-it", "revision": "a" * 40, "contract_sha256": "b" * 64,
         "verified_before_loading": {SHARD: {"bytes": 3, "sha256": "c" * 64}, "config.json": {"bytes": 1, "sha256": "d" * 64}}}
KINDS = ["sliding_attention"] * 5 + ["full_attention"]
STATE = {"dtype": "float16", "shape": [1, 1, 4, None, 512]}


def context_record(contract_sha256="b" * 64):
    entry = {"positions": 26, "tokens_sha256": "e" * 64, "rows": list(range(26)), "calls": [[0, 26]], "seconds": 0.5,
             "logits": {"bytes": 10, "sha256": "f" * 64, "dtype": "float32", "shape": [26, 262144], "non_finite": 0}}
    return {"schema": layers.CONTEXT_SCHEMA, "implementation": "original", "layers": 6, "layer_types": KINDS,
            "model": {**MODEL, "contract_sha256": contract_sha256},
            "weights": "every checkpoint tensor rounded to float16", "embed_scale": 73.5, "chunk": 1024,
            "versions": {"torch": "2.15.0"}, "inputs": {"short": entry, "context": {**entry, "positions": 16896}},
            "timing": {"total_s": 60.0}, "memory": {"max_rss_bytes": 10, "lifetime_peak_footprint_bytes": None}}


def export_record(contract_sha256="b" * 64):
    function = {"inputs": {"input_ids": {"dtype": "int32", "shape": [1, None]}}, "outputs": {}, "states": {"keyCache": STATE}}
    return {"schema": export.SCHEMA, "passed": True, "checks": {"asset": True, "free_disk": True}, "problems": [],
            "model": {**MODEL, "contract_sha256": contract_sha256}, "layers": 6, "layer_types": KINDS, "functions": ["main"], "context": 16896, "precision": "float16",
            "compression_config": None, "upstream_revision": "1" * 40, "versions": {"coreai-core": "1.0.0b3"},
            "weights": {"tensors": 85, "elements": 9, "elements_by_dtype": {"torch.float16": 9}, "quantized_layouts": 0},
            "asset": {"functions": {"main": function}, "output_order": {"main": ["logits", "hidden_states", "tap_hidden_states"]},
                      "storage_types": {"Float16": 13, "Int32": 2}, "float16_minus_weights": -12},
            "tokenizer": {"tokenizer.json": "2" * 64},
            "metadata": {"name": "gemma4_31b_l6_fp16_main", "language": {"vocab_size": 262144, "max_context_length": 16896},
                         "source": {"hf_model_id": "google/gemma-4-31B-it", "revision": "a" * 40}, "compression": None,
                         "tap_layers": [1, 2, 3, 4, 5], "tap_convention": export.TAP_CONVENTION},
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


def run_arrays(taps=TAPS):
    generator = np.random.default_rng(0)
    widths = {"logits": TEXT["vocab_size"], "hidden_states": TEXT["hidden_size"],
              "tap_hidden_states": TEXT["hidden_size"] * len(taps)}
    return {name: {output: generator.normal(size=(length, width)).astype(np.float32) for output, width in widths.items()}
            for name, (length, _) in layers.INPUTS.items()}


class Records(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.folder))
        self.contract = write_contract(self.folder, dict.fromkeys(TOKENIZER_FILES, {"bytes": 1, "sha256": "0" * 64}))
        self.contract_sha256 = sha256(self.contract)
        self.reference = self.folder / "context.json"
        write_json(self.reference, context_record(self.contract_sha256))

    def options(self):
        return []

    def summarize(self, *runs, options=None):
        argv = ["fp16_summary", "--reference", str(self.reference), *(self.options() if options is None else options)]
        for report, record, comparison in runs:
            argv += ["--report", str(report), "--export", str(record), "--comparison", str(comparison)]
        output, errors = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.object(fp16_summary, "CONTRACT", self.contract), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = fp16_summary.main()
        self.errors = errors.getvalue()
        return code, output.getvalue()

    def run_files(self, label, record=None, origin=None, stage_role="export", manifest=None):
        artifacts = self.folder / label / "artifacts"
        write_json(artifacts / "export/export.json", record or export_record(self.contract_sha256))
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


@unittest.skipIf(np is None, "NumPy is required for the comparison of the model class")
class SummaryTests(Records):
    def setUp(self):
        super().setUp()
        self.runs = self.write_runs("runs")

    def write_runs(self, label, contract=None, taps=TAPS, change=None):
        arrays, folder = run_arrays(taps), self.folder / label
        write_run(folder / "original", "original", contract or self.contract, arrays, taps)
        if change:
            change(arrays)
        write_run(folder / "class", "class", contract or self.contract, arrays, taps)
        return folder / "original", folder / "class"

    def options(self):
        return ["--original", str(self.runs[0]), "--model-class", str(self.runs[1])]

    def test_an_fp16_summary_stops_without_either_run_or_with_the_effect_of_a_quantization(self):
        run, effect_file = self.run_files("first"), self.folder / "effect.json"
        write_json(effect_file, effect_record(sha256(self.reference)))
        original, model_class = self.options()[:2], self.options()[2:]
        cases = {"without the runs": [], "without the original run": model_class,
                 "without the model-class run": original, "with an effect": [*self.options(), "--effect", str(effect_file)],
                 "with an effect and without the runs": ["--effect", str(effect_file)]}
        for label, options in cases.items():
            with self.subTest(case=label):
                self.assertEqual(self.summarize(run, options=options), (1, ""))
                self.assertEqual(self.errors, "Cannot summarize local records (PipelineError).\n")
        self.assertEqual(self.summarize(run)[0], 0)

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
        self.assertEqual((asset["export"]["metadata"]["tap_layers"], asset["export"]["metadata"]["tap_convention"],
                          asset["export"]["asset"]["output_order"]),
                         ([1, 2, 3, 4, 5], export.TAP_CONVENTION, {"main": ["logits", "hidden_states", "tap_hidden_states"]}))
        compared = result["model_class"]
        self.assertEqual((compared["passed"], compared["problems"], compared["layers"], compared["tap_layers"],
                          compared["ratio_limit"]), (True, 0, 6, [1, 2, 3, 4, 5], 1e-3))
        self.assertEqual(compared["records_sha256"], [sha256(folder / "record.json") for folder in self.runs])
        self.assertEqual(set(compared["inputs"]), {"short", "long"})
        self.assertEqual(compared["inputs"]["long"]["hidden_states"], {"largest_ratio": 0.0, "passed": True})
        self.assertEqual(set(compared["inputs"]["long"]["tap_hidden_states"]), {"1", "2", "3", "4", "5"})
        self.assertNotIn("non_finite", compared["inputs"]["long"])

    def test_runs_of_another_contract_layer_count_or_tap_layers_or_with_a_replaced_array_stop_the_summary(self):
        run = self.run_files("first")
        self.assertEqual(self.summarize(run)[0], 0)
        other = self.folder / "other-contract"
        other.mkdir()
        another = write_contract(other, dict.fromkeys(TOKENIZER_FILES, {"bytes": 1, "sha256": "1" * 64}))

        def twelve(folder):
            record = read_json(folder / "record.json")
            record.update(layers=12, layer_types=TEXT["layer_types"])
            write_json(folder / "record.json", record)

        def replaced(folder):
            path = folder / "long.hidden_states.npy"
            np.save(path, np.load(path) + 1)

        cases = {"runs of another contract": (lambda: self.write_runs("another", contract=another), None),
                 "runs of four tap layers": (lambda: self.write_runs("four", taps=(1, 2, 3, 4)), None),
                 "a model-class run of twelve layers": (lambda: self.write_runs("twelve"), twelve),
                 "a model-class array replaced after its record": (lambda: self.write_runs("replaced"), replaced),
                 "the run of the original implementation twice": (lambda: (self.runs[0], self.runs[0]), None)}
        for label, (make, change) in cases.items():
            self.runs = make()
            if change:
                change(self.runs[1])
            with self.subTest(case=label):
                self.assertEqual(self.summarize(run), (1, ""))
        self.runs = self.write_runs("again")
        self.assertEqual(self.summarize(run)[0], 0)
        exports = self.run_files("other-exports", record=export_record("2" * 64))
        with self.subTest(case="exports of another contract"):
            self.assertEqual(self.summarize(exports), (1, ""))
        write_json(self.reference, context_record("2" * 64))
        with self.subTest(case="a reference of another contract"):
            self.assertEqual(self.summarize(self.run_files("other-reference")), (1, ""))

    def test_runs_that_fail_a_criterion_are_summarized_with_passed_false_like_a_failed_condition(self):
        def logit(arrays):
            arrays["short"]["logits"][3, 4] += 0.002

        self.runs = self.write_runs("failing", change=logit)
        code, output = self.summarize(self.run_files("first"))
        self.assertEqual(code, 0)
        compared = json.loads(output)["model_class"]
        self.assertEqual((compared["passed"], compared["problems"]), (False, 0))
        self.assertEqual({key for key, passed in compared["inputs"]["short"]["checks"].items() if not passed},
                         {"max_abs_logit"})

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
            argv = ["fp16_summary", "--reference", str(self.reference), "--original", str(self.runs[0]),
                    "--model-class", str(self.runs[1]),
                    "--report", str(report), "--export", str(record), "--comparison", str(comparison)]
            output = io.StringIO()
            with self.subTest(case=label), mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(output), \
                    mock.patch.object(fp16_summary, "CONTRACT", self.contract), contextlib.redirect_stderr(errors):
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

    def test_numbers_too_large_for_a_float_stop_with_the_fixed_error_and_print_nothing(self):
        record = export_record()
        record["weights"]["elements"] = 10**400
        cases = {"an export record": self.run_files("huge-export", record=record), "a comparison": self.run_files("huge-comparison")}
        data = json.loads(cases["a comparison"][2].read_text())
        data["conditions"][1]["verifier"]["main_calls"] = 10**400
        cases["a comparison"][2].write_text(json.dumps(data))
        for label, files in cases.items():
            with self.subTest(case=label):
                self.assertEqual(self.summarize(files), (1, ""))


def quantized_record(contract_sha256="b" * 64):
    record = export_record(contract_sha256)
    record.update(precision="8bit", compression_config=copy.deepcopy(export.QUANTIZATION))
    record["weights"].update(elements=9, elements_by_dtype={"torch.float16": 5, "torch.uint8": 4}, quantized_layouts=41)
    record["quantized"] = {"layers": [{"name": "model.layers.0.self_attn.q_proj.weight", "block": [1, 32]}],
                           "skipped": 0, "warnings_enabled": True, "file": {"bytes": 3, "sha256": "7" * 64},
                           "tensors": {"model.layers.0.self_attn.q_proj.weight.scale": "8" * 64}}
    record["asset"].update(storage_types={"UInt8": 132, "Float16": 9, "Int32": 2}, float16_minus_weights=4,
                           uint8_minus_weights=128, dequantizations=77)
    name = "gemma4_31b_l6_8bit_main"
    record["metadata"].update(name=name, compression="8bit")
    record["bundle"].update(name=name, files={f"{name}.aimodel/main.mlirb": {"bytes": 4, "sha256": "4" * 64}})
    return record


def restored_record(contract_sha256="b" * 64):
    return {**context_record(contract_sha256), "weights": layers.RESTORED,
            "quantized": {"export_sha256": "a" * 64, "file": {"bytes": 3, "sha256": "7" * 64}, "tensors": 123}}


def effect_record(reference_hash, **changes):
    entry = {"name": "tail", "input": "context", "tokens": 16896, "start": 16832, "chunk": 1, "rows": 64, "non_finite": 0,
             "reference_non_finite": 0, "max_abs_logit": 4.14, "top1_mismatches": {"near_tie": 2, "outside": 1}}
    return {"schema": effect.SCHEMA, "problems": [], "restored": {"sha256": reference_hash, "layers": 6},
            "rounded": {"sha256": "9" * 64, "layers": 6}, "criteria": {"near_tie": 0.01}, "conditions": [entry], **changes}


class QuantizedSummaryTests(Records):
    def setUp(self):
        super().setUp()
        write_json(self.reference, restored_record(self.contract_sha256))
        self.effect = self.folder / "effect.json"
        write_json(self.effect, effect_record(sha256(self.reference)))

    def options(self):
        return ["--effect", str(self.effect)]

    def run_files(self, label, record=None, origin=None, stage_role="export", manifest=None):
        return super().run_files(label, record or quantized_record(self.contract_sha256), origin, stage_role, manifest)

    def test_the_summary_keeps_numbers_hashes_and_fixed_words_and_drops_paths_and_free_text(self):
        code, output = self.summarize(self.run_files("first"))
        self.assertEqual(code, 0)
        result = json.loads(output)
        self.assertNotIn(SENTINEL, output)
        self.assertNotIn(str(self.folder), output)
        self.assertEqual(result["schema"], fp16_summary.QUANTIZED_SCHEMA)
        asset = result["assets"][0]["export"]
        self.assertEqual((asset["metadata"]["tap_layers"], asset["metadata"]["tap_convention"]),
                         ([1, 2, 3, 4, 5], export.TAP_CONVENTION))
        self.assertEqual(asset["quantized"], {"layers": 1, "skipped": 0, "warnings_enabled": True,
                                              "file": {"bytes": 3, "sha256": "7" * 64}, "tensors": 1})
        self.assertEqual((asset["asset"]["uint8_minus_weights"], asset["asset"]["float16_minus_weights"],
                          asset["asset"]["dequantizations"]), (128, 4, 77))
        self.assertEqual(result["reference"]["weights"], "8bit")
        self.assertEqual(result["reference"]["quantized"], {"export_sha256": "a" * 64,
                                                            "file": {"bytes": 3, "sha256": "7" * 64}, "tensors": 123})
        self.assertEqual(result["effect"]["rounded_sha256"], "9" * 64)
        self.assertEqual(result["effect"]["conditions"][0]["top1_mismatches"], {"near_tie": 2, "outside": 1})
        self.assertEqual(result["effect"]["conditions"][0]["max_abs_logit"], 4.14)

    def test_free_text_in_a_shareable_field_is_rejected(self):
        changes = [lambda record: record["quantized"]["file"].update(sha256=SENTINEL),
                   lambda record: record["asset"].update(uint8_minus_weights=SENTINEL),
                   lambda record: record.update(compression_config={"quantization": SENTINEL}),
                   lambda record: record["quantized"].update(warnings_enabled="yes"),
                   lambda record: record["quantized"].update(layers=SENTINEL),
                   lambda record: record["quantized"].update(tensors=[SENTINEL]),
                   lambda record: record["compression_config"]["module_type_configs"]["torch.nn.modules.linear.Linear"][
                       "op_state_spec"]["weight"]["granularity"].update(axis=True)]
        for index, change in enumerate(changes):
            record = quantized_record(self.contract_sha256)
            change(record)
            with self.subTest(change=index), self.assertRaises(PipelineError):
                fp16_summary.exported(record)

    def test_an_8bit_summary_needs_the_effect_of_its_reference_and_an_fp16_summary_refuses_one(self):
        self.assertEqual(self.summarize(self.run_files("no-effect"), options=[]), (1, ""))
        self.assertEqual(self.errors, "Cannot summarize local records (PipelineError).\n")
        write_json(self.effect, effect_record("0" * 64))
        self.assertEqual(self.summarize(self.run_files("other-reference")), (1, ""))
        write_json(self.effect, effect_record(sha256(self.reference), problems=["two rounded references"]))
        self.assertEqual(self.summarize(self.run_files("with-problems")), (1, ""))
        write_json(self.effect, effect_record(sha256(self.reference), schema="other"))
        self.assertEqual(self.summarize(self.run_files("other-schema")), (1, ""))
        write_json(self.effect, effect_record(sha256(self.reference), restored=[sha256(self.reference)]))
        self.assertEqual(self.summarize(self.run_files("listed-reference")), (1, ""))
        self.effect.write_text("[]")
        self.assertEqual(self.summarize(self.run_files("listed-effect")), (1, ""))
        write_json(self.effect, effect_record(sha256(self.reference)))
        fp16 = Records.run_files(self, "fp16", export_record(self.contract_sha256))
        self.assertEqual(self.summarize(self.run_files("mixed"), fp16), (1, ""))
        self.assertEqual(self.summarize(fp16), (1, ""))
        self.assertEqual(self.summarize(self.run_files("paired"))[0], 0)

    def test_an_8bit_summary_needs_no_run_of_the_model_class_and_stops_when_one_is_given(self):
        run, missing = self.run_files("first"), str(self.folder / "missing-run")
        cases = {"with the original run": [*self.options(), "--original", missing],
                 "with the model-class run": [*self.options(), "--model-class", missing],
                 "with both runs": [*self.options(), "--original", missing, "--model-class", missing],
                 "with both runs and without the effect": ["--original", missing, "--model-class", missing]}
        for label, options in cases.items():
            with self.subTest(case=label):
                self.assertEqual(self.summarize(run, options=options), (1, ""))
                self.assertEqual(self.errors, "Cannot summarize local records (PipelineError).\n")
        code, output = self.summarize(run)
        self.assertEqual(code, 0)
        self.assertNotIn("model_class", json.loads(output))


if __name__ == "__main__":
    unittest.main()
