import contextlib
import copy
import hashlib
import io
import json
import math
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
    import torch
except ImportError:
    np = torch = None

from Tools.Conversion.adapters.gemma4_31b import effect, export, fp16_summary, layers, native
from Tools.Conversion.adapters.gemma4_31b.prepare import TOKENIZER_FILES, load_contract
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import PipelineError, canonical, digest, read_json, write_json
from Tools.Conversion.tests.test_gemma4_31b_layers import TAPS, TEXT, complete_export, write_contract, write_run

ROOT = Path(__file__).resolve().parents[3]
CHILD = ("import sys\nfrom pathlib import Path\nfrom unittest import mock\n"
         "from Tools.Conversion.adapters.gemma4_31b import fp16_summary\n"
         "contract, sys.argv = Path(sys.argv[1]), ['fp16_summary', *sys.argv[2:]]\n"
         "with mock.patch.object(fp16_summary, 'CONTRACT', contract):\n    sys.exit(fp16_summary.main())\n")
SENTINEL = "private-example-sentinel"
SHARD = "model-00001-of-00002.safetensors"
MODEL = {"id": "google/gemma-4-31B-it", "revision": "a" * 40, "contract_sha256": "b" * 64,
         "verified_before_loading": {SHARD: {"bytes": 3, "sha256": "c" * 64}, "config.json": {"bytes": 1, "sha256": "d" * 64}}}
KINDS = ["sliding_attention"] * 5 + ["full_attention"]
STATE = {"dtype": "float16", "shape": [1, 1, 4, None, 512]}


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


def reference_arrays(vocabulary, seed=0):
    generator = np.random.default_rng(seed)
    arrays = {}
    for name in ("short", "context"):
        arrays[name] = generator.normal(size=(len(layers.kept_rows(name)), vocabulary)).astype(np.float32)
        arrays[name][:, 0] += 20
    return arrays


def rounded_arrays(vocabulary):
    arrays = reference_arrays(vocabulary)
    rows = layers.kept_rows("context")
    arrays["short"][3, 7] += 1.5
    arrays["context"][rows.index(1100), 1] = arrays["context"][rows.index(1100), 0] + 1.0
    arrays["context"][rows.index(16890), :2] = (30.0, 30.005)
    return arrays


def reference_record(folder, contract, weights, quantized=None, arrays=None):
    folder.mkdir(parents=True, exist_ok=True)
    pinned = load_contract(contract)
    config, inputs = pinned["config"], {}
    arrays = reference_arrays(config["vocab_size"]) if arrays is None else arrays
    for name in ("short", "context"):
        ids, rows = layers.tokens(name, config), layers.kept_rows(name)
        np.save(folder / f"{name}.npy", arrays[name])
        inputs[name] = {"positions": len(ids), "tokens_sha256": digest(ids), "rows": rows, "seconds": 0.5,
                        "calls": [[start, min(start + layers.REFERENCE_CHUNK, len(ids))]
                                  for start in range(0, len(ids), layers.REFERENCE_CHUNK)],
                        "logits": {**layers.file_record(folder / f"{name}.npy"), "dtype": "float32",
                                   "shape": [len(rows), config["vocab_size"]],
                                   "non_finite": int(arrays[name].size - np.isfinite(arrays[name]).sum())}}
    write_json(folder / "record.json", {
        "schema": layers.CONTEXT_SCHEMA, "implementation": "original", "layers": 6,
        "layer_types": layers.check_layers(config, 6), "model": native.pinned_model(pinned, contract), "weights": weights,
        **({"quantized": quantized} if quantized else {}), "embed_scale": layers.rounded_scale(config["hidden_size"]),
        "chunk": layers.REFERENCE_CHUNK, "versions": dict.fromkeys(layers.PACKAGES, "2.15.0"), "inputs": inputs,
        "timing": {"input_hash_s": 1.0, "load_s": 2.0, "total_s": 60.0},
        "memory": {"max_rss_bytes": 10, "lifetime_peak_footprint_bytes": None}})
    return folder / "record.json"


def quantized_export(folder, contract):
    folder.mkdir(parents=True)
    tensors = folder / layers.QUANTIZED
    tensors.write_bytes(b"quantized tensors")
    names = sorted(export.quantized_names(layers.check_layers(load_contract(contract)["config"], 6)))
    return complete_export(folder, contract, "8bit", quantized={
        "layers": [{"name": name, "class": "Dequantize", "shape": [1, 32], **export.LAYOUT} for name in names],
        "skipped": 0, "warnings_enabled": True, "file": {"bytes": tensors.stat().st_size, "sha256": sha256(tensors)},
        "tensors": {f"{name}.{part}": "1" * 64 for name in names for part in layers.PARTS}})


def logits_file(path, arrays, source, start, length, noise=0.001, seed=7):
    index = {row: position for position, row in enumerate(layers.kept_rows(source))}
    reference = arrays[source][[index[position] for position in range(start, length)]]
    generator = np.random.default_rng(seed)
    logits = (reference + generator.normal(scale=noise, size=reference.shape)).astype(np.float16)
    path.write_bytes(logits.astype("<f2").tobytes())


def measured_entry(entry, output, config, arrays):
    numbers = native.measure_condition(entry, config["vocab_size"], output.read_bytes(), arrays)
    entry.update(numbers, output={"bytes": output.stat().st_size, "sha256": sha256(output)})
    entry["verifier"]["non_finite_count"] = numbers["non_finite"]
    entry["passed"] = native.condition_passed(entry)
    return entry


def comparison_record(folder, export_hash, reference_hash, exported, config, arrays):
    conditions = []
    for name, source, length, start, chunk in layers.CONDITIONS:
        output, ids = folder / f"{name}.f16", layers.tokens(source, config)[:length]
        logits_file(output, arrays, source, start, length)
        conditions.append(measured_entry({
            "name": name, "input": source, "tokens": length, "start": start, "chunk": chunk, "rows": length - start,
            "problems": [], "seconds": 4.0,
            "verifier": {"record": "logits", "token_count": length, "start": start, "chunk": chunk,
                         "tokens_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
                         "shape": [length - start, config["vocab_size"]], "dtype": "float16", "byte_order": "little",
                         "main_calls": -(-(length - start) // chunk), "main_hash": exported["bundle"]["main_hash"],
                         **native.prefix_record(start, 4096, "prefill" in exported["functions"]), "cached": True,
                         "non_finite_count": 0, "load_s": 1.0, "prefix_s": 1.0, "scoring_s": 1.0, "specialize_s": 0.0}},
            output, config, arrays))
    return {"schema": native.SCHEMA, "passed": all(entry["passed"] for entry in conditions), "problems": [],
            "criteria": {"max_abs_logit": 2.0, "near_tie": 0.01}, "known_call": {"ends": [97, 112], "shortest": 8},
            "reference": {"sha256": reference_hash, "layers": 6},
            "export": {"sha256": export_hash, "layers": 6, "functions": exported["functions"],
                       "main_hash": exported["bundle"]["main_hash"]},
            "verifier": {"bytes": 1, "sha256": "6" * 64}, "conditions": conditions}


def rewrite(path, change):
    record = read_json(path)
    change(record)
    write_json(path, record)


def run_arrays(taps=TAPS):
    generator = np.random.default_rng(0)
    widths = {"logits": TEXT["vocab_size"], "hidden_states": TEXT["hidden_size"],
              "tap_hidden_states": TEXT["hidden_size"] * len(taps)}
    return {name: {output: generator.normal(size=(length, width)).astype(np.float32) for output, width in widths.items()}
            for name, (length, _) in layers.INPUTS.items()}


@unittest.skipIf(np is None or torch is None, "NumPy and PyTorch are required to write the reference records")
class Records(unittest.TestCase):
    PRECISION = "fp16"

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        (self.folder / "snapshot").mkdir()
        for index, item in enumerate(TOKENIZER_FILES):
            (self.folder / "snapshot" / item).write_text("abc"[index])
        files = {item: {"bytes": 1, "sha256": sha256(self.folder / "snapshot" / item)} for item in TOKENIZER_FILES}
        self.contract = write_contract(self.folder, files)
        self.contract_sha256 = sha256(self.contract)
        self.reference = reference_record(self.folder / "reference", self.contract, layers.WEIGHTS)

    def options(self):
        return []

    def arguments(self, *runs, options=None):
        argv = ["--reference", str(self.reference), *(self.options() if options is None else options)]
        for report, record, comparison in runs:
            argv += ["--report", str(report), "--export", str(record), "--comparison", str(comparison)]
        return argv

    def summarize_within(self, *runs, options=None, seconds=20):
        try:
            done = subprocess.run([sys.executable, "-c", CHILD, str(self.contract), *self.arguments(*runs, options=options)],
                                  cwd=ROOT, capture_output=True, text=True, timeout=seconds,
                                  env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        except subprocess.TimeoutExpired:
            self.fail(f"the summary did not finish within {seconds} s")
        return done.returncode, done.stdout

    def summarize(self, *runs, options=None):
        argv = ["fp16_summary", *self.arguments(*runs, options=options)]
        output, errors = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.object(fp16_summary, "CONTRACT", self.contract), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = fp16_summary.main()
        self.errors = errors.getvalue()
        return code, output.getvalue()

    def run_files(self, label, change=None, origin=None, stage_role="export", manifest=None, precision=None):
        artifacts = self.folder / label / "artifacts"
        if (precision or self.PRECISION) == "8bit":
            quantized_export(artifacts / "export", self.contract)
        else:
            complete_export(artifacts / "export", self.contract)
        if change:
            rewrite(artifacts / "export/export.json", change)
        data, written = (artifacts / "export/export.json").read_bytes(), read_json(artifacts / "export/export.json")
        recorded = {"version": 1, "key": "0" * 64, "status": "completed", "run_id": origin or label, "stage": "export",
                    "outputs": {"required": ["export"], "artifacts": {"tree": {
                        "export": {"directory": True}, "export/bundle": {"directory": True},
                        **{f"export/bundle/{name}": dict(entry) for name, entry in written["bundle"]["files"].items()},
                        **({"export/quantized.safetensors": dict(written["quantized"]["file"])}
                           if "quantized" in written else {}),
                        "export/export.json": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}},
                    "execute_s": 10.0}
        if manifest:
            manifest(recorded)
        write_json(self.folder / label / "manifest.json", recorded)
        report = {"version": 1, "run_id": label, "status": "completed", "verification": "not_configured",
                  "stages": [{"id": "prepare", "role": "prepare", "status": "executed", "origin": label,
                              "prepare_s": 0.5, "execute_s": 2.0, "total_s": 3.0},
                             {"id": "export", "role": stage_role, "status": "executed", "origin": origin or label,
                              "artifacts": str(artifacts), "prepare_s": 0.5, "execute_s": 10.0, "total_s": 11.0}],
                  "elapsed_s": 15.0}
        write_json(self.folder / label / "report.json", report)
        comparison = comparison_record(self.folder / label, sha256(artifacts / "export/export.json"), sha256(self.reference),
                                       written, load_contract(self.contract)["config"], self.arrays())
        write_json(self.folder / label / "comparison.json", comparison)
        return self.folder / label / "report.json", artifacts / "export/export.json", self.folder / label / "comparison.json"

    def arrays(self):
        return {name: np.load(self.reference.parent / f"{name}.npy") for name in ("short", "context")}

    def referenced(self, change, run=None):
        rewrite(self.reference, change)
        if run:
            rewrite(run[2], lambda record: record["reference"].update(sha256=sha256(self.reference)))


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

    def test_an_fp16_summary_stops_without_either_run_or_with_a_rounded_reference(self):
        run, rounded = self.run_files("first"), reference_record(self.folder / "rounded", self.contract, layers.WEIGHTS).parent
        original, model_class = self.options()[:2], self.options()[2:]
        cases = {"without the runs": [], "without the original run": model_class,
                 "without the model-class run": original,
                 "with a rounded reference": [*self.options(), "--rounded", str(rounded)],
                 "with a rounded reference and without the runs": ["--rounded", str(rounded)]}
        for label, options in cases.items():
            with self.subTest(case=label):
                self.assertEqual(self.summarize(run, options=options), (1, ""))
                self.assertEqual(self.errors, "Cannot summarize local records (PipelineError).\n")
        self.assertEqual(self.summarize(run)[0], 0)

    def test_the_summary_keeps_numbers_hashes_and_fixed_words_and_drops_paths_and_free_text(self):
        code, output = self.summarize(self.run_files("first"), self.run_files("second"))
        self.assertEqual(code, 0, self.errors)
        result = json.loads(output)
        self.assertNotIn(SENTINEL, output)
        self.assertNotIn(str(self.folder), output)
        self.assertNotIn("rounded to float16", output)
        asset, written = result["assets"][0], read_json(self.folder / "first/artifacts/export/export.json")
        recorded = read_json(self.folder / "first/comparison.json")["conditions"]
        self.assertEqual((asset["export"]["resources"]["swap"]["increase_bytes"], asset["export"]["disk"]["lowest_free_bytes"],
                          asset["export"]["bundle"]["files"], asset["comparison"]["conditions"][1]["max_abs_logit"]),
                         (0, export.MARGIN_BYTES, written["bundle"]["files"], recorded[1]["max_abs_logit"]))
        self.assertGreater(recorded[1]["max_abs_logit"], 0)
        self.assertEqual(asset["comparison"]["conditions"][5]["known"]["rows"], 16)
        self.assertEqual([entry["verifier"]["prefix_function"] for entry in asset["comparison"]["conditions"]],
                         [None if start == 0 else "prefill" for _, _, _, start, _ in layers.CONDITIONS])
        self.assertEqual(result["reference"]["inputs"]["context"]["rows"], len(layers.kept_rows("context")))
        self.assertEqual(asset["export"]["asset"]["float16_minus_weights"], 4)
        self.assertEqual({key for entry in asset["comparison"]["conditions"] for key in entry["verifier"]}, set(native.PUBLIC))
        self.assertEqual(asset["pipeline"]["stages"][1]["role"], "export")
        self.assertEqual((asset["export"]["metadata"]["tap_layers"], asset["export"]["metadata"]["tap_convention"],
                          asset["export"]["asset"]["output_order"]),
                         ([1, 2, 3, 4, 5], export.TAP_CONVENTION, {"main": ["logits", "hidden_states", "tap_hidden_states"],
                                                                   "prefill": ["tap_hidden_states"]}))
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

        def restored():
            runs = self.write_runs("restored")
            rewrite(runs[0] / "record.json", lambda record: record.update(weights=layers.RESTORED))
            rewrite(runs[1] / "record.json", lambda record: record.update(weights=layers.WEIGHTS))
            return runs

        cases = {"runs of another contract": (lambda: self.write_runs("another", contract=another), None),
                 "an original run of the restored weights": (restored, None),
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
        exports = self.run_files("other-exports", change=lambda record: record["model"].update(contract_sha256="2" * 64))
        with self.subTest(case="exports of another contract"):
            self.assertEqual(self.summarize(exports), (1, ""))
        run = self.run_files("other-reference")
        self.referenced(lambda record: record["model"].update(contract_sha256="2" * 64), run)
        with self.subTest(case="a reference of another contract"):
            self.assertEqual(self.summarize(run), (1, ""))

    def test_a_number_of_an_fp16_condition_that_its_logits_file_and_the_reference_do_not_give_stops_the_summary(self):
        run = self.run_files("first")
        self.assertEqual(self.summarize(run)[0], 0)
        rewrite(run[2], lambda record: record["conditions"][1].update(
            max_abs_logit=math.nextafter(record["conditions"][1]["max_abs_logit"], 9)))
        self.assertEqual(self.summarize(run), (1, ""))

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

    def test_an_fp16_summary_refuses_a_reference_of_other_weights_than_the_rounded_ones(self):
        run = self.run_files("first")
        self.assertEqual(self.summarize(run)[0], 0)
        self.referenced(lambda record: record.update(weights="every checkpoint tensor in bfloat16"), run)
        self.assertEqual(self.summarize(run), (1, ""))

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
        rewrite(other[2], lambda record: record["export"].update(sha256="0" * 64))
        variants = {"paired": ([first, second], 0), "the same export twice": ([first, first], 1),
                    "reports swapped": ([(second[0], first[1], first[2]), (first[0], second[1], second[2])], 1),
                    "a reused export step": ([reused], 1), "a comparison of another export": ([other], 1)}
        for label, (runs, expected) in variants.items():
            with self.subTest(case=label):
                code, output = self.summarize(*runs)
                self.assertEqual((code, output == ""), (expected, expected == 1))
        self.referenced(lambda record: record.update(embed_scale=74.0), first)
        self.assertEqual(self.summarize(first)[0], 1)

    def test_the_time_of_an_export_must_fit_in_the_time_of_the_stage_that_ran_it(self):
        report = {"stages": [{"prepare_s": 0.5, "execute_s": 2.0, "total_s": 3.0},
                             {"prepare_s": 0.5, "execute_s": 10.0, "total_s": 11.0}], "elapsed_s": 15.0}
        stage, manifest = report["stages"][1], {"execute_s": 10.0}
        cases = {"an export that took as long as its stage": (10.0, True),
                 "an export that took the next number above the time of its stage": (math.nextafter(10.0, 11.0), False),
                 "an export that recorded no time": (None, True), "an export that recorded a time of another type": ("x", True)}
        for label, (spent, expected) in cases.items():
            with self.subTest(case=label):
                self.assertEqual(fp16_summary.timed(report, manifest, stage, {"timing": {"total_s": spent}}), expected)

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
        rewrite(record, lambda written: written.update(context=16384))
        rewrite(comparison, lambda written: written["export"].update(sha256=sha256(record)))
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
        cases = {"an export record": self.run_files("huge-export",
                                                    change=lambda record: record["weights"].update(elements=10**400)),
                 "a comparison": self.run_files("huge-comparison")}
        data = json.loads(cases["a comparison"][2].read_text())
        data["conditions"][1]["verifier"]["main_calls"] = 10**400
        cases["a comparison"][2].write_text(json.dumps(data))
        for label, files in cases.items():
            with self.subTest(case=label):
                self.assertEqual(self.summarize(files), (1, ""))


class QuantizedSummaryTests(Records):
    PRECISION = "8bit"

    def setUp(self):
        super().setUp()
        default = quantized_export(self.folder / "default-export", self.contract)
        self.export_sha256, written = sha256(default / "export.json"), read_json(default / "export.json")
        self.vocabulary = load_contract(self.contract)["config"]["vocab_size"]
        self.quantized = {"export_sha256": self.export_sha256, "file": written["quantized"]["file"],
                          "tensors": len(written["quantized"]["tensors"])}
        self.reference = reference_record(self.folder / "reference", self.contract, layers.RESTORED,
                                          quantized=self.quantized)
        self.rounded = reference_record(self.folder / "rounded", self.contract, layers.WEIGHTS,
                                        arrays=rounded_arrays(self.vocabulary)).parent

    def options(self):
        return ["--rounded", str(self.rounded)]

    def given_effect(self):
        computed = effect.effect(self.reference.parent, self.rounded, self.contract)
        self.assertEqual(computed["problems"], [])

        def given(restored, rounded, contract):
            return {**computed, "restored": {**computed["restored"], "sha256": sha256(restored / "record.json")}}

        return lambda: mock.patch.object(fp16_summary, "quantization_effect", side_effect=given)

    def test_the_summary_keeps_numbers_hashes_and_fixed_words_and_drops_paths_and_free_text(self):
        code, output = self.summarize(self.run_files("first"))
        self.assertEqual(code, 0, self.errors)
        result = json.loads(output)
        self.assertNotIn(SENTINEL, output)
        self.assertNotIn(str(self.folder), output)
        self.assertEqual(result["schema"], fp16_summary.QUANTIZED_SCHEMA)
        asset, written = result["assets"][0]["export"], read_json(self.folder / "first/artifacts/export/export.json")
        self.assertEqual((asset["metadata"]["tap_layers"], asset["metadata"]["tap_convention"]),
                         ([1, 2, 3, 4, 5], export.TAP_CONVENTION))
        self.assertEqual(asset["quantized"], {"layers": 41, "skipped": 0, "warnings_enabled": True,
                                              "file": written["quantized"]["file"], "tensors": 123})
        self.assertEqual((asset["asset"]["uint8_minus_weights"], asset["asset"]["float16_minus_weights"],
                          asset["asset"]["dequantizations"]), (128, 4, written["asset"]["dequantizations"]))
        self.assertEqual(result["reference"]["weights"], "8bit")
        self.assertEqual(result["reference"]["quantized"], {"export_sha256": self.export_sha256,
                                                            "file": written["quantized"]["file"], "tensors": 123})
        computed = effect.effect(self.reference.parent, self.rounded, self.contract)
        self.assertEqual(computed["problems"], [])
        self.assertEqual(result["effect"]["rounded_sha256"], sha256(self.rounded / "record.json"))
        shown = ("name", "input", "tokens", "start", "chunk", "rows", "non_finite", "reference_non_finite", "max_abs_logit",
                 "top1_mismatches")
        self.assertEqual(result["effect"]["conditions"], [{key: entry[key] for key in shown} for entry in computed["conditions"]])
        conditions = {entry["name"]: entry for entry in result["effect"]["conditions"]}
        self.assertAlmostEqual(conditions["short-c1"]["max_abs_logit"], 1.5, places=5)
        self.assertEqual({key: value for key, value in conditions["short-c1"].items() if key not in ("name", "chunk")},
                         {key: value for key, value in conditions["short-c26"].items() if key not in ("name", "chunk")})
        self.assertEqual((conditions["window-c1"]["top1_mismatches"], conditions["tail"]["top1_mismatches"],
                          conditions["capacity-2048"]["top1_mismatches"]),
                         ({"near_tie": 0, "outside": 1}, {"near_tie": 1, "outside": 0}, {"near_tie": 0, "outside": 0}))

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

    def rounded_variant(self, label, change=None, weights=layers.WEIGHTS):
        folder = reference_record(self.folder / label, self.contract, weights, arrays=rounded_arrays(self.vocabulary)).parent
        if change:
            rewrite(folder / "record.json", change)
        return folder

    def test_an_8bit_summary_needs_a_rounded_reference_of_its_layers_and_an_fp16_summary_refuses_one(self):
        self.assertEqual(self.summarize(self.run_files("no-rounded"), options=[]), (1, ""))
        self.assertEqual(self.errors, "Cannot summarize local records (PipelineError).\n")

        def twelve(record):
            record.update(layers=12, layer_types=layers.check_layers(load_contract(self.contract)["config"], 12))

        def replaced(folder):
            np.save(folder / "short.npy", np.zeros((len(layers.kept_rows("short")), self.vocabulary), dtype=np.float32))
            return folder

        cases = {"the restored reference given as the rounded one": self.reference.parent,
                 "a rounded reference of the weights of the restored one": self.rounded_variant(
                     "restored-weights", weights=layers.RESTORED),
                 "a rounded reference of other weights": self.rounded_variant(
                     "other-weights", weights="every checkpoint tensor in bfloat16"),
                 "a rounded reference of twelve layers": self.rounded_variant("twelve-layers", twelve),
                 "a rounded reference of another contract": self.rounded_variant(
                     "other-contract", lambda record: record["model"].update(contract_sha256="2" * 64)),
                 "a rounded reference of other rows": self.rounded_variant(
                     "other-rows", lambda record: record["inputs"]["context"].update(rows=[0])),
                 "a rounded reference of another input": self.rounded_variant(
                     "other-input", lambda record: record["inputs"]["short"].update(tokens_sha256="0" * 64)),
                 "a rounded reference whose logits were replaced after its record": replaced(
                     self.rounded_variant("replaced-logits")),
                 "a rounded reference that is a list": self.rounded_variant("listed", None),
                 "a folder without a rounded reference": self.folder / "missing"}
        (cases["a rounded reference that is a list"] / "record.json").write_text("[]")
        run = self.run_files("with-rounded")
        for label, folder in cases.items():
            with self.subTest(case=label):
                self.assertEqual(self.summarize(run, options=["--rounded", str(folder)]), (1, ""))
                self.assertEqual(self.errors[:len("Cannot summarize local records (")], "Cannot summarize local records (")
        self.assertEqual(self.summarize(run)[0], 0)
        fp16 = self.run_files("fp16", precision="fp16")
        self.assertEqual(self.summarize(self.run_files("mixed"), fp16), (1, ""))
        self.assertEqual(self.summarize(fp16), (1, ""))
        self.assertEqual(self.summarize(self.run_files("paired"))[0], 0)

    def test_an_8bit_summary_refuses_records_that_do_not_correspond_to_each_other(self):
        reference, given = read_json(self.reference), self.given_effect()

        def twelve(run):
            config = load_contract(self.contract)["config"]
            rewrite(self.reference, lambda record: record.update(layers=12, layer_types=layers.check_layers(config, 12)))
            rewrite(run[2], lambda record: record["reference"].update(sha256=sha256(self.reference), layers=12))

        def referenced(change):
            return lambda run: self.referenced(change, run)

        def compared(change):
            return lambda run: rewrite(run[2], change)

        def manifested(change):
            return lambda run: rewrite(run[0].parent / "manifest.json",
                                       lambda record: change(record["outputs"]["artifacts"]["tree"]))

        name = read_json(self.folder / "default-export/export.json")["bundle"]["name"]
        cases = {"a restored reference of twelve layers": twelve,
                 "a restored reference of another quantized file": referenced(
                     lambda record: record["quantized"]["file"].update(sha256="0" * 64)),
                 "a reference restored for another export record": referenced(
                     lambda record: record["quantized"].update(export_sha256="0" * 64)),
                 "a reference restored from another number of tensors": referenced(
                     lambda record: record["quantized"].update(tensors=2)),
                 "a reference of the rounded weights": referenced(lambda record: record.update(weights=layers.WEIGHTS)),
                 "a comparison of a reference of twelve layers": compared(
                     lambda record: record["reference"].update(layers=12)),
                 "a comparison of an export of twelve layers": compared(lambda record: record["export"].update(layers=12)),
                 "a comparison of other functions": compared(
                     lambda record: record["export"].update(functions=["main"])),
                 "a comparison of another asset": compared(lambda record: record["export"].update(main_hash="0" * 64)),
                 "a manifest of another bundle file": manifested(
                     lambda tree: tree[f"export/bundle/{name}.aimodel/main.mlirb"].update(bytes=5)),
                 "a manifest of one more bundle file": manifested(
                     lambda tree: tree.update({"export/bundle/notes.txt": {"bytes": 1, "sha256": "0" * 64}})),
                 "a manifest of another quantized file": manifested(
                     lambda tree: tree["export/quantized.safetensors"].update(sha256="0" * 64)),
                 "a reference of another contract than the repository": referenced(
                     lambda record: record["model"].update(contract_sha256="2" * 64))}
        self.assertEqual(self.summarize(self.run_files("consistent"))[0], 0)
        for index, (label, change) in enumerate(cases.items()):
            write_json(self.reference, reference)
            run = self.run_files(f"case-{index}")
            change(run)
            with self.subTest(case=label):
                self.assertEqual(self.summarize(run), (1, ""))
            with self.subTest(case=label + ", with the effect of the references that were not changed"), given():
                self.assertEqual(self.summarize(run), (1, ""))
        write_json(self.reference, reference)
        run = self.run_files("other-contract", change=lambda record: record["model"].update(contract_sha256="2" * 64))
        self.referenced(lambda record: record["quantized"].update(export_sha256=sha256(run[1])), run)
        with self.subTest(case="an export of another contract than the repository"):
            self.assertEqual(self.summarize(run), (1, ""))

    def test_an_8bit_summary_takes_fixed_values_from_the_tools_and_computes_the_verdicts_again(self):
        reference, given = read_json(self.reference), self.given_effect()

        def conditions(change):
            def apply(record):
                change(record["conditions"])
            return apply

        def swapped(entries):
            entries[0], entries[1] = entries[1], entries[0]

        def unpassed(record):
            record["conditions"][0]["passed"] = False
            record["passed"] = False

        def failed(record):
            record.update(problems=["the verifier is unreadable (OSError)"], conditions=[], verifier=None)

        def dropped(run):
            rewrite(run[2], lambda record: record["conditions"].pop())

        def output(run):
            (run[2].parent / "window-c1.f16").write_bytes(b"other")

        def sized(run):
            (run[2].parent / "window-c1.f16").write_bytes(b"ab")
            rewrite(run[2], conditions(lambda entries: entries[2].update(
                output={"bytes": 2, "sha256": hashlib.sha256(b"ab").hexdigest()})))

        def trailing(run):
            path = self.reference.parent / "short.npy"
            path.write_bytes(path.read_bytes() + b"end")
            self.referenced(lambda record: record["inputs"]["short"]["logits"].update(layers.file_record(path)), run)

        def report(change):
            return lambda run: rewrite(run[0], change)

        def unpassed_export(record):
            record["checks"]["free_disk"] = False
            record["passed"] = False

        exports = {"an export whose checks and passed do not follow from its record": unpassed_export,
                   "an export of one more disk record": lambda record: record["disk"].update(free_text=1),
                   "an export of one more swap record": lambda record: record["resources"]["swap"].update(free_text=1),
                   "an export of other versions": lambda record: record["versions"].update(free_text="1")}

        cases = {"a condition left out of the comparison": dropped,
                 "two conditions in another order": lambda run: rewrite(run[2], conditions(swapped)),
                 "a condition of another chunk": lambda run: rewrite(run[2], conditions(
                     lambda entries: entries[-1].update(chunk=2))),
                 "other criteria": lambda run: rewrite(run[2], lambda record: record["criteria"].update(max_abs_logit=3.0)),
                 "another known call": lambda run: rewrite(run[2], lambda record: record["known_call"].update(ends=[96, 112])),
                 "a condition that did not pass with checks that did": lambda run: rewrite(run[2], unpassed),
                 "a comparison that did not pass with conditions that did": lambda run: rewrite(
                     run[2], lambda record: record.update(passed=False)),
                 "a comparison with problems recorded as passed": lambda run: rewrite(run[2], failed),
                 "a logits file other than its record": output,
                 "a reference of another chunk": lambda run: self.referenced(lambda record: record.update(chunk=512), run),
                 "a reference of another embedding scale": lambda run: self.referenced(
                     lambda record: record.update(embed_scale=1.0), run),
                 "a reference of other calls": lambda run: self.referenced(
                     lambda record: record["inputs"]["short"].update(calls=[[0, 13], [13, 26]]), run),
                 "a reference whose logits are recorded as float16": lambda run: self.referenced(
                     lambda record: record["inputs"]["short"]["logits"].update(dtype="float16"), run),
                 "a reference logits file other than its record": lambda run: np.save(
                     self.reference.parent / "short.npy", np.ones((len(layers.kept_rows("short")), TEXT["vocab_size"]),
                                                                  dtype=np.float32)),
                 "a report of a run that did not complete": report(lambda record: record.update(status="failed")),
                 "a report of a verified run": report(lambda record: record.update(verification="passed")),
                 "a report of other stages": report(lambda record: record["stages"][0].update(role="reference")),
                 "a report whose first stage was reused": report(lambda record: record["stages"][0].update(status="reused")),
                 "a verifier record of other calls": lambda run: rewrite(run[2], conditions(
                     lambda entries: entries[2]["verifier"].update(main_calls=entries[2]["verifier"]["main_calls"] + 1))),
                 "a verifier record of other reads before the start": lambda run: rewrite(run[2], conditions(
                     lambda entries: entries[6]["verifier"].update(prefix_calls=2))),
                 "a verifier record of another prefill chunk": lambda run: rewrite(run[2], conditions(
                     lambda entries: entries[6]["verifier"].update(prefix_chunk=1000))),
                 "a verifier that counted other non-finite values": lambda run: rewrite(run[2], conditions(
                     lambda entries: entries[0]["verifier"].update(non_finite_count=1))),
                 "problems of a comparison given as an empty mapping": lambda run: rewrite(
                     run[2], lambda record: record.update(problems={})),
                 "problems of a condition given as an empty text": lambda run: rewrite(run[2], conditions(
                     lambda entries: entries[0].update(problems=""))),
                 "a reference of one more input": lambda run: self.referenced(
                     lambda record: record["inputs"].update(long=record["inputs"]["short"]), run),
                 "a reference of other versions": lambda run: self.referenced(
                     lambda record: record["versions"].update(free_text="1"), run),
                 "a reference of other times": lambda run: self.referenced(
                     lambda record: record["timing"].update(free_text=1.0), run),
                 "a reference of other memory records": lambda run: self.referenced(
                     lambda record: record["memory"].update(free_text=1), run),
                 "a logits file of another size than the rows of its condition": sized,
                 "a verifier record with a field that the verifier does not write": lambda run: rewrite(run[2], conditions(
                     lambda entries: entries[1]["verifier"].update(cache_query_error=SENTINEL))),
                 "a verifier that specialized an asset that it found in its cache": lambda run: rewrite(
                     run[2], conditions(lambda entries: entries[1]["verifier"].update(specialize_s=0.5))),
                 "a verifier that specialized longer than it loaded": lambda run: rewrite(run[2], conditions(
                     lambda entries: entries[1]["verifier"].update(cached=False, specialize_s=2.0))),
                 "a comparison without problems and without its verifier": lambda run: rewrite(
                     run[2], lambda record: record.update(verifier=None)),
                 "a comparison with problems and with its verifier": lambda run: rewrite(
                     run[2], lambda record: record.update(problems=["the verifier is unreadable (OSError)"], conditions=[],
                                                          passed=False)),
                 "a condition that took less time than its verifier recorded for its steps": lambda run: rewrite(
                     run[2], conditions(lambda entries: entries[0].update(seconds=2.0))),
                 "a reference whose total time is shorter than its steps": lambda run: self.referenced(
                     lambda record: record["timing"].update(total_s=3.0), run),
                 "a reference logits file with bytes after its array": trailing,
                 "a report whose stage times exceed its total": report(lambda record: record["stages"][1].update(prepare_s=1.5)),
                 "a report whose stages take longer than the run": report(lambda record: record.update(elapsed_s=4.0)),
                 "a report of another export time than the manifest": report(
                     lambda record: record["stages"][1].update(execute_s=10.25)),
                 "a report of an export that its stage ran for less time than the export recorded": lambda run: (
                     rewrite(run[0], lambda record: record["stages"][1].update(execute_s=5.0, total_s=6.0)),
                     rewrite(run[0].parent / "manifest.json", lambda record: record.update(execute_s=5.0)))}
        self.assertEqual(self.summarize(self.run_files("consistent"))[0], 0)
        for index, (label, change) in enumerate(cases.items()):
            reference_record(self.reference.parent, self.contract, layers.RESTORED, quantized=reference["quantized"])
            run = self.run_files(f"fixed-{index}")
            change(run)
            with self.subTest(case=label):
                self.assertEqual(self.summarize(run), (1, ""))
            with self.subTest(case=label + ", with the effect of the references that were not changed"), given():
                self.assertEqual(self.summarize(run), (1, ""))
        for index, (label, change) in enumerate(exports.items()):
            reference_record(self.reference.parent, self.contract, layers.RESTORED, quantized=reference["quantized"])
            run = self.run_files(f"export-{index}", change=change)
            self.referenced(lambda record: record["quantized"].update(export_sha256=sha256(run[1])), run)
            with self.subTest(case=label):
                self.assertEqual(self.summarize(run), (1, ""))

    def test_conditions_that_stopped_before_their_numbers_and_a_comparison_with_problems_are_summarized(self):
        run = self.run_files("stopped")
        (run[2].parent / "window-c2.f16").write_bytes(b"ab")

        def stopped(record):
            entries = record["conditions"]
            fixed = [{key: entry[key] for key in fp16_summary.CONDITION_KEYS} for entry in entries[:4]]
            failed = {"checks": {}, "passed": False}
            verifier = {key: value for key, value in entries[2]["verifier"].items() if key != "cached"}
            entries[:4] = [{**fixed[0], **failed, "problems": ["the condition could not be compared (OSError)"]},
                           {**fixed[1], **failed, "problems": ["the verifier did not print one JSON object"], "seconds": 4.0},
                           {**fixed[2], **failed, "problems": ["the verifier record is malformed (cached)"], "seconds": 4.0,
                            "verifier": verifier},
                           {**fixed[3], **failed, "problems": ["the logits file holds 2 B"], "seconds": 4.0,
                            "verifier": entries[3]["verifier"],
                            "output": {"bytes": 2, "sha256": hashlib.sha256(b"ab").hexdigest()}}]
            record["passed"] = False

        rewrite(run[2], stopped)
        code, output = self.summarize(run)
        self.assertEqual(code, 0, self.errors)
        compared = json.loads(output)["assets"][0]["comparison"]
        self.assertEqual((compared["passed"], [entry["passed"] for entry in compared["conditions"][:5]]),
                         (False, [False, False, False, False, True]))
        base = ["name", "input", "tokens", "start", "chunk", "rows", "passed", "problems", "checks"]
        self.assertEqual([list(entry) for entry in compared["conditions"][:4]],
                         [base, [*base, "seconds"], [*base, "seconds", "verifier"], [*base, "seconds", "output", "verifier"]])
        self.assertEqual(compared["conditions"][2]["verifier"]["cached"], None)
        rewrite(run[2], lambda record: record.update(problems=["the verifier is unreadable (FileNotFoundError)"],
                                                     verifier=None, conditions=[], passed=False))
        code, output = self.summarize(run)
        self.assertEqual(code, 0, self.errors)
        compared = json.loads(output)["assets"][0]["comparison"]
        self.assertEqual((compared["passed"], compared["problems"], compared["verifier_sha256"], compared["conditions"]),
                         (False, 1, None, []))

    def test_the_reference_reader_names_each_record_of_the_reference_that_its_writer_does_not_give(self):
        contract, arrays = load_contract(self.contract), self.arrays()
        record = read_json(self.reference)
        self.assertEqual(native.reference_record_problems(record, contract, arrays), [])

        def changed(change, held=arrays):
            copied = copy.deepcopy(record)
            change(copied)
            return native.reference_record_problems(copied, contract, held)

        def nan(name):
            held = arrays[name].copy()
            held.flat[5] = np.nan
            return {**arrays, name: held}

        recount = "the reference recorded another count of the non-finite {} logits than its file holds"

        cases = {"one more input": (lambda value: value["inputs"].update(long=value["inputs"]["short"]),
                                    ["the reference holds other inputs than those of the conditions"]),
                 "more non-finite logits than it kept": (
                     lambda value: value["inputs"]["context"]["logits"].update(
                         non_finite=len(layers.kept_rows("context")) * TEXT["vocab_size"] + 1),
                     [recount.format("context")]),
                 "a negative count of non-finite logits": (
                     lambda value: value["inputs"]["short"]["logits"].update(non_finite=-1), [recount.format("short")]),
                 "one non-finite logit that its file does not hold": (
                     lambda value: value["inputs"]["short"]["logits"].update(non_finite=1), [recount.format("short")]),
                 "a count of non-finite logits of another type": (
                     lambda value: value["inputs"]["short"]["logits"].update(non_finite=0.0), [recount.format("short")]),
                 "no time of an input": (lambda value: value["inputs"]["short"].update(seconds=None),
                                         ["the reference recorded no time of the short input"]),
                 "the version of a package that its writer does not record": (
                     lambda value: value["versions"].update(peft="1.0"),
                     ["the reference recorded the versions of other packages than its writer records"]),
                 "a time of a step that its writer does not take": (
                     lambda value: value["timing"].update(save_s=1.0),
                     ["the reference recorded other times than those of its steps"]),
                 "a negative time of a step": (lambda value: value["timing"].update(load_s=-1.0),
                                               ["the reference recorded other times than those of its steps"]),
                 "a total time shorter than its steps": (lambda value: value["timing"].update(total_s=3.5),
                                                         ["the total time of the reference is shorter than its steps"]),
                 "no peak resident memory": (lambda value: value["memory"].update(max_rss_bytes=None),
                                             ["the reference recorded other memory readings than its writer records"]),
                 "a memory reading that its writer does not record": (
                     lambda value: value["memory"].update(rss_bytes=1),
                     ["the reference recorded other memory readings than its writer records"]),
                 "a negative peak footprint": (lambda value: value["memory"].update(lifetime_peak_footprint_bytes=-1),
                                               ["the reference recorded other memory readings than its writer records"])}
        for label, (change, expected) in cases.items():
            with self.subTest(case=label):
                self.assertEqual(changed(change), expected)
        holding = {"a file that holds a non-finite logit that the record does not count": (lambda value: None, nan("short"),
                                                                                           [recount.format("short")]),
                   "a file that holds the non-finite logit that the record counts": (
                       lambda value: value["inputs"]["short"]["logits"].update(non_finite=1), nan("short"), []),
                   "no array of an input": (lambda value: None, {"short": arrays["short"]}, [recount.format("context")])}
        for label, (change, held, expected) in holding.items():
            with self.subTest(case=label):
                self.assertEqual(changed(change, held), expected)
        path = self.reference.parent / "context.npy"
        path.write_bytes(path.read_bytes() + b"end")
        self.referenced(lambda value: value["inputs"]["context"]["logits"].update(layers.file_record(path)))
        self.assertEqual(native.load_reference(self.reference.parent, contract, self.contract)[3],
                         ["the context logits file holds 3 B after its array"])

    def test_the_reference_reader_names_each_field_of_another_type_and_never_raises(self):
        contract, record, arrays = load_contract(self.contract), read_json(self.reference), self.arrays()
        paths = [("chunk",), ("embed_scale",), ("timing",), ("timing", "input_hash_s"), ("timing", "load_s"),
                 ("timing", "total_s"), ("memory",), ("memory", "max_rss_bytes"), ("memory", "lifetime_peak_footprint_bytes"),
                 ("versions",), ("inputs",)]
        for name in ("short", "context"):
            base = ("inputs", name)
            paths += [base, base + ("calls",), base + ("seconds",), base + ("logits",), base + ("logits", "dtype"),
                      base + ("logits", "shape"), base + ("logits", "non_finite")]
        for path in paths:
            for value in (None, "x", [], {}, True, -1):
                copied = copy.deepcopy(record)
                target = copied
                for part in path[:-1]:
                    target = target[part]
                if canonical(target[path[-1]]) == canonical(value):
                    continue
                target[path[-1]] = value
                with self.subTest(path=path, value=value):
                    self.assertTrue(native.reference_record_problems(copied, contract, arrays))

    def comparison_with(self, run, change):
        rewrite(run[2], change)
        return self.summarize(run)

    def test_a_number_of_a_condition_that_its_logits_file_and_the_reference_do_not_give_stops_the_summary(self):
        run = self.run_files("first")
        original = run[2].read_bytes()
        names = [name for name, *_ in layers.CONDITIONS]

        def condition(name, change):
            return lambda record: change(next(entry for entry in record["conditions"] if entry["name"] == name))

        def uncounted(entry):
            entry.update(non_finite=1)
            entry["verifier"]["non_finite_count"] = 1

        def failed(record):
            entry = next(entry for entry in record["conditions"] if entry["name"] == "window-c8")
            entry["checks"]["finite"] = False
            entry["passed"] = record["passed"] = False

        above = lambda entry: entry.update(max_abs_logit=math.nextafter(entry["max_abs_logit"], 9))
        cases = {"the next number above the largest difference": ("window-c8", above),
                 "the next number below the largest difference": ("window-c8", lambda entry: entry.update(
                     max_abs_logit=math.nextafter(entry["max_abs_logit"], 0))),
                 "no largest difference": ("window-c8", lambda entry: entry.update(max_abs_logit=None)),
                 "one more non-finite value that the verifier counted too": ("window-c8", uncounted),
                 "one more non-finite value of the reference": ("window-c8", lambda entry: entry.update(reference_non_finite=1)),
                 "one more constant row": ("window-c8", lambda entry: entry.update(constant_rows=1)),
                 "one more top-1 change near a tie": ("window-c8", lambda entry: entry["top1_mismatches"].update(near_tie=1)),
                 "one more top-1 change outside the near tie": (
                     "window-c8", lambda entry: entry["top1_mismatches"].update(outside=1)),
                 "one known row less": ("window-c8", lambda entry: entry["known"].update(rows=15)),
                 "the next number above the largest difference of the known rows": (
                     "window-c8", lambda entry: entry["known"].update(
                         max_abs_logit=math.nextafter(entry["known"]["max_abs_logit"], 99))),
                 "a largest difference of known rows that the condition does not have": (
                     "short-c26", lambda entry: entry["known"].update(max_abs_logit=0.5)),
                 **{f"the next number above the largest difference of {name}": (name, above) for name in names}}
        for label, (name, change) in cases.items():
            run[2].write_bytes(original)
            with self.subTest(case=label):
                self.assertEqual(self.comparison_with(run, condition(name, change)), (1, ""))
        run[2].write_bytes(original)
        with self.subTest(case="a failed check with the verdict that it gives"):
            self.assertEqual(self.comparison_with(run, failed), (1, ""))
        run[2].write_bytes(original)
        self.assertEqual(self.summarize(run)[0], 0)

    def test_a_logits_file_replaced_together_with_its_record_is_refused_unless_the_numbers_follow_from_it(self):
        run = self.run_files("first")
        folder, config, arrays = run[2].parent, load_contract(self.contract)["config"], self.arrays()
        shown = ("max_abs_logit", "known", "top1_mismatches", "constant_rows")
        before = read_json(run[2])["conditions"][2]
        logits_file(folder / "window-c1.f16", arrays, "context", 0, 1300, noise=0.002, seed=99)
        output = {"bytes": (folder / "window-c1.f16").stat().st_size, "sha256": sha256(folder / "window-c1.f16")}
        self.assertNotEqual(output, before["output"])
        self.assertEqual(self.comparison_with(run, lambda record: record["conditions"][2].update(output=output)), (1, ""))
        self.assertEqual(self.comparison_with(run, lambda record: measured_entry(record["conditions"][2],
                                                                                  folder / "window-c1.f16", config, arrays))[0], 0)
        after = read_json(run[2])["conditions"][2]
        self.assertNotEqual({key: before[key] for key in shown}, {key: after[key] for key in shown})

    def test_a_logits_file_that_differs_from_its_record_in_a_value_that_no_number_depends_on_is_refused(self):
        run = self.run_files("first")
        path, config, arrays = run[2].parent / "window-c1.f16", load_contract(self.contract)["config"], self.arrays()
        entry, data = read_json(run[2])["conditions"][2], bytearray(path.read_bytes())
        data[2] ^= 1
        self.assertEqual(native.measure_condition(entry, config["vocab_size"], path.read_bytes(), arrays),
                         native.measure_condition(entry, config["vocab_size"], bytes(data), arrays))
        self.assertEqual(self.summarize(run)[0], 0)
        path.write_bytes(bytes(data))
        self.assertEqual(self.summarize(run), (1, ""))

    def test_a_reference_that_counts_other_non_finite_logits_than_its_file_holds_stops_the_summary(self):
        run = self.run_files("finite")
        self.referenced(lambda record: record["inputs"]["short"]["logits"].update(non_finite=1), run)
        self.assertEqual(self.summarize(run), (1, ""))
        held = reference_arrays(self.vocabulary)
        held["short"][3, 5] = np.nan
        reference_record(self.reference.parent, self.contract, layers.RESTORED, quantized=self.quantized, arrays=held)
        self.assertEqual(read_json(self.reference)["inputs"]["short"]["logits"]["non_finite"], 1)
        run = self.run_files("counted")
        code, output = self.summarize(run)
        self.assertEqual(code, 0, self.errors)
        result = json.loads(output)
        conditions = {entry["name"]: entry for entry in result["assets"][0]["comparison"]["conditions"]}
        self.assertEqual({name: (entry["reference_non_finite"], entry["max_abs_logit"], entry["passed"])
                          for name, entry in conditions.items() if name.startswith("short")},
                         {"short-c1": (1, None, False), "short-c26": (1, None, False)})
        self.assertEqual({entry["reference_non_finite"] for name, entry in conditions.items() if not name.startswith("short")},
                         {0})
        self.assertEqual(result["effect"]["conditions"][0]["non_finite"], 1)
        self.referenced(lambda record: record["inputs"]["short"]["logits"].update(non_finite=0), run)
        self.assertEqual(self.summarize(run), (1, ""))
        self.referenced(lambda record: record["inputs"]["short"]["logits"].update(non_finite=1), run)
        self.assertEqual(self.summarize(run)[0], 0)
        self.assertEqual(self.comparison_with(run, lambda record: record["conditions"][0].update(reference_non_finite=0)),
                         (1, ""))

    def test_the_effect_is_computed_from_the_references_that_were_given_and_from_the_record_of_the_restored_one(self):
        run, original = self.run_files("first"), fp16_summary.quantization_effect
        with mock.patch.object(fp16_summary, "quantization_effect", wraps=original) as computed:
            self.assertEqual(self.summarize(run)[0], 0)
        computed.assert_called_once_with(self.reference.parent, self.rounded, self.contract)

        def other(*arguments):
            result = original(*arguments)
            return {**result, "restored": {**result["restored"], "sha256": "0" * 64}}

        with mock.patch.object(fp16_summary, "quantization_effect", side_effect=other):
            self.assertEqual(self.summarize(run), (1, ""))

    def test_the_summary_stops_when_the_reader_of_the_reference_reports_a_problem_or_holds_another_record(self):
        run, original = self.run_files("first"), fp16_summary.load_reference
        self.assertEqual(self.summarize(run)[0], 0)

        def reported(*arguments):
            record, hashed, arrays, found = original(*arguments)
            return record, hashed, arrays, [*found, "the reference holds a problem"]

        def replaced(*arguments):
            record, hashed, arrays, found = original(*arguments)
            return record, "0" * 64, arrays, found

        for label, reader in {"a problem that the reader reports": reported, "another record than the one given": replaced}.items():
            with self.subTest(case=label), mock.patch.object(fp16_summary, "load_reference", side_effect=reader):
                self.assertEqual(self.summarize(run), (1, ""))

    def test_the_summary_measures_a_condition_with_the_function_that_the_comparison_writes_with(self):
        run = self.run_files("first")
        original = native.measure_condition
        self.assertEqual(self.summarize(run)[0], 0)
        shifted = lambda *arguments: {**original(*arguments), "max_abs_logit": 1.5}
        with mock.patch.object(native, "measure_condition", side_effect=shifted):
            self.assertEqual(self.summarize(run), (1, ""))

    def inputs(self, run):
        reference = self.reference.parent
        return {"a short array of the rounded reference": self.rounded / "short.npy",
                "a context array of the rounded reference": self.rounded / "context.npy",
                "the record of the rounded reference": self.rounded / "record.json",
                "a short array of the restored reference": reference / "short.npy",
                "a context array of the restored reference": reference / "context.npy",
                "the record of the restored reference": self.reference, "the report": run[0], "the export record": run[1],
                "the comparison": run[2], "the manifest of the stage": run[0].parent / "manifest.json"}

    def test_a_pipe_without_a_writer_in_place_of_an_input_is_refused_without_waiting(self):
        run = self.run_files("first")
        self.assertEqual(self.summarize_within(run)[0], 0)
        for label, path in self.inputs(run).items():
            saved = path.read_bytes()
            path.unlink()
            os.mkfifo(path)
            with self.subTest(case=label):
                self.assertEqual(self.summarize_within(run), (1, ""))
            path.unlink()
            path.write_bytes(saved)
        self.assertEqual(self.summarize_within(run)[0], 0)

    def test_a_link_or_a_folder_in_place_of_an_input_is_refused(self):
        run = self.run_files("first")
        self.assertEqual(self.summarize(run)[0], 0)
        for kind in ("a link", "a folder"):
            for label, path in self.inputs(run).items():
                saved, other = path.read_bytes(), path.with_name(path.name + ".copy")
                if kind == "a link":
                    path.rename(other)
                    path.symlink_to(other)
                else:
                    path.unlink()
                    path.mkdir()
                with self.subTest(case=kind, place=label):
                    self.assertEqual(self.summarize(run), (1, ""))
                path.unlink() if kind == "a link" else path.rmdir()
                other.unlink(missing_ok=True)
                path.write_bytes(saved)
        self.assertEqual(self.summarize(run)[0], 0)

    def test_a_rounded_reference_that_holds_the_logits_of_the_restored_one_stops_the_summary(self):
        run = self.run_files("first")
        self.assertEqual(self.summarize(run)[0], 0)

        def shared(label, places):
            folder = self.rounded_variant(label)
            record = read_json(folder / "record.json")
            for name, how in places:
                (folder / f"{name}.npy").unlink()
                how(self.reference.parent / f"{name}.npy", folder / f"{name}.npy")
                record["inputs"][name]["logits"].update(layers.file_record(folder / f"{name}.npy"))
            write_json(folder / "record.json", record)
            return folder

        link, copy = os.link, shutil.copyfile
        cases = {"a hard link of the short logits": (("short", link),), "a copy of the short logits": (("short", copy),),
                 "a hard link of the context logits": (("context", link),),
                 "a copy of the context logits": (("context", copy),),
                 "hard links of both": (("short", link), ("context", link)),
                 "copies of both": (("short", copy), ("context", copy))}
        for index, (label, places) in enumerate(cases.items()):
            folder = shared(f"shared-{index}", places)
            with self.subTest(case=label):
                self.assertEqual(effect.effect(self.reference.parent, folder, self.contract)["problems"],
                                 [f"the references hold the same {name} logits" for name, _ in places])
                self.assertEqual(self.summarize(run, options=["--rounded", str(folder)]), (1, ""))

    def test_an_8bit_summary_needs_no_run_of_the_model_class_and_stops_when_one_is_given(self):
        run, missing = self.run_files("first"), str(self.folder / "missing-run")
        cases = {"with the original run": [*self.options(), "--original", missing],
                 "with the model-class run": [*self.options(), "--model-class", missing],
                 "with both runs": [*self.options(), "--original", missing, "--model-class", missing],
                 "with both runs and without the rounded reference": ["--original", missing, "--model-class", missing]}
        for label, options in cases.items():
            with self.subTest(case=label):
                self.assertEqual(self.summarize(run, options=options), (1, ""))
                self.assertEqual(self.errors, "Cannot summarize local records (PipelineError).\n")
        code, output = self.summarize(run)
        self.assertEqual(code, 0)
        self.assertNotIn("model_class", json.loads(output))


if __name__ == "__main__":
    unittest.main()
