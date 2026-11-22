import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from Tools.Conversion.adapters.gemma4_31b import export
from Tools.Conversion.adapters.gemma4_31b.layers import check_layers
from Tools.Conversion.adapters.gemma4_31b.prepare import load_contract
from Tools.Conversion.recipe import expand, load_recipe
from Tools.Conversion.storage import PipelineError, read_json

REPOSITORY = Path(__file__).resolve().parents[3]
RECIPE = Path(__file__).resolve().parents[1] / "recipes/gemma4_31b"
CONTRACT = load_contract(RECIPE / "model.json")
KINDS = check_layers(CONTRACT["config"], 6)
ELEMENTS = 4_338_096_902
BOTH = ("main", "prefill")
STATES = {"keyCache": [1, 1, 4, None, 512], "valueCache": [1, 1, 4, None, 512],
          "slidingKeyCache": [5, 1, 16, None, 256], "slidingValueCache": [5, 1, 16, None, 256]}
STORAGE = {"Float16": ELEMENTS + 4, "Float32": 386, "Int32": 148, "UInt32": 29, "UInt64": 1}
BINDINGS = {"python": "/reference/bin/python", "hf_cache": "/cache", "setup_record": "/setup.json", "snapshot": "/model",
            "upstream": "/upstream"}


def function(outputs):
    return {"inputs": {name: {"dtype": "int32", "shape": [1, None]} for name in ("input_ids", "position_ids")},
            "outputs": outputs, "states": {name: {"dtype": "float16", "shape": list(shape)} for name, shape in STATES.items()}}


LAYOUT = {"main": function({"logits": {"dtype": "float16", "shape": [1, None, 262144]}}), "prefill": function({})}


def inspected(functions=BOTH):
    return {"compression_config": None, "weights": {"elements": ELEMENTS, "quantized_layouts": 0},
            "asset": {"functions": {name: copy.deepcopy(LAYOUT[name]) for name in functions},
                      "storage_types": dict(STORAGE)},
            "tokenizer": dict(CONTRACT["tokenizer"]["files"]),
            "metadata": export.metadata(CONTRACT, export.name_of(6, functions), 16896)}


def problems(change=None, functions=BOTH):
    record = inspected(functions)
    if change:
        change(record)
    return export.check_asset(record, CONTRACT, KINDS, functions, 16896)


def state(name, shape, function_name="main"):
    def change(record):
        record["asset"]["functions"][function_name]["states"][name]["shape"] = shape
    return change


def storage(types):
    def change(record):
        record["asset"]["storage_types"] = types
    return change


class AssetTests(unittest.TestCase):
    def test_the_measured_layout_of_the_first_six_layers_passes_in_both_function_layouts(self):
        for functions in (("main",), BOTH):
            with self.subTest(functions=functions):
                self.assertEqual(problems(functions=functions), [])
        self.assertEqual(export.metadata(CONTRACT, "bundle-name", 16896), {
            "metadata_version": "0.2", "kind": "llm", "name": "bundle-name", "assets": {"main": "bundle-name.aimodel"},
            "language": {"tokenizer": "google/gemma-4-31B-it", "vocab_size": 262144, "max_context_length": 16896,
                         "embedded_tokenizer": True, "function_map": {"main": ["main"]}},
            "source": {"model_definition": "torch", "hf_model_id": "google/gemma-4-31B-it",
                       "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475"},
            "compression": None})

    def test_the_states_follow_the_layer_pattern_of_the_layer_count(self):
        for layers, full, sliding in ((6, 1, 5), (30, 5, 25), (60, 10, 50)):
            functions = export.expected_functions(CONTRACT["config"], check_layers(CONTRACT["config"], layers), BOTH)
            with self.subTest(layers=layers):
                self.assertEqual(set(functions), set(BOTH))
                self.assertEqual(functions["prefill"]["outputs"], {})
                for name in BOTH:
                    shapes = {key: value["shape"] for key, value in functions[name]["states"].items()}
                    self.assertEqual(shapes, {"keyCache": [full, 1, 4, None, 512], "valueCache": [full, 1, 4, None, 512],
                                              "slidingKeyCache": [sliding, 1, 16, None, 256],
                                              "slidingValueCache": [sliding, 1, 16, None, 256]})

    def test_an_asset_with_a_compression_setting_fails(self):
        cases = {"a compression config passed to the export": lambda record: record.update(
                     compression_config={"quantization_config": {"module_type_configs": {}}}),
                 "quantized weights in the PyTorch model": lambda record: record["weights"].update(quantized_layouts=41),
                 "compression named in metadata.json": lambda record: record["metadata"].update(compression="8bit")}
        for label, change in cases.items():
            with self.subTest(case=label):
                self.assertTrue(problems(change))

    def test_an_asset_with_weights_stored_in_another_type_than_float16_fails(self):
        cases = {"uint8 weights": {"UInt8": 2_930_049_024, "Float16": 1_592_466_186, "Float32": 386, "Int32": 148,
                                   "UInt32": 29, "UInt64": 1},
                 "uint4 weights": {**STORAGE, "UInt4": 64},
                 "int8 weights": {**STORAGE, "Int8": 64},
                 "bfloat16 weights": {"BFloat16": ELEMENTS + 4, "Float32": 386, "Int32": 148, "UInt32": 29, "UInt64": 1},
                 "a norm weight stored in float32": {**STORAGE, "Float16": ELEMENTS + 4 - 5376, "Float32": 386 + 5376},
                 "every weight in float32": {"Float32": ELEMENTS + 390, "Int32": 148, "UInt32": 29, "UInt64": 1}}
        for label, types in cases.items():
            with self.subTest(case=label):
                self.assertTrue(problems(storage(types)))
        self.assertEqual(problems(storage({**STORAGE, "Float16": ELEMENTS})), [])

    def test_an_asset_whose_states_do_not_have_exactly_one_dynamic_dimension_fails(self):
        cases = {"two dynamic dimensions": state("keyCache", [1, 1, 4, None, None]),
                 "no dynamic dimension": state("slidingValueCache", [5, 1, 16, 16896, 256]),
                 "a dynamic layer axis": state("slidingKeyCache", [None, 1, 16, None, 256], "prefill"),
                 "a missing state": lambda record: record["asset"]["functions"]["main"]["states"].pop("valueCache"),
                 "a fifth state": lambda record: record["asset"]["functions"]["main"]["states"].update(
                     extraCache={"dtype": "float16", "shape": [1, None]})}
        for label, change in cases.items():
            with self.subTest(case=label):
                self.assertTrue(problems(change))

    def test_an_asset_whose_tokenizer_differs_from_the_distributed_files_fails(self):
        cases = {"another tokenizer.json": lambda record: record["tokenizer"].update({"tokenizer.json": "0" * 64}),
                 "no chat template": lambda record: record["tokenizer"].pop("chat_template.jinja"),
                 "an extra file": lambda record: record["tokenizer"].update({"special_tokens_map.json": "0" * 64})}
        for label, change in cases.items():
            with self.subTest(case=label):
                self.assertTrue(problems(change))

    def test_other_functions_inputs_outputs_or_metadata_fail(self):
        def logits(record):
            record["asset"]["functions"]["prefill"]["outputs"] = copy.deepcopy(LAYOUT["main"]["outputs"])

        def inputs(record):
            record["asset"]["functions"]["main"]["inputs"]["input_ids"]["dtype"] = "int64"

        def vocabulary(record):
            record["asset"]["functions"]["main"]["outputs"]["logits"]["shape"] = [1, None, 262143]

        cases = {"main only where both were asked": (None, ("main",), BOTH),
                 "both where main only was asked": (None, BOTH, ("main",)),
                 "logits from prefill": (logits, BOTH, BOTH), "int64 input IDs": (inputs, BOTH, BOTH),
                 "another vocabulary": (vocabulary, BOTH, BOTH),
                 "another context limit": (lambda record: record["metadata"]["language"].update(max_context_length=16384),
                                           BOTH, BOTH),
                 "another vocabulary in metadata.json": (
                     lambda record: record["metadata"]["language"].update(vocab_size=262143), BOTH, BOTH),
                 "another source model": (lambda record: record["metadata"]["source"].update(hf_model_id="google/other"),
                                          BOTH, BOTH),
                 "another source revision": (lambda record: record["metadata"]["source"].update(revision="0" * 40),
                                             BOTH, BOTH)}
        for label, (change, built, asked) in cases.items():
            record = inspected(built)
            if change:
                change(record)
            with self.subTest(case=label):
                self.assertTrue(export.check_asset(record, CONTRACT, KINDS, asked, 16896))

    def test_asset_types_are_read_from_the_summary_text(self):
        self.assertEqual(export.described("NDArray (Float16, 5 × 1 × 16 × ? × 256)"),
                         {"dtype": "float16", "shape": [5, 1, 16, None, 256]})
        self.assertEqual(export.described("NDArray (Int32, 1 × ?)"), {"dtype": "int32", "shape": [1, None]})
        self.assertEqual(export.described("Tensor <float16>"), {"raw": "Tensor <float16>"})


def arguments(folder, output, layers="6", functions="main,prefill"):
    return ["export", "--model", str(RECIPE / "model.json"), "--snapshot", str(folder / "snapshot"),
            "--upstream", str(folder / "upstream"), "--layers", layers, "--functions", functions,
            "--context", "16896", "--output", str(output)]


def observed(output, lowest):
    return {"samples": 3, "interval_s": 5, "aborts": 0, "failure": None, "unreadable_samples": 0,
            "first": {"swap_used_bytes": 7}, "peak": {"swap_used_bytes": 9, "rss_bytes": 11},
            "lowest": {"free_percent": 40, "free_bytes": {str(output): lowest}}}


class StartTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.folder))

    def main(self, argv, free=10**15):
        usage = mock.Mock(free=free)
        with mock.patch.object(sys, "argv", argv), mock.patch.object(export, "weight_elements", return_value=ELEMENTS) as count, \
                mock.patch.object(export.shutil, "disk_usage", return_value=usage), \
                mock.patch.object(export, "verify_inputs") as verify, \
                mock.patch.object(export, "supervise", return_value=0) as supervise:
            try:
                return export.main(), count, verify, supervise
            except BaseException as error:
                error.calls = count, verify, supervise
                raise

    def test_an_existing_output_is_refused_before_anything_is_read_and_is_left_as_it_was(self):
        target = self.folder / "target"
        target.write_text("target")
        outputs = {"file": lambda path: path.write_text("earlier"), "folder": lambda path: path.mkdir(),
                   "link": lambda path: path.symlink_to(target),
                   "dangling link": lambda path: path.symlink_to(self.folder / "missing")}
        for label, make in outputs.items():
            output = self.folder / label.replace(" ", "-")
            make(output)
            before = (os.readlink(output) if output.is_symlink() else None, output.read_bytes() if output.is_file() else None)
            with self.subTest(output=label):
                with self.assertRaises(FileExistsError) as raised:
                    self.main(arguments(self.folder, output))
                for call in raised.exception.calls:
                    call.assert_not_called()
                self.assertEqual((os.readlink(output) if output.is_symlink() else None,
                                  output.read_bytes() if output.is_file() else None), before)

    def test_an_existing_output_is_refused_before_the_contract_is_read(self):
        broken = self.folder / "broken.json"
        broken.write_text("{")
        outputs = {"file": lambda path: path.write_text("earlier"), "folder": lambda path: path.mkdir(),
                   "dangling link": lambda path: path.symlink_to(self.folder / "missing")}
        for label, make in outputs.items():
            output = self.folder / f"broken-{label.replace(' ', '-')}"
            make(output)
            before = (os.readlink(output) if output.is_symlink() else None, output.read_bytes() if output.is_file() else None)
            argv = arguments(self.folder, output)
            argv[argv.index("--model") + 1] = str(broken)
            with self.subTest(output=label), mock.patch.object(export, "load_contract", wraps=export.load_contract) as load:
                with self.assertRaises(FileExistsError) as raised:
                    self.main(argv)
                load.assert_not_called()
                for call in raised.exception.calls:
                    call.assert_not_called()
            self.assertEqual((os.readlink(output) if output.is_symlink() else None,
                              output.read_bytes() if output.is_file() else None), before)

    def test_first_layers_that_do_not_end_with_a_full_attention_layer_are_refused_before_the_output_is_made(self):
        for layers in ("0", "5", "7", "61", "-6"):
            output = self.folder / f"layers-{layers}"
            with self.subTest(layers=layers):
                with self.assertRaisesRegex(ValueError, "must end with a full-attention layer") as raised:
                    self.main(arguments(self.folder, output, layers=layers))
                for call in raised.exception.calls:
                    call.assert_not_called()
                self.assertFalse(output.exists())

    def test_the_export_starts_only_with_room_for_twice_the_float16_weights_and_60_gib(self):
        required = 4 * ELEMENTS + 60 * 2**30
        self.assertEqual(export.required_bytes(ELEMENTS), 81_776_897_048)
        self.assertEqual(export.required_bytes(ELEMENTS), required)
        with self.assertRaisesRegex(PipelineError, "free") as raised:
            self.main(arguments(self.folder, self.folder / "short"), free=required - 1)
        count, verify, supervise = raised.exception.calls
        count.assert_called_once()
        verify.assert_not_called()
        supervise.assert_not_called()
        code, count, verify, supervise = self.main(arguments(self.folder, self.folder / "enough"), free=required)
        self.assertEqual(code, 0)
        supervise.assert_called_once()
        self.assertEqual(supervise.call_args.args[0], self.folder / "enough")
        self.assertTrue((self.folder / "enough").is_dir())

    def test_a_function_layout_other_than_main_or_main_and_prefill_is_refused(self):
        for functions in ("prefill", "main,main", "prefill,main", ""):
            with self.subTest(functions=functions), self.assertRaises(SystemExit):
                with mock.patch("sys.stderr"):
                    self.main(arguments(self.folder, self.folder / "functions", functions=functions))
        self.assertFalse((self.folder / "functions").exists())


class RecordTests(unittest.TestCase):
    def conclude(self, lowest, asset_problems=()):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(folder))
        output = folder / "out"
        (output / export.INCOMPLETE).mkdir(parents=True)
        record = {"problems": list(asset_problems), "disk": {"required_free_bytes": 1}, "timing": {},
                  "bundle": {"folder": export.INCOMPLETE}}
        code = export.conclude(output, record, observed(output, lowest), 0.0)
        return code, read_json(output / "export.json"), output

    def test_a_run_whose_free_disk_fell_below_60_gib_fails_and_keeps_the_bundle_unpublished(self):
        code, record, output = self.conclude(60 * 2**30 - 1)
        self.assertEqual((code, record["passed"], record["checks"]["free_disk"]), (1, False, False))
        self.assertEqual(record["disk"]["lowest_free_bytes"], 60 * 2**30 - 1)
        self.assertTrue((output / export.INCOMPLETE).is_dir())
        self.assertFalse((output / export.BUNDLE).exists())
        code, record, output = self.conclude(None)
        self.assertEqual((code, record["passed"]), (1, False))

    def test_a_run_with_enough_disk_and_no_asset_problem_publishes_the_bundle_and_records_the_resources(self):
        code, record, output = self.conclude(60 * 2**30)
        self.assertEqual((code, record["passed"], record["checks"]), (0, True, {"asset": True, "free_disk": True}))
        self.assertEqual(record["bundle"]["folder"], export.BUNDLE)
        self.assertTrue((output / export.BUNDLE).is_dir())
        self.assertFalse((output / export.INCOMPLETE).exists())
        swap = record["resources"]["swap"]
        self.assertEqual((swap["first_bytes"], swap["peak_bytes"], swap["increase_bytes"]), (7, 9, 2))
        self.assertGreater(record["resources"]["max_rss_bytes"], 0)
        self.assertIn("total_s", record["timing"])

    def test_a_run_with_an_asset_problem_fails_even_with_enough_disk(self):
        code, record, output = self.conclude(10**15, ["main: state keyCache"])
        self.assertEqual((code, record["passed"], record["checks"]["asset"]), (1, False, False))
        self.assertFalse((output / export.BUNDLE).exists())


class RecipeTests(unittest.TestCase):
    def load(self, **variables):
        return load_recipe(RECIPE / "recipe-fp16.json", {**BINDINGS, **variables}, REPOSITORY)

    def test_the_recipe_stops_without_a_layer_count_or_a_function_layout(self):
        for missing in ("layers", "functions"):
            supplied = {"layers": "6", "functions": "main"}
            del supplied[missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(PipelineError, "Unresolved"):
                self.load(**supplied)

    def test_the_export_step_passes_the_choices_and_writes_into_a_new_folder_of_its_output(self):
        recipe = self.load(layers="6", functions="main,prefill")
        prepare, step = recipe["steps"]
        command = expand(step["command"], {**recipe["variables"], "out": "/stage", "prepare": "/prepared"})
        self.assertEqual(command[:3], ["/upstream/.venv/bin/python", "-m", "Tools.Conversion.adapters.gemma4_31b.export"])
        options = dict(zip(command[3::2], command[4::2]))
        self.assertEqual(options, {"--model": str(RECIPE / "model.json"), "--snapshot": "/model", "--upstream": "/upstream",
                                   "--layers": "6", "--functions": "main,prefill", "--context": "16896",
                                   "--output": "/stage/export"})
        self.assertEqual((step["role"], step["heavy"], step["depends"], step["outputs"]),
                         ("export", True, ["prepare"], ["export"]))
        self.assertEqual(step["checks"], [{"file": "export/export.json", "key": ["passed"], "equals": True}])
        sources = read_json(RECIPE / "recipe-fp16.json")
        self.assertEqual(sources["steps"][0], read_json(RECIPE / "recipe.json")["steps"][0])
        self.assertEqual(sources["environment"], read_json(RECIPE / "recipe.json")["environment"])


if __name__ == "__main__":
    unittest.main()
