import math
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

import coreai_models
from safetensors import safe_open

from Tools.Conversion.adapters.gemma4_31b import export, layers
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import file_hash, read_json
from Tools.Conversion.tests.test_gemma4_31b_layers import SHARD, TEXT, write_snapshot

LAYER = re.compile(r"model\.language_model\.layers\.(\d+)\.")
RECIPE = Path(__file__).resolve().parents[1] / "recipes/gemma4_31b"
CONFIG = RECIPE / "compression-8bit.yaml"
PARTS = ("quantized_data", "scale", "minval")


def checkpoint_elements(snapshot, layers):
    with safe_open(str(snapshot / SHARD), framework="pt") as handle:
        return sum(math.prod(handle.get_slice(name).get_shape()) for name in handle.keys()
                   if not (LAYER.match(name) and int(LAYER.match(name)[1]) >= layers))


def arguments(contract, snapshot, output, functions="main", precision="fp16"):
    upstream = Path(coreai_models.__file__).resolve().parents[3]
    return ["export", "--model", str(contract), "--snapshot", str(snapshot), "--upstream", str(upstream),
            "--layers", "6", "--functions", functions, "--context", "4096", "--precision", precision,
            "--output", str(output), *(["--compression", str(CONFIG)] if precision == "8bit" else [])]


class ExportTests(unittest.TestCase):
    def test_both_function_layouts_of_the_first_layers_pass_the_inspection_and_publish_the_bundle(self):
        upstream = Path(coreai_models.__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            snapshot, contract = write_snapshot(folder)
            elements = checkpoint_elements(snapshot, 6)
            self.assertEqual(export.weight_elements(snapshot, 6), elements)
            for functions in ("main", "main,prefill"):
                output = folder / functions.replace(",", "-")
                argv = ["export", "--model", str(contract), "--snapshot", str(snapshot), "--upstream", str(upstream),
                        "--layers", "6", "--functions", functions, "--context", "4096", "--precision", "fp16",
                        "--output", str(output)]
                with self.subTest(functions=functions), mock.patch("sys.argv", argv):
                    self.assertEqual(export.main(), 0)
                    record = read_json(output / "export.json")
                    self.assertTrue(record["passed"], record["problems"])
                    self.assertEqual(list(record["asset"]["functions"]), functions.split(","))
                    self.assertEqual((record["layers"], record["functions"], record["context"]), (6, functions.split(","), 4096))
                    self.assertEqual(record["weights"], {"tensors": record["weights"]["tensors"], "elements": elements,
                                                         "elements_by_dtype": {"torch.float16": elements},
                                                         "quantized_layouts": 0})
                    storage = record["asset"]["storage_types"]
                    self.assertEqual(storage["Float32"], TEXT["global_head_dim"] // 2 + TEXT["head_dim"] // 2 + 2)
                    self.assertEqual(record["asset"]["float16_minus_weights"], storage["Float16"] - elements)
                    bundle = output / export.BUNDLE
                    files = {path.relative_to(bundle).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                             for path in sorted(bundle.rglob("*")) if path.is_file()}
                    self.assertEqual(record["bundle"]["files"], files)
                    name = record["bundle"]["name"]
                    self.assertEqual(record["bundle"]["main_hash"], (bundle / f"{name}.aimodel/main.hash").read_bytes().hex())
                    self.assertEqual(read_json(bundle / "metadata.json"), export.metadata(read_json(contract), name, 4096))
                    for item in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                        self.assertEqual((bundle / "tokenizer" / item).read_bytes(), (snapshot / item).read_bytes())
                    self.assertFalse((output / export.INCOMPLETE).exists())
                    self.assertFalse((output / export.WORK).exists())
                    self.assertIsNotNone(record["resources"]["swap"]["increase_bytes"])

    def test_the_loading_work_area_is_removed_when_the_export_or_the_save_fails_or_is_interrupted(self):
        upstream = Path(coreai_models.__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            snapshot, contract = write_snapshot(folder)
            for label, error in {"export": RuntimeError("export failed"), "save": RuntimeError("save failed"),
                                 "interruption": KeyboardInterrupt()}.items():
                output, seen = folder / label, []

                class Program:
                    def save_asset(self, *arguments):
                        raise error

                def exported(*arguments, **keywords):
                    seen.append((output / export.WORK).is_dir())
                    if label == "save":
                        return Program()
                    raise error

                argv = ["export", "--model", str(contract), "--snapshot", str(snapshot), "--upstream", str(upstream),
                        "--layers", "6", "--functions", "main", "--context", "4096", "--precision", "fp16",
                        "--output", str(output)]
                with self.subTest(case=label), mock.patch("sys.argv", argv), \
                        mock.patch("coreai_models.export.macos.export_macos_model", exported), \
                        mock.patch.object(export, "supervise", side_effect=lambda folder, schema, work, conclude: work()), \
                        self.assertRaises(type(error)):
                    export.main()
                self.assertEqual(seen, [True])
                self.assertFalse((output / export.WORK).exists())
                self.assertFalse((output / export.BUNDLE).exists())


class QuantizedExportTests(unittest.TestCase):
    def test_the_fixed_config_file_is_read_by_the_converter_as_the_fixed_8bit_config(self):
        from coreai_models.llm.export import _load_compression_config_object

        self.assertEqual(export.normalized(_load_compression_config_object(CONFIG, "macOS")), export.QUANTIZATION)

    def test_the_8bit_export_of_the_first_layers_passes_and_records_the_tensors_that_it_passed_to_the_converter(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            snapshot, contract = write_snapshot(folder, hidden_size=64)
            config, kinds = read_json(contract)["config"], TEXT["layer_types"][:6]
            output = folder / "out"
            with mock.patch("sys.argv", arguments(contract, snapshot, output, "main,prefill", "8bit")):
                self.assertEqual(export.main(), 0)
            record = read_json(output / "export.json")
            self.assertTrue(record["passed"], record["problems"])
            self.assertEqual((record["precision"], record["compression_config"]), ("8bit", export.QUANTIZATION))
            quantized = record["quantized"]
            self.assertEqual(sorted(entry["name"] for entry in quantized["layers"]), sorted(export.quantized_names(kinds)))
            self.assertEqual((quantized["skipped"], quantized["warnings_enabled"], record["weights"]["quantized_layouts"]),
                             (0, True, 41))
            limit = export.smallest_weight(config, kinds)
            self.assertEqual(limit, 2048)
            for key in ("uint8_minus_weights", "float16_minus_weights"):
                self.assertLess(abs(record["asset"][key]), limit)
            storage = record["asset"]["storage_types"]
            self.assertEqual(record["asset"]["uint8_minus_weights"],
                             storage["UInt8"] - record["weights"]["elements_by_dtype"]["torch.uint8"])
            self.assertEqual(storage["Float32"], TEXT["global_head_dim"] // 2 + TEXT["head_dim"] // 2 + 2)
            tensors = output / layers.QUANTIZED
            self.assertEqual(quantized["file"], file_hash(tensors))
            with safe_open(str(tensors), framework="pt") as handle:
                found = {name: layers.tensor_sha256(handle.get_tensor(name)) for name in handle.keys()}
            self.assertEqual(found, quantized["tensors"])
            self.assertEqual(set(found), {f"{name}.{part}" for name in export.quantized_names(kinds) for part in PARTS})
            bundle = output / export.BUNDLE
            self.assertEqual(read_json(bundle / "metadata.json"),
                             export.metadata(read_json(contract), record["bundle"]["name"], 4096, "8bit"))
            self.assertFalse((bundle / layers.QUANTIZED).exists())
            for work in (export.WORK, export.QUANTIZING, export.INCOMPLETE):
                self.assertFalse((output / work).exists())

    def test_an_8bit_export_whose_linear_layers_the_converter_skips_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            snapshot, contract = write_snapshot(folder)
            output = folder / "out"
            with mock.patch("sys.argv", arguments(contract, snapshot, output, "main", "8bit")):
                self.assertEqual(export.main(), 1)
            record = read_json(output / "export.json")
            self.assertFalse(record["passed"])
            self.assertEqual(record["quantized"]["skipped"], 5 * 5 + 4)
            self.assertTrue(any("Skipping quantization." in problem for problem in record["problems"]))
            self.assertFalse((output / export.BUNDLE).exists())

    def test_the_work_areas_are_removed_when_the_8bit_export_fails_or_is_interrupted(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            snapshot, contract = write_snapshot(folder, hidden_size=64)
            for label, error in {"export": RuntimeError("export failed"), "interruption": KeyboardInterrupt()}.items():
                output, seen = folder / label, []

                def exported(*arguments, **keywords):
                    seen.append(((output / export.WORK).is_dir(), (output / export.QUANTIZING).is_dir(),
                                 (output / layers.QUANTIZED).is_file()))
                    raise error

                with self.subTest(case=label), mock.patch("sys.argv", arguments(contract, snapshot, output, "main",
                                                                                  "8bit")), \
                        mock.patch("coreai_models.export.macos.export_macos_model", exported), \
                        mock.patch.object(export, "supervise", side_effect=lambda folder, schema, work, conclude: work()), \
                        self.assertRaises(type(error)):
                    export.main()
                self.assertEqual(seen, [(True, True, True)])
                for work in (export.WORK, export.QUANTIZING, export.BUNDLE):
                    self.assertFalse((output / work).exists())



if __name__ == "__main__":
    unittest.main()
