import math
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

import coreai_models
from safetensors import safe_open

from Tools.Conversion.adapters.gemma4_31b import export
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import read_json
from Tools.Conversion.tests.test_gemma4_31b_layers import SHARD, TEXT, write_snapshot

LAYER = re.compile(r"model\.language_model\.layers\.(\d+)\.")


def checkpoint_elements(snapshot, layers):
    with safe_open(str(snapshot / SHARD), framework="pt") as handle:
        return sum(math.prod(handle.get_slice(name).get_shape()) for name in handle.keys()
                   if not (LAYER.match(name) and int(LAYER.match(name)[1]) >= layers))


class ExportTests(unittest.TestCase):
    def test_both_function_layouts_of_the_first_layers_pass_the_inspection_and_publish_the_bundle(self):
        upstream = Path(coreai_models.__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            snapshot, contract = write_snapshot(folder)
            elements = checkpoint_elements(snapshot, 6)
            self.assertEqual(export.weight_elements(snapshot, 6), elements)
            for functions, taps in (("main", "2,6"), ("main,prefill", "1,2,3,4,5")):
                output = folder / functions.replace(",", "-")
                argv = ["export", "--model", str(contract), "--snapshot", str(snapshot), "--upstream", str(upstream),
                        "--layers", "6", "--functions", functions, "--taps", taps, "--context", "4096",
                        "--output", str(output)]
                width = TEXT["hidden_size"] * len(taps.split(","))
                with self.subTest(functions=functions), mock.patch("sys.argv", argv):
                    self.assertEqual(export.main(), 0)
                    record = read_json(output / "export.json")
                    self.assertTrue(record["passed"], record["problems"])
                    self.assertEqual(list(record["asset"]["functions"]), functions.split(","))
                    found = {name: entry["outputs"] for name, entry in record["asset"]["functions"].items()}
                    tapped = {"tap_hidden_states": {"dtype": "float16", "shape": [1, None, width]}}
                    self.assertEqual(found["main"], {"logits": {"dtype": "float16", "shape": [1, None, TEXT["vocab_size"]]},
                                                     "hidden_states": {"dtype": "float16",
                                                                       "shape": [1, None, TEXT["hidden_size"]]},
                                                     **tapped})
                    self.assertEqual(found.get("prefill", tapped), tapped)
                    self.assertEqual(record["asset"]["output_order"],
                                     {"main": ["logits", "hidden_states", "tap_hidden_states"],
                                      "prefill": ["tap_hidden_states"]} if "prefill" in found
                                     else {"main": ["logits", "hidden_states", "tap_hidden_states"]})
                    self.assertEqual(record["metadata"]["tap_layers"], [int(tap) for tap in taps.split(",")])
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
                    self.assertEqual(read_json(bundle / "metadata.json"),
                                     export.metadata(read_json(contract), name, 4096, export.check_taps(taps, 6)))
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
                        "--layers", "6", "--functions", "main", "--taps", "1", "--context", "4096", "--output", str(output)]
                with self.subTest(case=label), mock.patch("sys.argv", argv), \
                        mock.patch.object(export, "staged", exported), \
                        mock.patch.object(export, "supervise", side_effect=lambda folder, schema, work, conclude: work()), \
                        self.assertRaises(type(error)):
                    export.main()
                self.assertEqual(seen, [True])
                self.assertFalse((output / export.WORK).exists())
                self.assertFalse((output / export.BUNDLE).exists())


if __name__ == "__main__":
    unittest.main()
