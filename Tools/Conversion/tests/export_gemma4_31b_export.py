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
from Tools.Conversion.tests.test_gemma4_31b_layers import SHARD, write_snapshot

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
            for functions in ("main", "main,prefill"):
                output = folder / functions.replace(",", "-")
                argv = ["export", "--model", str(contract), "--snapshot", str(snapshot), "--upstream", str(upstream),
                        "--layers", "6", "--functions", functions, "--context", "4096", "--output", str(output)]
                with self.subTest(functions=functions), mock.patch("sys.argv", argv):
                    self.assertEqual(export.main(), 0)
                    record = read_json(output / "export.json")
                    self.assertTrue(record["passed"], record["problems"])
                    self.assertEqual(list(record["asset"]["functions"]), functions.split(","))
                    self.assertEqual((record["layers"], record["functions"], record["context"]), (6, functions.split(","), 4096))
                    self.assertEqual(record["weights"], {"tensors": record["weights"]["tensors"], "elements": elements,
                                                         "elements_by_dtype": {"torch.float16": elements},
                                                         "quantized_layouts": 0})
                    self.assertGreaterEqual(record["asset"]["storage_types"]["Float16"], elements)
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


if __name__ == "__main__":
    unittest.main()
