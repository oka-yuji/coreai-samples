import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

from Tools.Conversion.adapters.gemma4_31b import effect, layers
from Tools.Conversion.adapters.gemma4_31b.prepare import TOKENIZER_FILES, load_contract
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import digest, read_json, write_json
from Tools.Conversion.tests.test_gemma4_31b_layers import TEXT, write_contract
from Tools.Conversion.tests.test_gemma4_31b_native import bounded, model_record, release

VOCABULARY = TEXT["vocab_size"]


@unittest.skipIf(np is None, "NumPy is required for the comparison")
class EffectTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.contract = write_contract(self.folder, dict.fromkeys(TOKENIZER_FILES, {"bytes": 1, "sha256": "0" * 64}))
        self.config = load_contract(self.contract)["config"]
        generator = np.random.default_rng(2)
        rounded = {}
        for name in ("short", "context"):
            rounded[name] = generator.normal(size=(len(layers.kept_rows(name)), VOCABULARY)).astype(np.float32)
            rounded[name][:, 0] += 20
        restored = {name: array.copy() for name, array in rounded.items()}
        restored["short"][3, 7] += 1.5
        rows = layers.kept_rows("context")
        restored["context"][rows.index(1100), 1] = restored["context"][rows.index(1100), 0] + 1.0
        rounded["context"][rows.index(16890), :2] = (30.005, 30.0)
        restored["context"][rows.index(16890), :2] = (30.0, 30.5)
        self.arrays = rounded
        self.rounded = self.write("rounded", layers.WEIGHTS, rounded)
        self.restored = self.write("restored", layers.RESTORED, restored,
                                   quantized={"export_sha256": "1" * 64, "file": {"bytes": 1, "sha256": "2" * 64},
                                              "tensors": 123})

    def write(self, label, weights, arrays, **extra):
        folder = self.folder / label
        folder.mkdir()
        inputs = {}
        for name, array in arrays.items():
            np.save(folder / f"{name}.npy", array)
            ids, rows = layers.tokens(name, self.config), layers.kept_rows(name)
            inputs[name] = {"positions": len(ids), "tokens_sha256": digest(ids), "rows": rows,
                            "logits": layers.file_record(folder / f"{name}.npy")}
        write_json(folder / "record.json", {"schema": layers.CONTEXT_SCHEMA, "implementation": "original", "layers": 6,
                                            "layer_types": layers.check_layers(self.config, 6), "chunk": 1024,
                                            "weights": weights, "model": model_record(self.contract), "inputs": inputs,
                                            **extra})
        return folder

    def run_main(self, restored, rounded):
        output, errors = io.StringIO(), io.StringIO()
        argv = ["effect", "--restored", str(restored), "--rounded", str(rounded)]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(effect, "CONTRACT", self.contract), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = effect.main()
        return code, output.getvalue(), errors.getvalue()

    def test_the_effect_of_the_quantization_is_printed_for_each_condition_without_a_judgment(self):
        code, output, _ = self.run_main(self.restored, self.rounded)
        self.assertEqual(code, 0)
        result = json.loads(output)
        self.assertEqual((result["schema"], result["problems"]), (effect.SCHEMA, []))
        self.assertEqual((result["restored"]["sha256"], result["rounded"]["sha256"]),
                         (sha256(self.restored / "record.json"), sha256(self.rounded / "record.json")))
        conditions = {entry["name"]: entry for entry in result["conditions"]}
        self.assertEqual(list(conditions), [name for name, *_ in layers.CONDITIONS])
        self.assertTrue(all("checks" not in entry and "passed" not in entry for entry in conditions.values()))
        self.assertAlmostEqual(conditions["short-c1"]["max_abs_logit"], 1.5, places=5)
        self.assertEqual(conditions["window-c1"]["top1_mismatches"], {"near_tie": 0, "outside": 1})
        self.assertEqual(conditions["tail"]["top1_mismatches"], {"near_tie": 1, "outside": 0})
        self.assertEqual(conditions["capacity-2048"]["max_abs_logit"], 0.0)
        self.assertEqual({name: entry["rows"] for name, entry in conditions.items()},
                         {name: length - start for name, _, length, start, _ in layers.CONDITIONS})
        for folder in (self.restored, self.rounded):
            self.assertEqual(sorted(path.name for path in folder.iterdir()), ["context.npy", "record.json", "short.npy"])

    def twelve_layers(self, label, weights, **extra):
        folder = self.write(label, weights, self.arrays, **extra)
        record = read_json(folder / "record.json")
        record.update(layers=12, layer_types=layers.check_layers(self.config, 12))
        write_json(folder / "record.json", record)
        return folder

    def test_references_of_other_weights_or_layers_are_refused_and_nothing_is_compared(self):
        checks = read_json(self.restored / "record.json")["quantized"]
        restored_weights = "the restored reference does not use the restored weights"
        rounded_weights = "the rounded reference does not use the rounded weights"
        layer_counts = "the references are of other layer counts"
        cases = {"two rounded references": (self.rounded, self.rounded, restored_weights),
                 "swapped references": (self.rounded, self.restored, rounded_weights),
                 "a restored reference of other layers": (self.twelve_layers("other-layers", layers.RESTORED), self.rounded,
                                                          layer_counts),
                 "rounded weights with the checks of the quantized tensors": (
                     self.write("rounded-with-checks", layers.WEIGHTS, self.arrays, quantized=checks), self.rounded,
                     restored_weights),
                 "two restored references": (
                     self.restored, self.write("second-restored", layers.RESTORED, self.arrays, quantized=checks),
                     rounded_weights),
                 "a rounded reference of other layers": (self.restored, self.twelve_layers("rounded-twelve", layers.WEIGHTS),
                                                         layer_counts)}
        for label, (restored, rounded, expected) in cases.items():
            with self.subTest(case=label):
                code, output, _ = self.run_main(restored, rounded)
                result = json.loads(output)
                self.assertEqual((code, result["conditions"]), (1, []))
                self.assertIn(expected, result["problems"])

    def test_a_restored_reference_without_the_checks_of_every_quantized_tensor_is_refused(self):
        cases = {"one tensor less": {"export_sha256": "1" * 64, "file": {"bytes": 1, "sha256": "2" * 64}, "tensors": 122},
                 "no hash of the export record": {"file": {"bytes": 1, "sha256": "2" * 64}, "tensors": 123},
                 "free text as the hash": {"export_sha256": "private", "file": {}, "tensors": 123},
                 "a tensor count as a float": {"export_sha256": "1" * 64, "file": {"bytes": 1, "sha256": "2" * 64},
                                               "tensors": 123.0},
                 "a list": ["tensors"]}
        for label, quantized in cases.items():
            restored = self.write(label.replace(" ", "-"), layers.RESTORED, self.arrays, quantized=quantized)
            with self.subTest(case=label):
                code, output, _ = self.run_main(restored, self.rounded)
                result = json.loads(output)
                self.assertEqual((code, result["conditions"]), (1, []))
                self.assertTrue(result["problems"])

    def test_references_that_hold_the_same_logits_are_refused_and_nothing_is_compared(self):
        def shared(label, places):
            folder = self.folder / label
            shutil.copytree(self.rounded, folder)
            record = read_json(folder / "record.json")
            for name, how in places:
                (folder / f"{name}.npy").unlink()
                how(self.restored / f"{name}.npy", folder / f"{name}.npy")
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
            with self.subTest(case=label):
                code, output, _ = self.run_main(self.restored, shared(f"shared-{index}", places))
                result = json.loads(output)
                self.assertEqual((code, result["conditions"], result["problems"]),
                                 (1, [], [f"the references hold the same {name} logits" for name, _ in places]))
        self.assertEqual(self.run_main(self.restored, self.rounded)[0], 0)

    def test_a_reference_does_not_wait_on_a_pipe_and_a_link_or_a_folder_is_refused(self):
        def piped(path):
            path.unlink()
            os.mkfifo(path)

        def linked(path):
            path.rename(path.with_name(path.name + ".copy"))
            path.symlink_to(path.with_name(path.name + ".copy"))

        def folder(path):
            path.unlink()
            path.mkdir()

        places = {"an array of the rounded reference": ("rounded", "short.npy"),
                  "the record of the rounded reference": ("rounded", "record.json"),
                  "an array of the restored reference": ("restored", "context.npy")}
        index = 0
        for kind, make in {"a pipe": piped, "a link": linked, "a folder": folder}.items():
            for label, (which, name) in places.items():
                index += 1
                restored, rounded = self.folder / f"restored-{index}", self.folder / f"rounded-{index}"
                shutil.copytree(self.restored, restored)
                shutil.copytree(self.rounded, rounded)
                path = (restored if which == "restored" else rounded) / name
                make(path)
                with self.subTest(case=kind, place=label):
                    thread, outcome = bounded(lambda: effect.effect(restored, rounded, self.contract))
                    self.assertFalse(release(path, thread))
                    self.assertIsInstance(outcome.get("error"), (OSError, ValueError))

    def test_an_unreadable_reference_stops_with_the_fixed_error_and_prints_nothing(self):
        (self.restored / "record.json").write_text("{")
        code, output, errors = self.run_main(self.restored, self.rounded)
        self.assertEqual((code, output), (1, ""))
        self.assertRegex(errors, r"^Cannot compare the references \(\w+\)\.\n$")


if __name__ == "__main__":
    unittest.main()
