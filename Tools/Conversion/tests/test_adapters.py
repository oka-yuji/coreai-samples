import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import types

ADAPTERS = Path(__file__).resolve().parents[1] / "adapters"
sys.path.insert(0, str(ADAPTERS))
from pinned import apple_toolchain, load_contract, pinned_hub
from verify_llm import compare, decode_swift
import export_llm
from reference_llm import encode_case
import verify_llm


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.revision = "a" * 40
        self.snapshot = self.root / "snapshots" / self.revision
        self.snapshot.mkdir(parents=True)
        for name, data in {"config.json": '{"model_type":"qwen3"}', "tokenizer.json": "{}",
                           "tokenizer_config.json": "{}", "model.safetensors": "weights"}.items():
            (self.snapshot / name).write_text(data)
        self.contract = {"model_id": "Qwen/Qwen3-0.6B", "revision": self.revision,
                         "model_type": "qwen3", "precision": "float16", "compression": "none",
                         "context_length": 1024, "output_name": "qwen3_06b_fp16",
                         "upstream_revision": "b" * 40}
        self.contract_path = self.root / "model.json"
        self.contract_path.write_text(json.dumps(self.contract))

    def tearDown(self):
        self.temp.cleanup()

    def test_private_hf_main_reference_does_not_modify_original_cache(self):
        original = self.root / "original_hub"
        original.mkdir()
        ref = original / "main"
        ref.write_text("old-revision")
        before = {p.name: p.read_bytes() for p in self.snapshot.iterdir()}
        previous = os.environ.get("HF_HUB_CACHE")
        os.environ["HF_HUB_CACHE"] = str(original)
        try:
            with pinned_hub(self.snapshot, self.contract) as hub:
                repository = hub / "models--Qwen--Qwen3-0.6B"
                self.assertEqual((repository / "refs/main").read_text(), self.revision)
                copy = repository / "snapshots" / self.revision / "model.safetensors"
                self.assertEqual(copy.read_bytes(), b"weights")
                self.assertNotEqual(str(hub), str(original))
            self.assertFalse(hub.exists())
            self.assertEqual(ref.read_text(), "old-revision")
            self.assertEqual(os.environ["HF_HUB_CACHE"], str(original))
            self.assertEqual(before, {p.name: p.read_bytes() for p in self.snapshot.iterdir()})
        finally:
            if previous is None:
                os.environ.pop("HF_HUB_CACHE", None)
            else:
                os.environ["HF_HUB_CACHE"] = previous

    def test_snapshot_revision_and_architecture_are_checked_before_model_import(self):
        load_contract(self.contract_path, self.snapshot)
        self.contract["revision"] = "c" * 40
        self.contract_path.write_text(json.dumps(self.contract))
        with self.assertRaisesRegex(ValueError, "immutable"):
            load_contract(self.contract_path, self.snapshot)
        self.contract["revision"] = self.revision
        self.contract_path.write_text(json.dumps(self.contract))
        (self.snapshot / "config.json").write_text('{"model_type":"other"}')
        with self.assertRaisesRegex(ValueError, "architecture"):
            load_contract(self.contract_path, self.snapshot)

    def test_quantized_recipe_is_not_compared_with_wrong_reference(self):
        self.contract["compression"] = "4bit"
        self.contract_path.write_text(json.dumps(self.contract))
        with self.assertRaisesRegex(ValueError, "unquantized"):
            load_contract(self.contract_path, self.snapshot)

    def test_missing_checkpoint_cannot_be_exported(self):
        (self.snapshot / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "weights"):
            load_contract(self.contract_path, self.snapshot)

    def test_reference_reads_input_ids_instead_of_backend_encoding_objects(self):
        class Tokenizer:
            def encode(self, *_args, **_kwargs):
                raise AssertionError("Backend encode returns Encoding, not token IDs")

            def __call__(self, *_args, **_kwargs):
                return {"input_ids": [10, 11]}

            def apply_chat_template(self, _messages, **kwargs):
                self.return_dict = kwargs["return_dict"]
                return {"input_ids": [20, 21]}

        tokenizer = Tokenizer()
        self.assertEqual(encode_case(tokenizer, {"text": "hello"}), [10, 11])
        self.assertEqual(encode_case(tokenizer, {"messages": []}), [20, 21])
        self.assertTrue(tokenizer.return_dict)

    def test_apple_toolchain_captures_compiler_bytes_as_well_as_version(self):
        compiler = self.root / "coreai-build"
        compiler.write_bytes(b"first compiler")
        responses = [str(compiler), "Xcode fixture", "fixture-sdk"]
        with mock.patch("pinned.subprocess.check_output", side_effect=responses):
            first = apple_toolchain()
        compiler.write_bytes(b"other compiler")
        with mock.patch("pinned.subprocess.check_output", side_effect=responses):
            second = apple_toolchain()
        self.assertNotEqual(first["coreai_build_sha256"], second["coreai_build_sha256"])
        self.assertEqual(first["xcode"], second["xcode"])

    def test_export_uses_pinned_private_cache_and_preserves_tokenizer_bytes(self):
        upstream = self.root / "upstream"
        package = upstream / "python/src/coreai_models"
        package.mkdir(parents=True)
        module = types.ModuleType("coreai_models")
        module.__file__ = str(package / "__init__.py")
        out = self.root / "exported"
        out.mkdir()
        calls = []

        def fake_export(name, run_name):
            calls.append(list(sys.argv))
            cache = Path(os.environ["HF_HUB_CACHE"]) / "models--Qwen--Qwen3-0.6B"
            self.assertEqual((cache / "refs/main").read_text(), self.revision)
            if "--dry-run" in sys.argv:
                return
            bundle = out / self.contract["output_name"]
            (bundle / "tokenizer").mkdir(parents=True)
            (bundle / "tokenizer/tokenizer.json").write_text("reserialized")
            (bundle / "tokenizer/extra.json").write_text("generated")
            (bundle / "metadata.json").write_text(json.dumps({"source": {"hf_model_id": self.contract["model_id"]}}))

        argv = ["export_llm", "--contract", str(self.contract_path), "--snapshot", str(self.snapshot),
                "--upstream", str(upstream), "--out", str(out)]
        previous_path = list(sys.path)
        try:
            with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules, {"coreai_models": module}),\
                    mock.patch.object(export_llm.subprocess, "check_output", return_value="b" * 40),\
                    mock.patch.object(export_llm.runpy, "run_module", side_effect=fake_export):
                export_llm.main()
        finally:
            sys.path[:] = previous_path
        self.assertIn("--compute-precision", calls[0])
        self.assertIn("--max-context-length", calls[0])
        self.assertNotIn("--overwrite", calls[0])
        self.assertIn("--dry-run", calls[0])
        self.assertNotIn("--dry-run", calls[1])
        bundle = out / self.contract["output_name"]
        self.assertEqual((bundle / "tokenizer/tokenizer.json").read_bytes(),
                         (self.snapshot / "tokenizer.json").read_bytes())
        self.assertFalse((bundle / "tokenizer/extra.json").exists())
        self.assertEqual(json.loads((bundle / "metadata.json").read_text())["source"]["revision"], self.revision)
        with mock.patch.object(sys, "argv", argv),\
                mock.patch.object(export_llm.subprocess, "check_output", return_value="b" * 40):
            with self.assertRaises(FileExistsError):
                export_llm.main()


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Numerical adapter tests require the model environment's NumPy")
class NumericalTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.criteria = {"near_tie_probability_margin": 0.01, "max_probability_error": 0.03}

    def test_identical_outputs_pass_and_wrong_pairing_fails(self):
        x = self.np.array([[5, 0, -1], [0, 4, -2]], dtype=self.np.float32)
        self.assertTrue(compare(x, x, self.criteria)["passed"])
        mismatch = compare(x, x[::-1], self.criteria)
        self.assertFalse(mismatch["passed"])
        self.assertEqual(mismatch["significant_mismatches"], 2)

    def test_near_tie_is_recorded_without_hiding_probability_error(self):
        x = self.np.array([[1.0, 1.0001, -1]], dtype=self.np.float32)
        y = self.np.array([[1.0001, 1.0, -1]], dtype=self.np.float32)
        result = compare(x, y, self.criteria)
        self.assertTrue(result["passed"])
        self.assertEqual(result["near_tie_mismatches"], 1)
        y[0, 0] = 10
        self.assertFalse(compare(x, y, self.criteria)["passed"])

    def test_nan_and_wrong_shape_are_rejected(self):
        x = self.np.zeros((2, 3), dtype=self.np.float32)
        with self.assertRaises(ValueError):
            compare(x, x[:1], self.criteria)
        x[0, 0] = self.np.nan
        with self.assertRaises(ValueError):
            compare(x, x, self.criteria)

    def test_third_ranked_winner_is_not_excused_by_unrelated_top_two_tie(self):
        a = self.np.log(self.np.array([[.34, .339, .321]], dtype=self.np.float32))
        b = self.np.log(self.np.array([[.325, .33, .345]], dtype=self.np.float32)).astype(self.np.float16)
        result = compare(a, b.astype(self.np.float32), self.criteria)
        self.assertFalse(result["passed"])
        self.assertEqual(result["near_tie_mismatches"], 0)
        self.assertEqual(result["significant_mismatches"], 1)

    def test_swift_float16_decode_checks_token_alignment_and_width(self):
        x = self.np.array([[1, 2, 3], [4, 5, 6]], dtype="<f2")
        case = {"tokens": [10, 11, 12], "shape": [2, 3]}
        data = {"context_tokens": [10], "continuation_tokens": [11, 12],
                "positions": [{"index": i, "target_token": 11 + i,
                               "logits_base64": base64.b64encode(row.tobytes()).decode()}
                              for i, row in enumerate(x)]}
        self.np.testing.assert_array_equal(decode_swift(data, case), x.astype(self.np.float32))
        data["positions"][0]["target_token"] = 99
        with self.assertRaisesRegex(ValueError, "alignment"):
            decode_swift(data, case)
        data["positions"][0]["target_token"] = 11
        data["positions"][0]["logits_base64"] = base64.b64encode(b"short").decode()
        with self.assertRaisesRegex(ValueError, "Float16"):
            decode_swift(data, case)

    def test_verification_cli_uses_registered_engine_and_scores_all_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference, export, out = root / "reference", root / "export", root / "result"
            reference.mkdir()
            bundle = export / "bundle"
            (bundle / "tokenizer").mkdir(parents=True)
            contract = {"model_id": "fixture/model", "revision": "a" * 40}
            (bundle / "metadata.json").write_text(json.dumps({"source": {"hf_model_id": contract["model_id"],
                                                                       "revision": contract["revision"]}}))
            (export / "provenance.json").write_text(json.dumps({"contract": contract, "tokenizer_sha256": {}}))
            case = {"id": "fixture", "tokens": [10, 11, 12], "shape": [2, 3]}
            (reference / "reference.json").write_text(json.dumps({"contract": contract, "cases": [case]}))
            (reference / "fixture.tokens.json").write_text(json.dumps({"tokens": case["tokens"]}))
            self.np.save(reference / "fixture.npy", self.np.array([[1, 2, 3], [4, 5, 6]], dtype=self.np.float32))
            criteria = root / "criteria.json"
            criteria.write_text(json.dumps(self.criteria))
            binary = root / "runner"
            binary.write_text("#!" + sys.executable + "\n" + """
import argparse,base64,json,pathlib,struct
p=argparse.ArgumentParser()
for k in ['model','raw-tokens','save-logits','save-logits-length','sampling-strategy','warmup']:
 p.add_argument('--'+k,required=True)
p.add_argument('--inference-engine-variant',choices=['coreai-sequential'],required=True)
a=p.parse_args()
ids=json.loads(pathlib.Path(a.raw_tokens).read_text())['tokens']
positions=[{'index':i,'target_token':ids[i+1],'logits_base64':base64.b64encode(struct.pack('<3e',*row)).decode()}
 for i,row in enumerate([[1,2,3],[4,5,6]])]
pathlib.Path(a.save_logits).write_text(json.dumps({'context_tokens':ids[:1],'continuation_tokens':ids[1:],
 'positions':positions}))
""")
            binary.chmod(0o755)
            argv = ["verify", "--reference", str(reference), "--bundle", str(bundle),
                    "--runner", str(binary), "--criteria", str(criteria), "--out", str(out)]
            with mock.patch.object(sys, "argv", argv):
                verify_llm.main()
            result = json.loads((out / "comparison.json").read_text())
            self.assertTrue(result["passed"])
            self.assertEqual(result["cases"][0]["positions"], 2)


if __name__ == "__main__":
    unittest.main()
