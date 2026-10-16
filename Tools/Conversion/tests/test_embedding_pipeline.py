import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Tools.Conversion.adapters.embeddinggemma2.prepare import check_environment, environment_inputs, validate_source
from Tools.Conversion.adapters.embeddinggemma2.verify import comparison
from Tools.Conversion.recipe import load_recipe
from Tools.Embedding.package_bundle import REVISION, package


class EmbeddingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference"
        self.reference.mkdir()
        self.vector = np.eye(1, 768, dtype=np.float32)
        self.specification = {"fixtures": [{"id": f"case_{i}"} for i in range(20)],
                              "low_precision_gate": {"cosine_min": 0.999, "max_abs_max": 0.01}}
        self.records = []
        for fixture in self.specification["fixtures"]:
            folder = self.reference / fixture["id"]
            folder.mkdir()
            np.save(folder / "reference.npy", self.vector, allow_pickle=False)
            self.records.append({"id": fixture["id"], "tokensMatch": True, "passed": True,
                                 "vector": self.vector[0].tolist()})

    def tearDown(self):
        self.temp.cleanup()

    def test_native_vectors_are_recomputed_instead_of_trusting_passed_flags(self):
        self.assertTrue(comparison(self.specification, self.reference, self.records)["passed"])
        self.records[0]["vector"] = np.roll(self.vector, 1)[0].tolist()
        self.assertFalse(comparison(self.specification, self.reference, self.records)["passed"])

    def test_native_case_identity_and_count_cannot_be_omitted_duplicated_or_reordered(self):
        for records in ([], self.records[:-1], self.records[::-1], [self.records[0]] * 20):
            with self.assertRaises(ValueError):
                comparison(self.specification, self.reference, records)

    def test_weaker_gates_and_invalid_native_vectors_are_rejected(self):
        weakened = copy.deepcopy(self.specification)
        weakened["low_precision_gate"]["cosine_min"] = 0.9
        with self.assertRaises(ValueError):
            comparison(weakened, self.reference, self.records)
        for value in (True, "1", None):
            records = copy.deepcopy(self.records)
            records[0]["vector"][0] = value
            with self.assertRaises(ValueError):
                comparison(self.specification, self.reference, records)
        for key in ("tokensMatch", "passed"):
            records = copy.deepcopy(self.records)
            records[0][key] = False
            with self.assertRaises(ValueError):
                comparison(self.specification, self.reference, records)

    def test_pinned_source_bytes_and_shape_contract_are_checked(self):
        snapshot = self.root / ("a" * 40)
        snapshot.mkdir()
        data = json.dumps({"model_type": "embedding_gemma2"}).encode()
        (snapshot / "config.json").write_bytes(data)
        contract = {"model_id": "google/embeddinggemma-2", "model_type": "embedding_gemma2",
                    "revision": snapshot.name, "text_length": 128, "vision_tokens": 70,
                    "precision": "fp16_safe", "vector_dimensions": 768,
                    "source_files": {"config.json": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}
        self.assertEqual(validate_source(snapshot, contract), snapshot.resolve())
        contract["text_length"] = 256
        with self.assertRaises(ValueError):
            validate_source(snapshot, contract)
        contract["text_length"] = 128
        (snapshot / "config.json").write_text("{}")
        with self.assertRaises(ValueError):
            validate_source(snapshot, contract)

    def test_packaging_refuses_mixed_or_unverified_encoders_before_creating_output(self):
        self.specification["model_revision"] = REVISION
        (self.reference / "fixtures.json").write_text(json.dumps(self.specification))
        text, vision = self.root / "text", self.root / "vision"
        for kind, path in (("text", text), ("vision", vision)):
            path.mkdir()
            (path / "contract.json").write_text(json.dumps(
                {"kind": kind, "precision": "fp16_safe", "complete": True, "fixtures_sha256": "wrong"}))
        output = self.root / "bundle"
        with self.assertRaises(ValueError):
            package(self.root, text, vision, self.reference, output)
        self.assertFalse(output.exists())

    def test_packaging_requires_explicit_revision_even_with_matching_fixture_hash(self):
        self.specification["model_revision"] = REVISION
        data = json.dumps(self.specification).encode()
        (self.reference / "fixtures.json").write_bytes(data)
        text, vision = self.root / "text", self.root / "vision"
        for kind, path in (("text", text), ("vision", vision)):
            path.mkdir()
            (path / "contract.json").write_text(json.dumps(
                {"kind": kind, "precision": "fp16_safe", "complete": True,
                 "fixtures_sha256": hashlib.sha256(data).hexdigest()}))
        with self.assertRaisesRegex(ValueError, "identities differ"):
            package(self.root, text, vision, self.reference, self.root / "bundle")
        self.assertFalse((self.root / "bundle").exists())

    def test_environment_requires_setup_after_pin_or_installed_version_changes(self):
        (self.root / "pyproject.toml").write_text('[project]\ndependencies = ["torch==2.9.0"]\n')
        (self.root / "uv.lock").write_text("version = 1\n")
        stamp = environment_inputs(self.root)
        with patch("importlib.metadata.version", return_value="2.9.0"):
            check_environment(self.root, stamp)
        with patch("importlib.metadata.version", return_value="2.10.0"):
            with self.assertRaisesRegex(ValueError, "Installed dependency differs"):
                check_environment(self.root, stamp)
        (self.root / "uv.lock").write_text("version = 2\n")
        with self.assertRaisesRegex(ValueError, "requirements changed"):
            check_environment(self.root, stamp)

    def test_recipe_declares_the_model_jobs_and_native_verification_gate(self):
        repo = Path(__file__).resolve().parents[3]
        recipe = load_recipe(repo / "Tools/Conversion/recipes/embeddinggemma2/recipe.json",
                             {"python": "/python", "snapshot": "/snapshot", "runner": "/runner", "native_build": "/build"}, repo)
        stages = {step["id"]: step for step in recipe["steps"]}
        self.assertEqual(set(stages), {"prepare", "reference", "text_export", "vision_export", "bundle", "verify"})
        self.assertFalse(stages["prepare"]["heavy"])
        self.assertTrue(all(stages[name]["heavy"] for name in stages if name != "prepare"))
        self.assertEqual(stages["verify"]["checks"], [
            {"file": "comparison.json", "key": ["passed"], "equals": True},
            {"file": "comparison.json", "key": ["count"], "equals": 20}])


if __name__ == "__main__":
    unittest.main()
