import copy
import math
import unittest
from unittest.mock import patch

from Tools.Conversion.adapters.chronos2 import artifacts, reference
from Tools.Conversion.storage import file_hash, read_json, write_json
from Tools.Conversion.tests.test_chronos2_support import ChronosFiles, RECIPE


class SourceTests(ChronosFiles, unittest.TestCase):
    def test_fixed_contract_and_all_nullable_fixtures_validate(self):
        contract = artifacts.load_contract(RECIPE / "model.json")
        self.assertEqual(contract["abi"], artifacts.ABI)
        self.assertEqual(len(self.fixtures["cases"]), 18)
        prepared = artifacts.nullable_context([None, 3.0, None])
        self.assertEqual(len(prepared), 16)
        self.assertTrue(all(math.isnan(value) for value in prepared[:14]))
        self.assertEqual(prepared[-2], 3.0)
        self.assertTrue(math.isnan(prepared[-1]))

    def test_contract_identity_precision_and_abi_tampering_are_rejected(self):
        mutations = [
            lambda c: c.update(model_id="amazon/other"), lambda c: c.update(revision="0" * 40),
            lambda c: c.update(precision="float16"), lambda c: c.update(schema_version=True),
            lambda c: c["library"].update(version="2.3.1"),
            lambda c: c["abi"]["inputs"][0].update(max_length=8193),
            lambda c: c["abi"]["outputs"][0].update(shape=[1, 16, 21]),
            lambda c: c["abi"].update(state_names=["state"]), lambda c: c["quantiles"].reverse(),
            lambda c: c["files"].update({"../config.json": c["files"].pop("config.json")}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                contract = copy.deepcopy(self.contract)
                mutate(contract)
                write_json(self.model_path, contract)
                with self.assertRaises(ValueError):
                    artifacts.load_contract(self.model_path)

    def test_each_source_file_is_hashed_and_missing_files_fail(self):
        self.assertEqual(artifacts.validate_source(self.snapshot, self.contract), self.snapshot)
        for name in self.contract["files"]:
            path = self.snapshot / name
            original = path.read_bytes()
            for changed in (original + b"x", bytes([original[0] ^ 1]) + original[1:]):
                with self.subTest(name=name, size=len(changed)):
                    path.write_bytes(changed)
                    with self.assertRaises(ValueError):
                        artifacts.validate_source(self.snapshot, self.contract)
            path.unlink()
            with self.assertRaises(ValueError):
                artifacts.validate_source(self.snapshot, self.contract)
            path.write_bytes(original)

    def test_snapshot_directory_must_match_revision(self):
        wrong = self.root / "main"
        self.snapshot.rename(wrong)
        with self.assertRaises(ValueError):
            artifacts.validate_source(wrong, self.contract)

    def test_config_contract_is_checked_after_hash_validation(self):
        changes = [("architectures", ["OtherModel"]), ("num_heads", 1), ("torch_dtype", "float16"),
                   ("reg_token_id", 0), ("rope_theta", 1.0)]
        for name, value in changes:
            with self.subTest(field=name):
                config = copy.deepcopy(self.config)
                config[name] = value
                write_json(self.snapshot / "config.json", config)
                self.contract["files"]["config.json"] = file_hash(self.snapshot / "config.json")
                with self.assertRaises(ValueError):
                    artifacts.validate_source(self.snapshot, self.contract)
        for name, value in (("input_patch_stride", 8), ("output_patch_size", 32), ("use_arcsinh", False),
                            ("quantiles", list(reversed(artifacts.QUANTILES))), ("time_encoding_scale", 16)):
            with self.subTest(field=name):
                config = copy.deepcopy(self.config)
                config["chronos_config"][name] = value
                with self.assertRaises(ValueError):
                    artifacts.validate_config(config)

    def test_tampered_weights_fail_before_loading_the_reference_model(self):
        (self.snapshot / "model.safetensors").write_bytes(b"altered")
        with patch.object(reference, "load_official") as loader, self.assertRaises(ValueError):
            reference.reference(self.model_path, self.fixture_path, self.snapshot, self.ready, self.root / "out")
        loader.assert_not_called()

    def test_fixture_ids_tolerance_and_missing_value_rules_are_fixed(self):
        mutations = [lambda f: f["cases"].pop(), lambda f: f["cases"].append(f["cases"][0]),
                     lambda f: f["cases"].reverse(), lambda f: f["cases"][1].update(id=f["cases"][0]["id"]),
                     lambda f: f.update(atol=0.1), lambda f: f["cases"][0].update(context=[None]),
                     lambda f: f["cases"][0].update(context=[True])]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                fixtures = copy.deepcopy(self.fixtures)
                mutate(fixtures)
                write_json(self.fixture_path, fixtures)
                with self.assertRaises(ValueError):
                    artifacts.load_fixtures(self.fixture_path)

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_contexts(self):
        for text in ('{"cases":[],"cases":[]}', '{"context":[NaN]}', '{"context":[Infinity]}'):
            self.fixture_path.write_text(text)
            with self.assertRaises(ValueError):
                artifacts.load_fixtures(self.fixture_path)


class VectorTests(ChronosFiles, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.actuals = artifacts.vector_document("actuals", self.identity, self.fixtures, self.values)

    def compare(self, actuals):
        return artifacts.compare_documents(self.reference, actuals, self.identity, self.fixtures)

    def test_all_6048_values_are_compared_in_pinned_order(self):
        result = self.compare(self.actuals)
        self.assertTrue(result["passed"])
        self.assertEqual((result["count"], result["elements"]), (18, 6048))
        self.assertEqual([case["id"] for case in result["cases"]], list(artifacts.CASE_IDS))
        self.actuals["cases"][-1]["quantiles"][0][-1][-1] += 0.1
        result = self.compare(self.actuals)
        self.assertFalse(result["passed"])
        self.assertFalse(result["cases"][-1]["passed"])
        self.assertEqual(result["elements"], 6048)

    def test_missing_extra_duplicate_reordered_and_flag_only_cases_fail(self):
        mutations = [lambda c: c.pop(), lambda c: c.append(copy.deepcopy(c[0])), lambda c: c.reverse(),
                     lambda c: c[1].update(id=c[0]["id"]),
                     lambda c: c[0].update(quantiles=None), lambda c: c[0].pop("quantiles"),
                     lambda c: c[0].update(passed=True)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                document = copy.deepcopy(self.actuals)
                mutate(document["cases"])
                with self.assertRaises(ValueError):
                    self.compare(document)

    def test_shape_finite_and_precision_validation_is_strict(self):
        for value in (math.nan, math.inf, -math.inf, None, True, "1", 1e39):
            with self.subTest(value=value):
                document = copy.deepcopy(self.actuals)
                document["cases"][-1]["quantiles"][0][-1][-1] = value
                with self.assertRaises(ValueError):
                    self.compare(document)
        for mutate in (lambda d: d.update(precision="float16"),
                       lambda d: d["cases"][-1]["quantiles"][0][-1].pop(),
                       lambda d: d["cases"][-1]["quantiles"].append(d["cases"][-1]["quantiles"][0])):
            document = copy.deepcopy(self.actuals)
            mutate(document)
            with self.assertRaises(ValueError):
                self.compare(document)

    def test_context_quantile_identity_and_threshold_changes_are_rejected(self):
        for mutate in (lambda d: d["quantiles"].reverse(), lambda d: d.update(rtol=1.0),
                       lambda d: d["identity"]["fixtures"].update(sha256="0" * 64),
                       lambda d: d["cases"][0]["context"].append(0.0)):
            document = copy.deepcopy(self.actuals)
            mutate(document)
            with self.assertRaises(ValueError):
                self.compare(document)
        self.actuals["cases"][0]["quantiles"][0].reverse()
        self.assertFalse(self.compare(self.actuals)["passed"])

    def test_reference_vectors_are_validated_independently(self):
        self.reference["cases"].pop()
        with self.assertRaises(ValueError):
            self.compare(self.actuals)


class PathTests(ChronosFiles, unittest.TestCase):
    def test_inventory_paths_cannot_escape_or_alias(self):
        for name in ("", ".", "..", "../asset", "/asset", "a//b", "a/./b", "a/../b", "a\\b", "C:/b", "a\x00b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                artifacts.relative_path(name)
        self.assertEqual(artifacts.relative_path("Model.aimodel/functions/main.mlirb").as_posix(),
                         "Model.aimodel/functions/main.mlirb")

    def test_existing_outputs_and_symlink_ancestors_are_rejected(self):
        out = artifacts.new_output(self.root / "out")
        artifacts.write_new(out / "result.json", {"passed": True})
        with self.assertRaises(ValueError):
            artifacts.new_output(out)
        with self.assertRaises(ValueError):
            artifacts.write_new(out / "result.json", {})
        link = self.root / "link"
        link.symlink_to(out, target_is_directory=True)
        with self.assertRaises(ValueError):
            artifacts.new_output(link / "child")
        (out / "linked.json").symlink_to(out / "result.json")
        with self.assertRaises(ValueError):
            artifacts.inventory(out)
        self.assertEqual(read_json(out / "result.json"), {"passed": True})
