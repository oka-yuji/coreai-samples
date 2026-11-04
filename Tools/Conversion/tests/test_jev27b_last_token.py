import copy
import dataclasses
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from Tools.Conversion.adapters.jev27b.decision import DECISION, LAST_OUTPUT, OUTPUT, listing, sha256, validate, write_sums
from Tools.Conversion.adapters.jev27b.last_token_bundle import finish, source_identity
from Tools.Conversion.recipe import load_recipe
from Tools.Conversion.storage import read_json, write_json
from Tools.Conversion.tests.test_jev27b_decision import ASSET, bundle

ROOT = Path(__file__).resolve().parents[3]
CONVERTER = importlib.util.find_spec("coreai_models") is not None


def save(root, decision):
    decision["files"] = listing(root)
    write_json(root / DECISION, decision)
    write_sums(root)


def source_bundle(root):
    decision = bundle(root)
    write_json(root / "metadata.json", {
        "assets": {"main": ASSET}, "modifications": ["Original changes."],
        "distribution": {"repository": "owner/source", "evaluated_decision_sha256": "old",
                         "changes_since_evaluation": "The asset is unchanged."}})
    write_json(root / "calibration-refit.json", {"schema": "fixture", "temperature": 1.1})
    digest = sha256(root / "calibration-refit.json")
    decision["temperatures"] = {"per_kind": {"noul": 1.1, "choice": 1.2, "score": 1.3}, "source": "refit",
                                "calibration_sha256": digest, "calibration_id": "jev27b-refit/" + digest[:16]}
    save(root, decision)
    return decision


class LastTokenBundleTests(unittest.TestCase):
    def test_both_declared_outputs_are_valid_but_an_unknown_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            decision = bundle(root)
            for name in (OUTPUT, LAST_OUTPUT):
                decision["backbone"]["output"] = name
                self.assertEqual(validate(decision), [])
            decision["backbone"]["output"] = "unknown"
            self.assertTrue(validate(decision))

    def test_the_candidate_keeps_calibration_and_file_bytes_without_claiming_the_old_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source", Path(folder) / "target"
            original = source_bundle(source)
            generated = bundle(target)
            (target / ASSET).rename(target / "last.aimodel")
            (target / "last.aimodel/main.mlirb").write_bytes(b"new graph")
            generated["backbone"]["asset"] = "last.aimodel"
            write_json(target / "metadata.json", {"assets": {"main": "last.aimodel"}, "source": {"revision": "4" * 40}})
            save(target, generated)
            identity = source_identity(source, source / ASSET, "4" * 40)
            result = finish(target, source, identity, "last.aimodel", 5120)
            candidate = read_json(target / DECISION)
            self.assertEqual(validate(candidate, target), [])
            self.assertEqual(candidate["backbone"]["output"], LAST_OUTPUT)
            self.assertEqual(candidate["temperatures"], original["temperatures"])
            for path in ("head.safetensors", "tokenizer/tokenizer.json", "calibration-refit.json", "LICENSE"):
                self.assertEqual((source / path).read_bytes(), (target / path).read_bytes())
                self.assertEqual(result["preserved_file_sha256"][path], sha256(source / path))
            metadata = read_json(target / "metadata.json")
            self.assertEqual(metadata["assets"], {"main": "last.aimodel"})
            self.assertEqual(metadata["distribution"]["status"], "unpublished-candidate")
            self.assertNotIn("repository", metadata["distribution"])
            self.assertNotIn("evaluated_decision_sha256", metadata["distribution"])
            self.assertEqual(metadata["distribution"]["derived_from"]["decision_sha256"], identity["decision_sha256"])
            self.assertEqual(read_json(source / DECISION), original)

    def test_a_different_baseline_asset_or_merge_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_bundle(root)
            with self.assertRaises(ValueError):
                source_identity(root, root / ASSET, "9" * 40)
            other = root.parent / (root.name + "-other")
            try:
                other.mkdir()
                (other / "main.mlirb").write_bytes(b"other")
                with self.assertRaises(ValueError):
                    source_identity(root, other, "4" * 40)
            finally:
                if (other / "main.mlirb").exists():
                    (other / "main.mlirb").unlink()
                if other.exists():
                    other.rmdir()

    def test_a_source_change_during_export_is_rejected_before_copying(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source", Path(folder) / "target"
            source_bundle(source)
            bundle(target)
            identity = source_identity(source, source / ASSET, "4" * 40)
            (source / "head.safetensors").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                finish(target, source, identity, ASSET, 5120)
            self.assertEqual((target / "head.safetensors").read_bytes(), b"head")

    def test_a_changed_head_contract_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source", Path(folder) / "target"
            source_bundle(source)
            generated = bundle(target)
            identity = source_identity(source, source / ASSET, "4" * 40)
            changed = copy.deepcopy(generated)
            changed["head"]["sha256"] = "9" * 64
            write_json(target / DECISION, changed)
            with self.assertRaises(ValueError):
                finish(target, source, identity, ASSET, 5120)

    def test_the_standalone_recipe_reuses_frozen_inputs_without_repeating_merge_and_references(self):
        bindings = {key: "/unused" for key in ("base", "hf_cache", "python", "snapshot", "upstream",
                                               "verified_merge", "baseline_gdnfp32", "published_bundle")}
        recipe = load_recipe(ROOT / "Tools/Conversion/recipes/jev27b/last-token.json", bindings, ROOT)
        self.assertEqual([step["id"] for step in recipe["steps"]], ["export_last_token"])
        step = recipe["steps"][0]
        self.assertIn("export-last-token", step["command"])
        self.assertIn("$published_bundle", step["inputs"])
        self.assertIn("$verified_merge/checkpoint", step["inputs"])
        self.assertTrue(step["heavy"])
        self.assertEqual(step["outputs"][0], recipe["variables"]["bundle_last_token"])


@unittest.skipUnless(CONVERTER, "The JEV-27B converter environment is required")
class LastTokenBackboneTests(unittest.TestCase):
    def test_the_registered_variant_keeps_parameters_and_wraps_every_linear_attention_layer(self):
        from Tools.Conversion.adapters.jev27b import backbone, backbone_gdnfp32, backbone_last_token
        from Tools.Conversion.tests.test_jev27b_backbone import composite_config

        @dataclasses.dataclass(frozen=True)
        class Entry:
            macos_class: object
            hf_config_attr: str = "text_config"
            hf_state_dict_prefix: str = "model.language_model."

        original = Entry(object)
        pipeline = SimpleNamespace(get_model_entry=lambda _: original)
        entry = backbone_last_token.install(pipeline)
        self.assertIs(entry.macos_class, backbone_last_token.TypedDecisionBackboneLastToken)
        self.assertEqual((entry.hf_config_attr, entry.hf_state_dict_prefix),
                         (original.hf_config_attr, original.hf_state_dict_prefix))
        self.assertEqual(pipeline.get_model_entry(backbone.COMPOSITE), entry)
        before = backbone_gdnfp32.TypedDecisionBackboneGdnFp32(config=composite_config(), model_device="meta")
        after = entry.macos_class(config=composite_config(), model_device="meta")
        describe = lambda model: [(name, tuple(value.shape), value.dtype) for name, value in model.named_parameters()]
        self.assertEqual(describe(before), describe(after))
        self.assertEqual(backbone_gdnfp32.wrapped_count(after), 3)
        self.assertFalse(hasattr(after, "lm_head"))

    def test_the_real_forward_selects_before_the_float16_cast_without_changing_the_other_states(self):
        import torch

        from Tools.Conversion.adapters.jev27b.backbone_last_token import TypedDecisionBackboneLastToken
        from Tools.Conversion.adapters.jev27b.decision import STATES

        self.assertEqual(TypedDecisionBackboneLastToken.export_output_names(), {"main": (LAST_OUTPUT,)})
        self.assertEqual(TypedDecisionBackboneLastToken.export_state_names(), {"main": STATES})
        self.assertFalse(TypedDecisionBackboneLastToken.exports_prefill_graph)
        for tokens in (1, 7, 8192):
            hidden = torch.arange(tokens * 4, dtype=torch.float32).reshape(1, tokens, 4).to(torch.bfloat16)
            model = SimpleNamespace(model=lambda *_: hidden)
            inputs = torch.zeros((1, tokens), dtype=torch.int32)
            state = torch.zeros((1,), dtype=torch.bfloat16)
            output = TypedDecisionBackboneLastToken.forward(model, inputs, inputs, state, state, state, state)
            self.assertEqual(list(output.shape), [1, 1, 4])
            self.assertEqual(output.dtype, torch.float16)
            self.assertTrue(torch.equal(output, hidden[:, -1:, :].to(torch.float16)))
            self.assertTrue(torch.equal(state, torch.zeros_like(state)))
