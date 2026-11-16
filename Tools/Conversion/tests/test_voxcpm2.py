import copy
import importlib.util
import json
import math
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from Tools.Conversion.adapters.voxcpm2.checks import Checks, aggregate, classify_fuse, fsq_summary, fuse_candidates, verdict
from Tools.Conversion.adapters.voxcpm2.runtime import validate_inputs
from Tools.Conversion.adapters.voxcpm2.source import compare, expected_config, load_contract, validate_source, with_capacity
from Tools.Conversion.recipe import load_recipe
from Tools.Conversion.storage import file_hash

RECIPE = Path(__file__).resolve().parents[1] / "recipes" / "voxcpm2"
CONTRACT = load_contract(RECIPE / "model.json")


def configuration(contract):
    config = {}
    for keys, value in expected_config(contract).items():
        node = config
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return config


class Snapshot:
    def __init__(self, directory, config=None):
        self.root = Path(directory) / CONTRACT["revision"]
        self.root.mkdir()
        (self.root / "config.json").write_text(json.dumps(config or configuration(CONTRACT)))
        self.contract = {**CONTRACT, "files": {"config.json": file_hash(self.root / "config.json")}}


class VoxCPM2SourceTests(unittest.TestCase):
    def test_pinned_snapshot_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Snapshot(directory)
            self.assertEqual(validate_source(snapshot.root, snapshot.contract), snapshot.root.resolve())

    def test_changed_source_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Snapshot(directory)
            (snapshot.root / "config.json").write_text(json.dumps(configuration(CONTRACT)) + " ")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_source(snapshot.root, snapshot.contract)

    def test_unmanaged_files_are_rejected_but_download_metadata_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Snapshot(directory)
            metadata = snapshot.root / ".cache" / "huggingface" / "download" / "config.json.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("download metadata")
            validate_source(snapshot.root, snapshot.contract)
            (snapshot.root / "audiovae.safetensors").write_bytes(b"replacement weights")
            with self.assertRaisesRegex(ValueError, "Unmanaged source file"):
                validate_source(snapshot.root, snapshot.contract)

    def test_links_escaping_the_snapshot_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Snapshot(directory)
            outside = Path(directory) / "outside.json"
            outside.write_text("{}")
            (snapshot.root / "extra.json").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "escapes"):
                validate_source(snapshot.root, snapshot.contract)

    def test_another_revision_or_architecture_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Snapshot(directory)
            with self.assertRaisesRegex(ValueError, "pinned source"):
                validate_source(snapshot.root, {**snapshot.contract, "revision": "b" * 40})
        config = configuration(CONTRACT)
        config["lm_config"]["use_mup"] = True
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Snapshot(directory, config)
            with self.assertRaisesRegex(ValueError, "lm_config.use_mup"):
                validate_source(snapshot.root, snapshot.contract)


class VoxCPM2ContractTests(unittest.TestCase):
    def test_non_finite_candidates_fail_and_non_finite_references_raise(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                result = compare([1.0, 2.0], [1.0, value], 0.001)
                self.assertFalse(result["passed"])
                self.assertEqual(result["nonfinite"], 1)
                self.assertFalse(verdict([{"gate": True, "case": "embed", **result}], CONTRACT["criteria"]))
        with self.assertRaises(ValueError):
            compare([np.nan], [1.0], 0.001)
        with self.assertRaises(ValueError):
            compare([1.0, 2.0], [1.0], 0.001)
        self.assertTrue(compare([2.0, -4.0], [2.0, -4.004], 0.001)["passed"])
        self.assertFalse(compare([2.0, -4.0], [2.0, -4.005], 0.001)["passed"])

    def test_inputs_follow_the_contract(self):
        hidden = np.zeros((1, 2048), np.float32)
        position = np.array([1023], np.int32)
        self.assertEqual(set(validate_inputs(CONTRACT, "tslm", "step", {"embed": hidden, "position": position})), {"embed", "position"})
        cases = [{"embed": hidden.astype(np.float16), "position": position},
                 {"embed": np.zeros((1, 1024), np.float32), "position": position},
                 {"embed": np.full((1, 2048), np.nan, np.float32), "position": position},
                 {"embed": hidden, "position": np.array([1024], np.int32)},
                 {"embed": hidden, "position": np.array([-1], np.int32)},
                 {"embed": hidden, "position": position.astype(np.int64)},
                 {"embed": hidden}]
        for values in cases:
            with self.subTest(values={key: (value.dtype, value.shape) for key, value in values.items()}), self.assertRaises(ValueError):
                validate_inputs(CONTRACT, "tslm", "step", values)
        embeds = np.zeros((1, 145, 2048), np.float32)
        for length in (0, 146):
            with self.subTest(length=length), self.assertRaises(ValueError):
                validate_inputs(CONTRACT, "tslm", "prefill", {"embeds": embeds, "length": np.array([length], np.int32)})
        with self.assertRaises(ValueError):
            validate_inputs(CONTRACT, "tslm", "embed", {"ids": np.full((1, 145), 73448, np.int32)})

    def test_half_precision_interface_keeps_integer_inputs(self):
        values = {"embed": np.zeros((1, 2048), np.float16), "position": np.array([0], np.int32)}
        validate_inputs(CONTRACT, "ralm", "step", values, "fp16")
        with self.assertRaises(ValueError):
            validate_inputs(CONTRACT, "ralm", "step", {**values, "embed": values["embed"].astype(np.float32)}, "fp16")

    def test_capacity_override_changes_only_the_cache_states(self):
        changed = with_capacity(CONTRACT, 2048)
        self.assertEqual(changed["assets"]["tslm"]["functions"]["step"]["states"][0], ["k_cache", "float32", [28, 2, 2048, 128]])
        self.assertEqual(CONTRACT["assets"]["tslm"]["functions"]["step"]["states"][0][2][2], 1024)
        self.assertEqual(changed["assets"]["vae"], CONTRACT["assets"]["vae"])
        validate_inputs(changed, "tslm", "step", {"embed": np.zeros((1, 2048), np.float32), "position": np.array([2047], np.int32)})

    def test_capacity_holds_the_longest_supported_text(self):
        execution = CONTRACT["execution"]
        tokens = execution["max_text_tokens"]
        self.assertEqual(execution["prefill_length"], tokens + 1)
        generation = CONTRACT["generation"]
        patches = min(int(tokens * generation["max_patches_ratio"] + generation["max_patches_offset"]), generation["max_len"])
        self.assertLessEqual(tokens + 1 + patches, execution["capacity"])

    def test_bundled_fixtures_follow_the_generation_contract(self):
        fixtures = json.loads((RECIPE / "fixtures.json").read_text(encoding="utf-8"))
        for key in ("inference_timesteps", "cfg_value", "min_len", "max_len"):
            self.assertEqual(fixtures[key], CONTRACT["generation"][key], key)

    def test_recorded_rows_do_not_gate_the_result(self):
        recorded = {"gate": False, "passed": False, "nonfinite": 0, "case": "chunk"}
        self.assertTrue(verdict([recorded, {"gate": True, "passed": True, "nonfinite": 0, "case": "step"}], CONTRACT["criteria"]))
        self.assertFalse(verdict([recorded], CONTRACT["criteria"]))

    def test_recipe_declares_sources_and_checks(self):
        bindings = {"python": "/usr/bin/python3", "snapshot": "/snapshot", "upstream": "/upstream"}
        recipe = load_recipe(RECIPE / "recipe.json", bindings, Path(__file__).resolve().parents[3])
        stages = {stage["id"]: stage for stage in recipe["steps"]}
        self.assertEqual(stages["reference"]["role"], "reference")
        for stage in stages.values():
            self.assertTrue(stage["heavy"])
            self.assertTrue(stage["checks"])
            self.assertIn("$recipe_dir/model.json", stage["inputs"])
        exports = [stage for name, stage in stages.items() if name.endswith("_export")]
        self.assertEqual(len(exports), 8)
        adapters = Path(__file__).resolve().parents[1] / "adapters" / "voxcpm2"

        def imports(name):
            text = (adapters / name).read_text(encoding="utf-8")
            return {(module or member) + ".py" for module, member in re.findall(r"^from \.(\w*) import (\w+)", text, re.M)}

        for stage in stages.values():
            self.assertNotIn("$adapters", stage["inputs"])
            declared = {item.split("/")[-1] for item in stage["inputs"] if item.startswith("$adapters/")}
            self.assertIn(stage["command"][2].rsplit(".", 1)[-1] + ".py", declared)
            for name in declared:
                self.assertLessEqual(imports(name), declared)
        self.assertIn("host.py", imports("synthesis.py"))
        for stage in exports:
            self.assertEqual(stage["depends"], [])
            self.assertNotIn("$recipe_dir/fixtures.json", stage["inputs"])
            self.assertTrue({"$snapshot", "$upstream/src/voxcpm", "$recipe_dir/requirements.lock"} <= set(stage["inputs"]))
        self.assertNotIn("$snapshot", stages["synthesis_compare"]["inputs"])
        self.assertTrue({"$snapshot/tokenizer.json", "$recipe_dir/synthesis.json"} <= set(stages["synthesis_compare"]["inputs"]))
        self.assertTrue({"$snapshot", "$upstream/src/voxcpm", "$recipe_dir/synthesis.json"} <= set(stages["synthesis_check"]["inputs"]))
        self.assertIn("$recipe_dir/fixtures.json", stages["reference"]["inputs"])
        for name in ("wrapper_check", "verify_fp32", "synthesis_check"):
            self.assertIn(["passed"], [check["key"] for check in stages[name]["checks"]])
        for name in ("verify_fp16", "synthesis_compare"):
            self.assertNotIn(["passed"], [check["key"] for check in stages[name]["checks"]])


class VoxCPM2BoundaryRuleTests(unittest.TestCase):
    limit, distance = 0.001, 1e-5

    def setUp(self):
        generator = np.random.default_rng(7)
        self.out_weight, self.out_bias = generator.normal(size=(6, 4)), generator.normal(size=6)
        self.fuse_weight, self.fuse_bias = generator.normal(size=(5, 9)), generator.normal(size=5)
        self.embed = generator.normal(size=(1, 3))

    def outputs(self, codes, embed=None):
        lm_hidden = (codes / 9) @ self.out_weight.T + self.out_bias
        residual = np.concatenate((lm_hidden, self.embed if embed is None else embed), -1) @ self.fuse_weight.T + self.fuse_bias
        return lm_hidden.astype(np.float32), residual.astype(np.float32)

    def candidates(self, scaled, codes):
        rows = fuse_candidates(scaled, codes, self.distance)
        for row in rows:
            changed = codes.copy()
            changed[0, row["dim"]] = row["candidate_code"]
            row["outputs"] = self.outputs(changed)
        return rows

    def classify(self, scaled, actual_codes, embed=None):
        codes = np.round(scaled)
        actual = self.outputs(actual_codes, embed)
        lm_hidden = self.outputs(actual_codes)[0] if embed is not None else actual[0]
        return classify_fuse(self.outputs(codes), (lm_hidden, actual[1]), self.candidates(scaled, codes), self.limit)

    def test_only_dimensions_near_a_rounding_boundary_are_candidates(self):
        scaled = np.array([[0.2, 2.5000004, -1.6, -3.4999996]])
        rows = fuse_candidates(scaled, np.round(scaled), self.distance)
        self.assertEqual([(row["dim"], row["code"], row["candidate_code"]) for row in rows], [(1, 3.0, 2.0), (3, -3.0, -4.0)])
        self.assertLess(max(row["distance"] for row in rows), self.distance)

    def test_one_flip_within_the_distance_is_boundary_consistent(self):
        scaled = np.array([[0.2, 2.5000004, -1.6, 3.4]])
        result = self.classify(scaled, np.array([[0.0, 2.0, -2.0, 3.0]]))
        self.assertEqual(result["category"], "boundary")
        self.assertEqual((result["boundary"]["dim"], result["boundary"]["code"], result["boundary"]["candidate_code"]), (1, 3.0, 2.0))
        self.assertEqual(self.classify(scaled, np.round(scaled))["category"], "match")

    def test_a_flip_farther_than_the_distance_fails(self):
        scaled = np.array([[0.2, 2.5000004, -1.6, 3.4]])
        self.assertEqual(self.classify(scaled, np.array([[0.0, 3.0, -2.0, 4.0]]))["category"], "fail")

    def test_two_flipped_dimensions_fail(self):
        scaled = np.array([[0.5000003, 2.5000004, -1.6, 3.4]])
        self.assertEqual(self.classify(scaled, np.array([[0.0, 2.0, -2.0, 3.0]]))["category"], "fail")

    def test_matching_lm_hidden_without_matching_residual_input_fails(self):
        scaled = np.array([[0.2, 2.5000004, -1.6, 3.4]])
        result = self.classify(scaled, np.array([[0.0, 2.0, -2.0, 3.0]]), embed=self.embed + 1)
        self.assertEqual(result["category"], "fail")

    def test_two_candidates_that_both_match_fail(self):
        self.out_weight[:, 1] = self.out_weight[:, 0]
        scaled = np.array([[2.5000003, 2.5000004, -1.6, 3.4]])
        result = self.classify(scaled, np.array([[3.0, 2.0, -2.0, 3.0]]))
        self.assertEqual((result["category"], result["matched_candidates"]), ("fail", 2))

    def test_boundary_calls_above_the_fraction_fail_the_whole_result(self):
        def rows(boundary):
            return [{"case": "fuse", "gate": True, "passed": True, "nonfinite": 0, "fixture": "f", "index": index,
                     "category": "boundary" if index < boundary else "match",
                     "boundary": {"dim": 0, "distance": 0.0, "code": 1.0, "candidate_code": 0.0}} for index in range(100)]
        self.assertTrue(verdict(rows(2), CONTRACT["criteria"]))
        self.assertFalse(verdict(rows(3), CONTRACT["criteria"]))
        summary = fsq_summary(rows(3), CONTRACT["criteria"])
        self.assertEqual((summary["calls"], summary["boundary"], summary["fixtures"]["f"]["boundary"]), (100, 3, 3))

    def test_non_finite_fuse_outputs_fail_without_breaking_the_summary(self):
        generator = np.random.default_rng(3)
        hidden, patches, length = 4, 3, 2
        contract = copy.deepcopy(CONTRACT)
        contract["execution"]["prefill_length"] = 3

        def values(*shape):
            return generator.normal(size=shape).astype(np.float32)

        arrays = {"lm_hidden_in": values(patches, 1, hidden), "residual_hidden_in": values(patches, 1, hidden),
                  "mu": values(patches, 1, hidden), "stop_logits": values(patches, 1, 2),
                  "tslm_step_out": values(patches - 1, 1, hidden), "tslm_step_in": values(patches - 1, 1, hidden),
                  "fsq_out": values(patches - 1, 1, hidden), "fuse_out": values(patches - 1, 1, hidden),
                  "fsq_tie_distance": np.full(patches - 1, 0.25, np.float32),
                  "fsq_candidate_step": np.zeros(0, np.int32), "fsq_candidate_dim": np.zeros(0, np.int32),
                  "fsq_candidate_distance": np.zeros(0, np.float32), "fsq_candidate_code": np.zeros((0, 2), np.float32),
                  "fsq_candidate_lm_hidden": np.zeros((0, 1, hidden), np.float32),
                  "fsq_candidate_residual_input": np.zeros((0, 1, hidden), np.float32),
                  "tslm_prefill_out": values(1, length, hidden), "prefill_fuse_out": values(1, length, hidden)}

        class Projection:
            calls = {"heads": 0, "fuse": 0}

            def call(self, asset, function, inputs, state=None):
                if function == "prefill_fuse":
                    padding = np.zeros((1, 1, hidden), np.float32)
                    return {"residual_inputs": np.concatenate((arrays["prefill_fuse_out"], padding), 1)}
                index = self.calls[function]
                self.calls[function] += 1
                if function == "heads":
                    return {"dit_hidden": arrays["mu"][index], "stop_logits": arrays["stop_logits"][index]}
                if index == 1:
                    return {"lm_hidden": np.full((1, hidden), np.nan, np.float32),
                            "residual_input": np.full((1, hidden), np.inf, np.float32)}
                return {"lm_hidden": arrays["fsq_out"][index], "residual_input": arrays["fuse_out"][index]}

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "fixture"
            folder.mkdir()
            for name, value in arrays.items():
                np.save(folder / f"{name}.npy", value, allow_pickle=False)
            record = {"id": "fixture", "directory": "fixture", "length": length}
            rows = Checks(Projection(), contract, directory, record, np.float32).run(["proj"])
        summary = aggregate(rows)
        self.assertEqual((summary["proj.fuse"]["count"], summary["proj.fuse"]["passed"]), (2, 1))
        broken = [row for row in rows if row["case"] == "fuse" and not row["passed"]]
        self.assertEqual([(row["index"], row["category"], row["nonfinite"]) for row in broken], [(1, "fail", 2 * hidden)])
        self.assertNotIn("normalized_max", broken[0])
        self.assertEqual(fsq_summary(rows, contract["criteria"])["fail"], 1)
        self.assertFalse(verdict(rows, contract["criteria"]))


def tiny_lm(rope, seed):
    import torch
    from voxcpm.modules.minicpm4 import MiniCPM4Config, MiniCPMModel

    torch.manual_seed(seed)
    factors = [1.0 + 0.05 * math.sin(index + 1) for index in range(8)]
    config = MiniCPM4Config(bos_token_id=1, eos_token_id=2, hidden_size=32, intermediate_size=48,
                            max_position_embeddings=32768, num_attention_heads=4, num_hidden_layers=2,
                            num_key_value_heads=2, rms_norm_eps=1e-5,
                            rope_scaling={"type": "longrope", "long_factor": factors, "short_factor": factors,
                                          "original_max_position_embeddings": 32768},
                            vocab_size=64 if rope else 0, use_mup=False, scale_emb=12.0, dim_model_base=256,
                            scale_depth=1.4, rope_theta=10000.0, kv_channels=16, no_rope=not rope)
    lm = MiniCPMModel(config)
    with torch.no_grad():
        for name, parameter in lm.named_parameters():
            if name.endswith("norm.weight"):
                parameter.uniform_(0.5, 1.5)
    return lm.eval().requires_grad_(False)


@unittest.skipUnless(importlib.util.find_spec("voxcpm"), "The pinned VoxCPM environment is required")
class VoxCPM2WrapperTests(unittest.TestCase):
    capacity, size = 24, 8

    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        torch.set_num_threads(2)

    def original(self, lm, embeds, length, inputs):
        torch = self.torch
        lm.setup_cache(1, self.capacity, "cpu", torch.float32)
        with torch.no_grad():
            hidden, kv = lm(embeds[:, :length], is_causal=True)
            lm.kv_cache.fill_caches(kv)
            steps = [lm.forward_step(value, torch.tensor([lm.kv_cache.step()])).clone() for value in inputs]
        return hidden, steps, torch.stack([key[0] for key, _ in kv]), torch.stack([value[0] for _, value in kv])

    def converted(self, prefill, step, embeds, length, inputs):
        torch = self.torch
        for module in (prefill, step):
            module.k_cache.zero_()
            module.v_cache.zero_()
        with torch.no_grad():
            hidden = prefill(embeds, torch.tensor([length], dtype=torch.int32))
            step.k_cache.copy_(prefill.k_cache)
            step.v_cache.copy_(prefill.v_cache)
            steps = [step(value, torch.tensor([length + index], dtype=torch.int32)) for index, value in enumerate(inputs)]
        return hidden, steps

    def test_prefill_then_steps_match_the_original_cache_to_capacity(self):
        from Tools.Conversion.adapters.voxcpm2.wrappers import Prefill, Step

        torch = self.torch
        for rope in (True, False):
            for length in (1, 5, self.size):
                with self.subTest(rope=rope, length=length):
                    lm = tiny_lm(rope, 3 + length)
                    embeds = torch.randn(1, self.size, 32)
                    inputs = [torch.randn(1, 32) for _ in range(self.capacity - length)]
                    hidden, steps, keys, values = self.original(lm, embeds, length, inputs)
                    prefill, step = Prefill(lm, self.capacity, self.size).eval(), Step(lm, self.capacity).eval()
                    actual, actual_steps = self.converted(prefill, step, embeds, length, inputs)
                    torch.testing.assert_close(actual[:, :length], hidden, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(prefill.k_cache[:, :, :length], keys, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(prefill.v_cache[:, :, :length], values, rtol=1e-5, atol=1e-6)
                    self.assertFalse(torch.any(prefill.k_cache[:, :, length:]))
                    self.assertFalse(torch.any(prefill.v_cache[:, :, length:]))
                    torch.testing.assert_close(torch.stack(actual_steps), torch.stack(steps), rtol=1e-5, atol=1e-6)

    def test_padding_after_the_length_does_not_change_the_result(self):
        from Tools.Conversion.adapters.voxcpm2.wrappers import Prefill, Step

        torch = self.torch
        lm = tiny_lm(True, 11)
        embeds = torch.randn(1, self.size, 32)
        noisy = torch.cat((embeds[:, :3], 100 * torch.randn(1, self.size - 3, 32)), dim=1)
        inputs = [torch.randn(1, 32) for _ in range(9)]
        prefill, step = Prefill(lm, self.capacity, self.size).eval(), Step(lm, self.capacity).eval()
        first, first_steps = self.converted(prefill, step, embeds, 3, inputs)
        cache = prefill.k_cache.clone()
        second, second_steps = self.converted(prefill, step, noisy, 3, inputs)
        torch.testing.assert_close(second[:, :3], first[:, :3], rtol=0, atol=0)
        torch.testing.assert_close(prefill.k_cache, cache, rtol=0, atol=0)
        torch.testing.assert_close(torch.stack(second_steps), torch.stack(first_steps), rtol=0, atol=0)

    def test_exported_programs_keep_the_cache_contract(self):
        from Tools.Conversion.adapters.voxcpm2.wrappers import Prefill, Step

        torch = self.torch
        lm = tiny_lm(True, 17)
        embeds, length = torch.randn(1, self.size, 32), 6
        inputs = [torch.randn(1, 32) for _ in range(10)]
        hidden, steps, _, _ = self.original(lm, embeds, length, inputs)
        prefill, step = Prefill(lm, self.capacity, self.size).eval(), Step(lm, self.capacity).eval()
        with torch.no_grad():
            exported_prefill = torch.export.export(prefill, (embeds, torch.tensor([self.size], dtype=torch.int32)), strict=False).module()
            exported_step = torch.export.export(step, (inputs[0], torch.tensor([self.size], dtype=torch.int32)), strict=False).module()
        actual, actual_steps = self.converted(exported_prefill, exported_step, embeds, length, inputs)
        torch.testing.assert_close(actual[:, :length], hidden, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(torch.stack(actual_steps), torch.stack(steps), rtol=1e-5, atol=1e-6)

    def test_projections_follow_the_original_prefill_and_loop_expressions(self):
        from voxcpm.modules.layers import ScalarQuantizationLayer
        from voxcpm.modules.locenc import VoxCPMLocEnc
        from Tools.Conversion.adapters.voxcpm2.wrappers import Fuse, Heads, LocEnc, PrefillFuse

        torch = self.torch
        torch.manual_seed(5)
        config = copy.deepcopy(tiny_lm(True, 23).config)
        config.vocab_size = 0
        model = SimpleNamespace(fsq_layer=ScalarQuantizationLayer(32, 32, 8, 9).eval(), fusion_concat_proj=torch.nn.Linear(64, 32),
                                lm_to_dit_proj=torch.nn.Linear(32, 16), res_to_dit_proj=torch.nn.Linear(32, 16),
                                stop_proj=torch.nn.Linear(32, 32), stop_actn=torch.nn.SiLU(), stop_head=torch.nn.Linear(32, 2, bias=False),
                                feat_encoder=VoxCPMLocEnc(config, input_dim=6).eval(), enc_to_lm_proj=torch.nn.Linear(32, 32))
        with torch.no_grad():
            enc, feat_embed = torch.randn(1, 5, 32), torch.randn(1, 5, 32)
            text_mask, feat_mask = torch.ones(1, 5), torch.zeros(1, 5)
            mixed = model.fsq_layer(enc) * feat_mask.unsqueeze(-1) + enc * text_mask.unsqueeze(-1)
            prefill = model.fusion_concat_proj(torch.cat((mixed, feat_mask.unsqueeze(-1) * feat_embed), dim=-1))
            torch.testing.assert_close(PrefillFuse(model)(enc), prefill, rtol=0, atol=0)
            raw, embed, residual = torch.randn(1, 32), torch.randn(1, 32), torch.randn(1, 32)
            lm_hidden, residual_input = Fuse(model)(raw, embed)
            torch.testing.assert_close(lm_hidden, model.fsq_layer(raw), rtol=0, atol=0)
            torch.testing.assert_close(residual_input, model.fusion_concat_proj(torch.cat((model.fsq_layer(raw), embed), dim=-1)), rtol=0, atol=0)
            dit_hidden, stop = Heads(model)(lm_hidden, residual)
            torch.testing.assert_close(dit_hidden, torch.cat((model.lm_to_dit_proj(lm_hidden), model.res_to_dit_proj(residual)), dim=-1), rtol=0, atol=0)
            torch.testing.assert_close(stop, model.stop_head(model.stop_actn(model.stop_proj(lm_hidden))), rtol=0, atol=0)
            patch = torch.randn(1, 6, 4)
            expected = model.enc_to_lm_proj(model.feat_encoder(patch.transpose(1, 2).unsqueeze(1)))[:, 0, :]
            torch.testing.assert_close(LocEnc(model)(patch), expected, rtol=0, atol=0)

    def test_export_examples_are_distinct_tensors(self):
        from Tools.Conversion.adapters.voxcpm2.wrappers import functions

        linear = self.torch.nn.Linear(2, 2)
        model = SimpleNamespace(fsq_layer=linear, fusion_concat_proj=linear, lm_to_dit_proj=linear, res_to_dit_proj=linear,
                                stop_proj=linear, stop_actn=linear, stop_head=linear,
                                feat_decoder=SimpleNamespace(estimator=linear))
        for asset in ("proj", "dit"):
            for name, _, example, inputs, _, _ in functions(model, asset, CONTRACT):
                with self.subTest(asset=asset, function=name):
                    self.assertEqual(len({id(value) for value in example}), len(inputs))

    def test_kept_fp32_modules_follow_the_contract(self):
        from Tools.Conversion.adapters.voxcpm2.wrappers import KEPT_FP32

        self.assertEqual(list(KEPT_FP32), CONTRACT["fp16"]["kept_fp32_modules"])

    def test_reference_rejects_fixtures_that_disagree_with_the_contract_before_loading(self):
        from unittest import mock

        from Tools.Conversion.adapters.voxcpm2 import reference

        bundled = json.loads((RECIPE / "fixtures.json").read_text(encoding="utf-8"))

        def run(specification):
            with tempfile.TemporaryDirectory() as directory:
                fixtures, output = Path(directory) / "fixtures.json", Path(directory) / "output"
                fixtures.write_text(json.dumps(specification), encoding="utf-8")
                arguments = ["reference", "--snapshot", directory, "--upstream", directory, "--contract",
                             str(RECIPE / "model.json"), "--fixtures", str(fixtures), "--output", str(output)]
                with mock.patch("sys.argv", arguments), mock.patch.object(reference, "load_model", side_effect=LookupError) as load:
                    with self.assertRaises((LookupError, ValueError)) as raised:
                        reference.main()
                return type(raised.exception), load.call_count, output.exists()

        self.assertEqual(run(bundled), (LookupError, 1, False))
        for key, value in (("inference_timesteps", 9), ("cfg_value", 1.5), ("min_len", 3), ("max_len", 2048)):
            with self.subTest(key=key):
                self.assertEqual(run({**bundled, key: value}), (ValueError, 0, False))

    def test_streaming_vae_matches_the_original_streaming_decoder(self):
        from voxcpm.modules.audiovae import AudioVAEConfigV2, AudioVAEV2
        from Tools.Conversion.adapters.voxcpm2.wrappers import VAEStream

        torch = self.torch
        torch.manual_seed(29)
        vae = AudioVAEV2(AudioVAEConfigV2(encoder_dim=4, encoder_rates=[2, 2], latent_dim=8, decoder_dim=16,
                                          decoder_rates=[2, 3])).eval().requires_grad_(False)
        with torch.no_grad():
            for name, parameter in vae.decoder.named_parameters():
                if "embed" in name or name.endswith("alpha"):
                    parameter.uniform_(0.5, 1.5)
        latent = torch.randn(1, 8, 4 * 5)
        with torch.no_grad(), vae.streaming_decode() as decoder:
            expected = [decoder.decode_chunk(latent[..., 4 * index:4 * index + 4]) for index in range(5)]
        with torch.no_grad():
            full = vae.decode(latent)
        stream = VAEStream(vae).eval()
        with torch.no_grad():
            actual = [stream(latent[..., 4 * index:4 * index + 4]) for index in range(5)]
            exported = torch.export.export(VAEStream(vae).eval(), (latent[..., :4],), strict=False).module()
            replay = [exported(latent[..., 4 * index:4 * index + 4]) for index in range(5)]
        torch.testing.assert_close(torch.cat(actual, -1), torch.cat(expected, -1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(torch.cat(replay, -1), torch.cat(expected, -1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(torch.cat(actual, -1), full, rtol=1e-4, atol=1e-5)
        self.assertEqual(len(stream.state_names), len({name for name in stream.state_names}))


if __name__ == "__main__":
    unittest.main()
