import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from Tools.Conversion import summary
from Tools.Conversion.adapters.jev27b.merge import (
    lora_pairs, manifest_files, merge, merge_projection, merged_revision, saved_index, saved_name, targeted,
)
from Tools.Conversion.adapters.jev27b.verify_merge import THRESHOLD, compare_outputs, identical
from Tools.Conversion.storage import PipelineError

TORCH = importlib.util.find_spec("torch") is not None
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj"]
LAYER = "model.layers.0.linear_attn"


def lora_key(name, side):
    return f"base_model.model.{name[:-len('.weight')]}.lora_{side}.weight"


@unittest.skipUnless(TORCH, "The JEV-27B reference environment is required")
class MergeFormulaTests(unittest.TestCase):
    def test_merge_rounds_the_fp32_sum_to_bfloat16_once(self):
        import torch

        generator = torch.Generator().manual_seed(20261102)
        weight = (torch.randn(64, 48, generator=generator) * 0.02).to(torch.bfloat16)
        a = torch.randn(16, 48, generator=generator) * 0.05
        b = torch.randn(64, 16, generator=generator) * 0.05
        merged = merge_projection(weight, a, b, 2.0)
        self.assertEqual(merged.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(merged, (weight.float() + 2.0 * (b @ a)).to(torch.bfloat16)))
        self.assertFalse(torch.equal(merged, weight + (2.0 * (b @ a)).to(torch.bfloat16)))

    def test_tensors_are_compared_bit_for_bit(self):
        import torch

        values = torch.tensor([float("nan"), 1.5]).to(torch.bfloat16)
        self.assertTrue(identical(values, values.clone()))
        self.assertFalse(identical(torch.zeros(1, dtype=torch.bfloat16), -torch.zeros(1, dtype=torch.bfloat16)))
        self.assertFalse(identical(values, values.float()))
        self.assertFalse(identical(values.reshape(1, 2), values))

    def test_merge_writes_renamed_shards_with_bit_identical_copies_and_a_stable_size(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        generator = torch.Generator().manual_seed(7)
        tensors = {"model.embed_tokens.weight": torch.randn(8, 4, generator=generator).to(torch.bfloat16),
                   f"{LAYER}.in_proj_qkv.weight": torch.randn(6, 4, generator=generator).to(torch.bfloat16),
                   f"{LAYER}.in_proj_a.weight": torch.randn(2, 4, generator=generator).to(torch.bfloat16),
                   f"{LAYER}.A_log": torch.randn(2, generator=generator).to(torch.bfloat16),
                   "lm_head.weight": torch.randn(8, 4, generator=generator).to(torch.bfloat16)}
        weight_map = {name: "model-00001-of-00002.safetensors" if "layers" not in name else "model-00002-of-00002.safetensors"
                      for name in tensors}
        target = f"{LAYER}.in_proj_qkv.weight"
        lora = {lora_key(target, "A"): torch.randn(16, 4, generator=generator),
                lora_key(target, "B"): torch.randn(6, 16, generator=generator)}
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            source.mkdir()
            for shard in set(weight_map.values()):
                save_file({name: tensor for name, tensor in tensors.items() if weight_map[name] == shard}, str(source / shard))
            outputs = []
            for name in ("first", "second"):
                checkpoint = Path(folder) / name
                checkpoint.mkdir()
                consumed, total, dtypes = merge(source, checkpoint, weight_map, lora_pairs(lora), lora, 2.0)
                outputs.append({path.name: path.read_bytes() for path in checkpoint.iterdir()})
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(dict(consumed), {target: 1})
            self.assertEqual(total, sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()))
            self.assertEqual(dtypes, {"source": {"torch.bfloat16": 5}, "saved": {"torch.bfloat16": 5}})
            with safe_open(str(checkpoint / "model-00002-of-00002.safetensors"), framework="pt") as handle:
                self.assertEqual(sorted(handle.keys()), sorted(saved_name(name) for name in weight_map
                                                               if weight_map[name].startswith("model-00002")))
                self.assertEqual(handle.metadata(), {"format": "pt"})
                self.assertTrue(torch.equal(handle.get_tensor(saved_name(target)),
                                            merge_projection(tensors[target], *lora.values(), 2.0)))
                self.assertTrue(torch.equal(handle.get_tensor(saved_name(f"{LAYER}.A_log")), tensors[f"{LAYER}.A_log"]))


class KeyMappingTests(unittest.TestCase):
    def test_lora_names_map_to_backbone_and_saved_names(self):
        names = [f"{LAYER}.in_proj_qkv.weight", "model.layers.3.self_attn.q_proj.weight", "model.layers.3.mlp.down_proj.weight"]
        keys = [lora_key(name, side) for name in names for side in "BA"]
        self.assertEqual(lora_pairs(reversed(keys)), {name: (lora_key(name, "A"), lora_key(name, "B")) for name in sorted(names)})
        self.assertEqual(saved_name(f"{LAYER}.in_proj_qkv.weight"), "model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
        self.assertEqual(saved_name("model.embed_tokens.weight"), "model.language_model.embed_tokens.weight")
        self.assertEqual(saved_name("model.norm.weight"), "model.language_model.norm.weight")
        self.assertEqual(saved_name("lm_head.weight"), "lm_head.weight")

    def test_unpaired_or_foreign_names_are_rejected(self):
        for keys in ([lora_key(f"{LAYER}.in_proj_z.weight", "A")], ["model.layers.0.mlp.up_proj.weight"],
                     [lora_key(f"{LAYER}.out_proj.weight", "A").replace("lora_A", "lora_C")]):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                lora_pairs(keys)
        for name in ("model.language_model.norm.weight", "mtp.fc.weight", "visual.blocks.0.weight", "lm_head.bias"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                saved_name(name)

    def test_only_the_ten_projection_targets_are_merged(self):
        merged = [f"{LAYER}.{name}.weight" for name in ("in_proj_qkv", "in_proj_z", "out_proj")] + [
            f"model.layers.3.self_attn.{name}.weight" for name in ("q_proj", "k_proj", "v_proj", "o_proj")] + [
            f"model.layers.1.mlp.{name}.weight" for name in ("gate_proj", "up_proj", "down_proj")]
        kept = [f"{LAYER}.{name}" for name in ("in_proj_a.weight", "in_proj_b.weight", "conv1d.weight", "A_log", "dt_bias",
                                                "norm.weight")] + [
            "model.layers.3.self_attn.q_norm.weight", "model.layers.0.input_layernorm.weight", "model.embed_tokens.weight",
            "model.norm.weight", "lm_head.weight"]
        self.assertEqual(targeted(merged + kept, TARGETS), sorted(merged))


class IndexAndRevisionTests(unittest.TestCase):
    def test_the_saved_index_keeps_membership_and_size_in_any_input_order(self):
        weight_map = {"lm_head.weight": "model-00002-of-00002.safetensors", "model.norm.weight": "model-00002-of-00002.safetensors",
                      "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                      f"{LAYER}.in_proj_qkv.weight": "model-00001-of-00002.safetensors"}
        index = saved_index(weight_map, 1234)
        self.assertEqual(json.dumps(index), json.dumps(saved_index(dict(reversed(weight_map.items())), 1234)))
        self.assertEqual(index["metadata"], {"total_size": 1234})
        self.assertEqual(list(index["weight_map"]), sorted(index["weight_map"]))
        self.assertEqual(index["weight_map"], {saved_name(name): shard for name, shard in weight_map.items()})

    def test_the_merged_revision_depends_only_on_file_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = [Path(folder) / name for name in ("first", "second")]
            for root, names in zip(roots, (("b.json", "a.safetensors"), ("a.safetensors", "b.json"))):
                root.mkdir()
                for name in names:
                    (root / name).write_bytes(name.encode() * 3)
            os.utime(roots[1] / "b.json", (1, 1))
            files = manifest_files(roots[0])
            self.assertEqual(files, manifest_files(roots[1]))
            self.assertEqual([entry["path"] for entry in files], ["a.safetensors", "b.json"])
            canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.assertEqual(merged_revision(files), hashlib.sha256(canonical).hexdigest()[:40])
            (roots[1] / "b.json").write_bytes(b"changed")
            self.assertNotEqual(merged_revision(manifest_files(roots[1])), merged_revision(files))


def row(identifier, p1, p1m, rejected=False, kind="noul", tokens=(1, 2)):
    stages = {} if rejected else {"logits0": [0.0] * 24, "logits1": [0.0] * 24, "p0": [0.5, 0.5], "p0_T": [0.5, 0.5],
                                  "p1": p1, "p1_T": p1, "elapsed0_s": 0.1, "elapsed1_s": 0.1}
    merged = {} if rejected else {"logits1m": [0.0] * 24, "p1m": p1m, "p1m_T": p1m, "elapsed1m_s": 0.1}
    empty = {key: None for key in ("logits0", "logits1", "logits1m", "p0", "p1", "p1m", "p0_T", "p1_T", "p1m_T",
                                   "elapsed0_s", "elapsed1_s", "elapsed1m_s")}
    shared = {"id": identifier, "kind": kind, "options": ["false", "true"], "band": None if rejected else "<=512",
              "row_sha256": "r", "text_sha256": "t", "token_count": len(tokens), "tokens": list(tokens), "rejected": rejected}
    return {**shared, **empty, **stages}, {**shared, **empty, **merged}


class OutputComparisonTests(unittest.TestCase):
    def compare(self, *rows):
        return compare_outputs({"passed": True, "rows": [first for first, _ in rows]},
                               {"passed": True, "rows": [second for _, second in rows]})

    def test_rows_within_the_threshold_pass_and_rejected_rows_are_not_compared(self):
        outputs, joined, compared = self.compare(row("a", [0.2, 0.8], [0.205, 0.795]), row("b", [0.6, 0.4], [0.6, 0.4]),
                                                 row("c", None, None, rejected=True))
        self.assertTrue(outputs["passed"])
        self.assertEqual((compared, len(joined)), (2, 3))
        self.assertAlmostEqual(outputs["max_probability_error"], 0.005)
        self.assertEqual(outputs["violations"], [])
        self.assertEqual(outputs["adaptation"]["rows_above"], 2)
        self.assertEqual(set(joined[0]), {"id", "kind", "options", "band", "row_sha256", "text_sha256", "token_count", "tokens",
                                          "rejected", "logits0", "logits1", "logits1m", "p0", "p1", "p1m", "p0_T", "p1_T",
                                          "p1m_T", "elapsed0_s", "elapsed1_s", "elapsed1m_s"})

    def test_a_larger_difference_or_different_tokens_fail(self):
        outputs, _, _ = self.compare(row("a", [0.49, 0.51], [0.51, 0.49]))
        self.assertFalse(outputs["passed"])
        self.assertGreater(outputs["max_probability_error"], THRESHOLD)
        self.assertEqual([violation["id"] for violation in outputs["violations"]], ["a"])
        self.assertEqual(outputs["top1_changed"], {"rows": 1, "ids": ["a"]})
        first, _ = row("a", [0.2, 0.8], [0.2, 0.8])
        _, second = row("a", [0.2, 0.8], [0.2, 0.8], tokens=(1, 3))
        outputs, _, _ = compare_outputs({"passed": True, "rows": [first]}, {"passed": True, "rows": [second]})
        self.assertFalse(outputs["passed"])
        self.assertEqual(outputs["differing_rows"], ["a"])
        first, second = row("a", [0.2, 0.8], [0.2, 0.8])
        outputs, _, _ = compare_outputs({"passed": False, "rows": [first]}, {"passed": True, "rows": [second]})
        self.assertEqual((outputs["passed"], outputs["references_passed"], outputs["rows_match"]), (False, False, True))
        _, rejected = row("a", None, None, rejected=True)
        outputs, joined, compared = compare_outputs({"passed": True, "rows": [first]}, {"passed": True, "rows": [rejected]})
        self.assertEqual((outputs["passed"], outputs["rows_match"], outputs["differing_rows"], compared, joined),
                         (False, False, ["a"], 0, []))


class MergeSummaryTests(unittest.TestCase):
    report = {"version": 1, "status": "completed", "verification": "passed", "stages": []}
    comparison = {"schema": "typed-decision-merge/1", "passed": True, "rows": 302, "compared": 301, "rejected": 1,
                  "merged_revision": "private-example-sentinel",
                  "weights": {"passed": True, "pairs_consumed": 400, "mismatched": ["private-example-sentinel"]},
                  "outputs": {"max_probability_error": 0.004, "mean_probability_error": 0.0005,
                              "violations": [{"id": "private-example-sentinel"}]}}

    def test_the_merge_comparison_projects_only_its_numbers(self):
        result = summary.project(self.report, self.comparison)
        self.assertEqual(result["comparison"], {"passed": True, "rows": 302, "compared": 301, "rejected": 1,
                                                "max_probability_error": 0.004, "mean_probability_error": 0.0005,
                                                "weights_passed": True, "pairs_consumed": 400})
        self.assertNotIn("private-example-sentinel", json.dumps(result))

    def test_the_merge_comparison_rejects_invalid_values(self):
        for mutate in (lambda c: c["weights"].update(passed="yes"), lambda c: c["outputs"].update(max_probability_error=-1),
                       lambda c: c.update(rows=1.5), lambda c: c["weights"].update(pairs_consumed=True)):
            comparison = json.loads(json.dumps(self.comparison))
            mutate(comparison)
            with self.subTest(comparison=comparison), self.assertRaises(PipelineError):
                summary.project(self.report, comparison)


if __name__ == "__main__":
    unittest.main()
