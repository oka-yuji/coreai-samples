import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from Tools.Conversion.adapters.jev27b.merge import (
    lora_pairs, manifest_files, merge, merge_projection, merged_revision, saved_index, saved_name, targeted,
)

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


if __name__ == "__main__":
    unittest.main()
