import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from Tools.Conversion import local
from Tools.Conversion.adapters.jev27b.prepare import (
    check_adapter, check_decision, check_integrity, check_setup, git_blob, load_contract, tokenizer_sections,
)
from Tools.Conversion.adapters.jev27b.typed_head import FIXED_OPTIONS, LABELS
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import PipelineError

RECIPE = Path(__file__).resolve().parents[1] / "recipes/jev27b"
REVISION = "a" * 40
REPOSITORY = "owner/model"


def write_record(folder, files, revision=REVISION, checked=None):
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, data in files.items():
        (folder / name).write_bytes(data)
        entries.append({"path": name, "expected_size": len(data), "actual_size": len(data),
                        "actual_sha256": hashlib.sha256(data).hexdigest(), "expected_sha256": None,
                        "expected_git_blob_sha1": git_blob(data), "actual_git_blob_sha1": git_blob(data),
                        "check": "git-blob-sha1", "ok": True})
    checked = checked or datetime.now(timezone.utc) + timedelta(seconds=5)
    record = {"repo_id": REPOSITORY, "revision": revision, "checked_at": checked.isoformat(timespec="seconds"),
              "files": entries, "files_checked": len(entries), "ok_true": len(entries), "ok_false": 0,
              "local_files_not_in_scope": []}
    Path(f"{folder}.integrity.json").write_text(json.dumps(record))
    return record


def write_header(path, tensors):
    header, offset = {}, 0
    for name, shape in tensors.items():
        size = 4 * math.prod(shape)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


class ContractTests(unittest.TestCase):
    def test_the_recipe_contract_is_accepted(self):
        contract = load_contract(RECIPE / "model.json")
        self.assertEqual((contract["lora"]["pairs"], contract["lora"]["scale"], contract["limits"]["input_tokens"]), (400, 2.0, 8192))

    def test_changed_lora_limits_revisions_or_fields_are_rejected(self):
        original = json.loads((RECIPE / "model.json").read_text())
        mutations = [lambda c: c["lora"].update(scale=1.0), lambda c: c["lora"].update(use_dora=True),
                     lambda c: c["lora"].update(use_rslora=True), lambda c: c["limits"].update(options=17),
                     lambda c: c.update(revision="main"), lambda c: c.update(extra=1), lambda c: c.pop("head"),
                     lambda c: c["tokenizer"].update(policy="resave"), lambda c: c.update(merged_model_id="no-owner")]
        with tempfile.TemporaryDirectory() as folder:
            for index, mutate in enumerate(mutations):
                contract = copy.deepcopy(original)
                mutate(contract)
                path = Path(folder) / f"{index}.json"
                path.write_text(json.dumps(contract))
                with self.subTest(index=index), self.assertRaises(ValueError):
                    load_contract(path)


class IntegrityTests(unittest.TestCase):
    def test_a_complete_record_passes_and_small_files_are_rehashed(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "model"
            write_record(source, {"config.json": b"{}", "weights.bin": b"\0" * 64})
            (source / ".cache").mkdir()
            (source / ".cache/download.metadata").write_text("x")
            summary = check_integrity(source, REPOSITORY, REVISION)
            self.assertEqual((summary["files"], summary["bytes"], summary["content_rehashed"]), (2, 66, 2))

    def test_another_source_a_failed_entry_an_extra_file_or_a_changed_file_is_rejected(self):
        def rewrite(path, record, **changes):
            Path(f"{path}.integrity.json").write_text(json.dumps({**record, **changes}))

        def entry(record, **changes):
            return [{**record["files"][0], **changes}]

        changes = [lambda path, record: check_integrity(path, REPOSITORY, "b" * 40),
                   lambda path, record: check_integrity(path, "other/model", REVISION),
                   lambda path, record: (path / "config.json").write_bytes(b"{} "),
                   lambda path, record: (path / "config.json").write_bytes(b"[]"),
                   lambda path, record: os.utime(path / "config.json", (4102444800, 4102444800)),
                   lambda path, record: (path / "added_tokens.json").write_text("{}"),
                   lambda path, record: rewrite(path, record, ok_true=0),
                   lambda path, record: rewrite(path, record, local_files_not_in_scope=["added_tokens.json"]),
                   lambda path, record: rewrite(path, record, files=entry(record, actual_git_blob_sha1="0" * 40)),
                   lambda path, record: rewrite(path, record, files=entry(record, check="unknown"))]
        for index, change in enumerate(changes):
            with tempfile.TemporaryDirectory() as folder, self.subTest(index=index):
                source = Path(folder) / "model"
                record = write_record(source, {"config.json": b"{}"})
                with self.assertRaises(ValueError):
                    change(source, record)
                    check_integrity(source, REPOSITORY, REVISION)


class AdapterTests(unittest.TestCase):
    CONTRACT = {"lora": {"r": 2, "alpha": 4, "scale": 2.0, "use_rslora": False, "use_dora": False, "pairs": 2,
                         "targets": ["q_proj", "in_proj_qkv"]}}

    def snapshot(self, folder, modules, config=None):
        snapshot = Path(folder)
        (snapshot / "adapter").mkdir()
        (snapshot / "adapter/adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "r": 2, "lora_alpha": 4, "use_rslora": False, "use_dora": False, "modules_to_save": None,
            "trainable_token_indices": None, "target_parameters": None, "layers_to_transform": None, "layers_pattern": None,
            "exclude_modules": None, "rank_pattern": {}, "alpha_pattern": {}, "bias": "none", "lora_bias": False,
            "fan_in_fan_out": False, "layer_replication": None, "use_qalora": False, "alora_invocation_tokens": None,
            "arrow_config": None, "use_bdlora": None, "velora_config": None, "monteclora_config": None, "kasa_config": None,
            "target_modules": ["in_proj_qkv", "q_proj"], **(config or {})}))
        tensors = {}
        for module in modules:
            tensors[f"base_model.model.{module}.lora_A.weight"] = [2, 8]
            tensors[f"base_model.model.{module}.lora_B.weight"] = [8, 2]
        write_header(snapshot / "adapter/adapter_model.safetensors", tensors)
        names = ["model.layers.0.linear_attn.in_proj_qkv.weight", "model.layers.0.linear_attn.in_proj_a.weight",
                 "model.layers.1.self_attn.q_proj.weight", "model.layers.1.self_attn.q_norm.weight", "lm_head.weight"]
        (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "x" for name in names}}))
        return snapshot

    def test_pairs_covering_every_targeted_projection_are_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = self.snapshot(folder, ["model.layers.0.linear_attn.in_proj_qkv", "model.layers.1.self_attn.q_proj"])
            summary = check_adapter(snapshot, self.CONTRACT)
            self.assertEqual((summary["pairs"], summary["scale"], summary["per_target"]), (2, 2.0, {"in_proj_qkv": 1, "q_proj": 1}))

    def test_missing_pairs_or_changed_adapter_settings_fail_without_touching_the_contract(self):
        cases = [(["model.layers.0.linear_attn.in_proj_qkv"], None), (["model.layers.0.linear_attn.in_proj_qkv",
                  "model.layers.1.self_attn.q_proj"], {"lora_alpha": 8}), (["model.layers.0.linear_attn.in_proj_qkv",
                  "model.layers.1.self_attn.q_proj"], {"modules_to_save": ["lm_head"]}),
                 (["model.layers.0.linear_attn.in_proj_qkv", "model.layers.1.self_attn.q_proj"],
                  {"alora_invocation_tokens": [1, 2, 3]})]
        for index, (modules, config) in enumerate(cases):
            with tempfile.TemporaryDirectory() as folder, self.subTest(index=index), self.assertRaises(ValueError):
                check_adapter(self.snapshot(folder, modules, config), self.CONTRACT)

    def test_tokenizer_sections_ignore_the_merge_notation_only(self):
        left = {"model": {"vocab": {"a": 0}, "merges": ["a b"]}, "pre_tokenizer": {"Regex": "x"}}
        self.assertEqual(tokenizer_sections(left, {"model": {"vocab": {"a": 0}, "merges": [["a", "b"]]}, "pre_tokenizer": {"Regex": "x"}}), [])
        self.assertEqual(tokenizer_sections(left, {"model": {"vocab": {"a": 1}, "merges": [["a", "b"]]}, "pre_tokenizer": {}}),
                         ["model.vocab", "pre_tokenizer"])


class DecisionTests(unittest.TestCase):
    CONTRACT = {"head": "head.safetensors", "template_version": "bare-v1",
                "lora": {"r": 2, "alpha": 4, "targets": ["q_proj", "in_proj_qkv"]}}

    def snapshot(self, folder, judge=None, calibration=None):
        snapshot = Path(folder)
        verbalizers = [*FIXED_OPTIONS["noul"], *FIXED_OPTIONS["score"], *LABELS]
        (snapshot / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {token: index for index, token in enumerate(verbalizers)}}}))
        (snapshot / "judge_config.json").write_text(json.dumps({
            "slots": {"num_slots": 24, "ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]},
                      "verbalizers": verbalizers, "template_version": "bare-v1"},
            "verbalizer_ids": list(range(24)), "kinds": ["noul", "choice", "score"], "hidden_size": 8,
            "weights_mode": "unmerged", "adapter_subfolder": "adapter", "softcap": None,
            "lora": {"r": 2, "alpha": 4, "target_modules": ["in_proj_qkv", "q_proj"]}, **(judge or {})}))
        write_header(snapshot / "head.safetensors", {"proj.weight": [24, 8], "proj.bias": [24]})
        (snapshot / "calibration.json").write_text(json.dumps({
            "per_kind": {"noul": 1.01, "choice": 1.02, "score": 1.0}, "per_kind_family": {}, "d1_excluded": True,
            "fit_rows": 3, **(calibration or {})}))
        return snapshot

    def test_the_decision_layout_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = check_decision(self.snapshot(folder), self.CONTRACT, {"hidden_size": 8})
            self.assertEqual((summary["head"], summary["temperatures"]["choice"]), ({"proj.weight": [24, 8], "proj.bias": [24]}, 1.02))

    def test_softcap_family_temperatures_or_another_adapter_summary_fail(self):
        cases = [({"softcap": 30.0}, None), ({"lora": {"r": 99, "alpha": 4, "target_modules": ["lm_head"]}}, None),
                 (None, {"per_kind_family": {"noul": {"knowledge": 1.2}}}), (None, {"per_kind": {"noul": 0.0, "choice": 1.0, "score": 1.0}}),
                 ({"verbalizer_ids": list(range(1, 25))}, None)]
        for index, (judge, calibration) in enumerate(cases):
            with tempfile.TemporaryDirectory() as folder, self.subTest(index=index), self.assertRaises(ValueError):
                check_decision(self.snapshot(folder, judge, calibration), self.CONTRACT, {"hidden_size": 8})


class SetupRecordTests(unittest.TestCase):
    def test_prepare_requires_the_environment_built_from_the_current_inputs(self):
        recorded = {"environment_inputs": {name: sha256(RECIPE / name) for name in ("pyproject.toml", "uv.lock")},
                    "upstream_revision": REVISION, "llm_runner_sha256": "c" * 64}
        self.assertEqual(check_setup(recorded, RECIPE, REVISION, "c" * 64), recorded)
        stale = [{**recorded, "environment_inputs": {**recorded["environment_inputs"], "uv.lock": "0" * 64}},
                 {**recorded, "upstream_revision": "b" * 40}, {**recorded, "llm_runner_sha256": "d" * 64}, {}]
        for index, setup in enumerate(stale):
            with self.subTest(index=index), self.assertRaises(ValueError):
                check_setup(setup, RECIPE, REVISION, "c" * 64)


class SetupTests(unittest.TestCase):
    def test_setup_checks_the_pinned_inputs_before_creating_environments(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder) / "workspace"
            with mock.patch.object(local, "command") as command, self.assertRaises((OSError, ValueError)):
                local.setup(workspace, local.recipe_path("jev27b"), Path(folder) / "missing", Path(folder) / "lock")
            command.assert_not_called()
            self.assertFalse(workspace.exists())
            self.assertFalse((Path(folder) / "lock").exists())

    def test_plan_and_run_use_the_jev27b_bindings_and_work_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            recipe = local.recipe_path("jev27b")
            with self.assertRaises(PipelineError):
                local.pipeline_command("plan", recipe, workspace, workspace / "lock", False)
            (workspace / "jev27b.paths.json").write_text("{}")
            command = local.pipeline_command("run", recipe, workspace, workspace / "lock", False)
            self.assertIn(str(workspace / "jev27b.paths.json"), command)
            self.assertIn(str(workspace / "work/jev27b"), command)


if __name__ == "__main__":
    unittest.main()
