import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Tools.Conversion import local
from Tools.Conversion.adapters.gemma4_31b.prepare import (
    TOKENIZER_FILES, check_config, check_hashes, check_record, check_setup, check_special_tokens, load_contract,
)
from Tools.Conversion.adapters.pinned import sha256
from Tools.Conversion.storage import PipelineError

RECIPE = Path(__file__).resolve().parents[1] / "recipes/gemma4_31b"
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


def write_record(snapshot, contract, **changes):
    rows = [{"path": name, "expected_size": entry["bytes"], "actual_size": entry["bytes"],
             "actual_sha256": entry["sha256"], "ok": True} for name, entry in contract["files"].items()]
    record = {"repo_id": contract["model_id"], "revision": contract["revision"], "checked_at": "2026-11-22T03:00:00+0900",
              "files": rows, "files_checked": len(rows), "ok_true": len(rows), "ok_false": 0,
              "local_files_not_in_scope": [".cache/huggingface/download/config.json.metadata"], **changes}
    Path(f"{snapshot}.integrity.json").write_text(json.dumps(record))


def write_snapshot(folder):
    contract = load_contract(RECIPE / "model.json")
    config = contract["config"]
    text = {key: value for key, value in config.items() if key not in ("layer_pattern", "eos_token_id")}
    text.update(layer_types=config["layer_pattern"] * (config["num_hidden_layers"] // len(config["layer_pattern"])),
                model_type="gemma4_text", dtype="bfloat16", eos_token_id=1)
    added = [{"id": 0, "content": "<pad>"}, {"id": 1, "content": "<eos>"}, {"id": 2, "content": "<bos>"},
             {"id": 50, "content": "<|tool_response>"}, {"id": 106, "content": "<turn|>"}]
    documents = {"config.json": {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
                                 "eos_token_id": [1, 106], "text_config": text},
                 "generation_config.json": {"bos_token_id": 2, "eos_token_id": [1, 106, 50], "do_sample": True},
                 "tokenizer.json": {"added_tokens": added, "model": {"type": "BPE", "vocab": {"a": 3, "b": 4}}},
                 "tokenizer_config.json": {"bos_token": "<bos>", "eos_token": "<eos>", "eot_token": "<turn|>",
                                           "pad_token": "<pad>"}}
    snapshot = Path(folder) / "model"
    (snapshot / ".cache/huggingface/download").mkdir(parents=True)
    (snapshot / ".cache/huggingface/download/config.json.metadata").write_text("download state")
    for name, value in documents.items():
        (snapshot / name).write_text(json.dumps(value))
    (snapshot / "chat_template.jinja").write_text("{{ bos_token }}")
    for index, name in enumerate(SHARDS):
        (snapshot / name).write_bytes(bytes([index + 1]) * (64 >> index))
    pinned = {path.relative_to(snapshot).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
              for path in snapshot.rglob("*") if path.is_file() and ".cache" not in path.relative_to(snapshot).parts}
    contract = {**contract, "files": pinned,
                "tokenizer": {"policy": "keep-distributed", "files": {name: pinned[name]["sha256"] for name in TOKENIZER_FILES}}}
    write_record(snapshot, contract)
    return snapshot, contract


def edit_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


class ContractTests(unittest.TestCase):
    def test_the_recipe_contract_pins_the_published_shards_and_configuration(self):
        contract = load_contract(RECIPE / "model.json")
        config = contract["config"]
        self.assertEqual((config["num_hidden_layers"], config["hidden_size"], config["intermediate_size"], config["vocab_size"],
                          config["sliding_window"], config["num_key_value_heads"], config["num_global_key_value_heads"],
                          config["head_dim"], config["global_head_dim"], config["final_logit_softcapping"],
                          config["eos_token_id"]), (60, 5376, 21504, 262144, 1024, 16, 4, 256, 512, 30.0, [1, 106, 50]))
        self.assertEqual(config["layer_pattern"], ["sliding_attention"] * 5 + ["full_attention"])
        self.assertTrue(config["attention_k_eq_v"] and config["tie_word_embeddings"])
        self.assertEqual([contract["files"][name] for name in SHARDS], [
            {"bytes": 49784788364, "sha256": "eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa"},
            {"bytes": 12761549884, "sha256": "018912220f559f7025d60333e0996183cd538aa77ad6f4988a89ce47be681f10"}])
        self.assertEqual((len(contract["files"]), contract["reference"]), (12, {"weights": "bfloat16", "compute": "bfloat16",
                                                                                "device": "mps"}))

    def test_malformed_contracts_are_rejected(self):
        original = json.loads((RECIPE / "model.json").read_text())
        mutations = [lambda c: c.update(extra=1), lambda c: c.pop("upstream_revision"), lambda c: c.update(revision="main"),
                     lambda c: c["config"].pop("sliding_window"), lambda c: c["config"].update(num_hidden_layers=61),
                     lambda c: c["config"].update(eos_token_id=[1, 1]), lambda c: c["reference"].update(compute="float32"),
                     lambda c: c["tokenizer"]["files"].update({"tokenizer.json": "0" * 64}),
                     lambda c: c["tokenizer"].update(policy="resave"),
                     lambda c: c["files"][SHARDS[0]].update(sha256="eeef"),
                     lambda c: c["files"][SHARDS[1]].update(bytes=0),
                     lambda c: c["files"].update({"../config.json": {"bytes": 1, "sha256": "0" * 64}}),
                     lambda c: c["files"].update({".cache/x": {"bytes": 1, "sha256": "0" * 64}})]
        with tempfile.TemporaryDirectory() as folder:
            for index, mutate in enumerate(mutations):
                contract = copy.deepcopy(original)
                mutate(contract)
                path = Path(folder) / f"{index}.json"
                path.write_text(json.dumps(contract))
                with self.subTest(index=index), self.assertRaises(ValueError):
                    load_contract(path)


class SnapshotTests(unittest.TestCase):
    def test_a_snapshot_matching_the_contract_passes_and_ignores_the_download_state(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot, contract = write_snapshot(folder)
            record = check_record(snapshot, contract)
            self.assertEqual((record["files"], record["download_state_files"]), (len(contract["files"]), 1))
            self.assertEqual(set(check_hashes(snapshot, contract)), set(contract["files"]))
            self.assertEqual(check_config(snapshot, contract)["layer_types"], {"full_attention": 10, "sliding_attention": 50})
            tokens = check_special_tokens(snapshot, contract)
            self.assertEqual(tokens["stop_tokens"], {"1": "<eos>", "106": "<turn|>", "50": "<|tool_response>"})

    def test_missing_extra_resized_linked_or_changed_files_are_rejected(self):
        def relink(snapshot, contract):
            target = snapshot.parent / "copy.json"
            target.write_bytes((snapshot / "config.json").read_bytes())
            (snapshot / "config.json").unlink()
            os.symlink(target, snapshot / "config.json")

        def rehash(snapshot, contract):
            rows = json.loads(Path(f"{snapshot}.integrity.json").read_text())["files"]
            rows[0]["actual_sha256"] = "0" * 64
            write_record(snapshot, contract, files=rows)

        changes = [lambda snapshot, contract: (snapshot / "tokenizer_config.json").unlink(),
                   lambda snapshot, contract: (snapshot / "added_tokens.json").write_text("{}"),
                   lambda snapshot, contract: (snapshot / SHARDS[1]).write_bytes(b"\2" * 33),
                   lambda snapshot, contract: (snapshot / SHARDS[0]).write_bytes(b"\3" * 64),
                   relink, rehash,
                   lambda snapshot, contract: write_record(snapshot, contract, ok_false=1),
                   lambda snapshot, contract: write_record(snapshot, contract, revision="0" * 40),
                   lambda snapshot, contract: write_record(snapshot, contract, local_files_not_in_scope=["notes.txt"]),
                   lambda snapshot, contract: Path(f"{snapshot}.integrity.json").unlink()]
        for index, change in enumerate(changes):
            with tempfile.TemporaryDirectory() as folder, self.subTest(index=index):
                snapshot, contract = write_snapshot(folder)
                change(snapshot, contract)
                with self.assertRaises((OSError, ValueError)):
                    check_record(snapshot, contract)
                    check_hashes(snapshot, contract)

    def test_changed_layers_hidden_vocabulary_window_or_stop_tokens_are_rejected(self):
        def text(**values):
            return lambda value: value["text_config"].update(values)

        changes = [("config.json", text(num_hidden_layers=59)), ("config.json", text(hidden_size=4096)),
                   ("config.json", text(vocab_size=256000)), ("config.json", text(sliding_window=512)),
                   ("config.json", text(attention_k_eq_v=1)),
                   ("config.json", lambda value: value["text_config"]["layer_types"].reverse()),
                   ("config.json", lambda value: value.update(eos_token_id=[1, 2])),
                   ("config.json", lambda value: value.update(architectures=["Gemma4ForCausalLM"])),
                   ("generation_config.json", lambda value: value.update(eos_token_id=[1, 106])),
                   ("generation_config.json", lambda value: value.update(bos_token_id=3)),
                   ("tokenizer_config.json", lambda value: value.update(eos_token="<pad>")),
                   ("tokenizer_config.json", lambda value: value.update(eot_token="<bos>")),
                   ("tokenizer.json", lambda value: value["model"]["vocab"].update(c=262144))]
        for index, (name, change) in enumerate(changes):
            with tempfile.TemporaryDirectory() as folder, self.subTest(index=index):
                snapshot, contract = write_snapshot(folder)
                edit_json(snapshot / name, change)
                with self.assertRaises(ValueError):
                    check_config(snapshot, contract)
                    check_special_tokens(snapshot, contract)


class SetupTests(unittest.TestCase):
    def test_prepare_requires_the_environment_built_from_the_current_inputs(self):
        recorded = {"environment_inputs": {name: sha256(RECIPE / name) for name in ("pyproject.toml", "uv.lock")}}
        self.assertEqual(check_setup(recorded, RECIPE), recorded)
        for setup in ({"environment_inputs": {**recorded["environment_inputs"], "uv.lock": "0" * 64}}, {}):
            with self.subTest(setup=setup), self.assertRaises(ValueError):
                check_setup(setup, RECIPE)

    def test_setup_checks_the_snapshot_before_creating_the_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder) / "workspace"
            with mock.patch.object(local, "command") as command, self.assertRaises((OSError, ValueError)):
                local.setup(workspace, local.recipe_path("gemma4_31b"), Path(folder) / "missing", Path(folder) / "lock")
            command.assert_not_called()
            self.assertFalse(workspace.exists())
            self.assertFalse((Path(folder) / "lock").exists())

    def run_setup(self, folder, steps, refuse=False):
        def command(*args, **kwargs):
            steps.append((*args, kwargs.get("env")))

        def check(path, revision):
            steps.append(("check", path, revision))
            if refuse:
                raise PipelineError("Existing exporter must be a clean checkout of the recipe's public revision")

        with mock.patch("Tools.Conversion.adapters.gemma4_31b.setup.check_record"), \
                mock.patch.object(local, "command", side_effect=command), \
                mock.patch.object(local, "check_checkout", side_effect=check):
            local.setup(Path(folder) / "workspace", local.recipe_path("gemma4_31b"), Path(folder) / "model",
                        Path(folder) / "lock")

    def test_setup_syncs_the_converter_lock_in_a_checkout_of_the_pinned_revision(self):
        revision, steps = load_contract(RECIPE / "model.json")["upstream_revision"], []
        with tempfile.TemporaryDirectory() as folder:
            upstream = Path(folder) / "workspace/upstream-gemma4-31b/coreai-models"
            self.run_setup(folder, steps)
        self.assertEqual(steps[2:], [("git", "clone", "--no-checkout", "--", local.UPSTREAM, upstream, None),
                                     ("git", "-C", upstream, "checkout", "--detach", revision, None),
                                     ("check", upstream, revision),
                                     ("uv", "sync", "--project", upstream, "--frozen", "--no-dev", "--package",
                                      "coreai-models", "--python", "3.11", {"UV_PROJECT_ENVIRONMENT": str(upstream / ".venv")}),
                                     ("check", upstream, revision)])

    def test_setup_refuses_a_converter_checkout_of_another_revision_or_with_edits(self):
        revision, steps = load_contract(RECIPE / "model.json")["upstream_revision"], []
        with tempfile.TemporaryDirectory() as folder:
            upstream = Path(folder) / "workspace/upstream-gemma4-31b/coreai-models"
            upstream.mkdir(parents=True)
            with self.assertRaises(PipelineError):
                self.run_setup(folder, steps, refuse=True)
            self.assertFalse((Path(folder) / "workspace/gemma4_31b/setup.json").exists())
        self.assertEqual(steps[2:], [("check", upstream, revision)])

    def test_plan_and_run_use_the_gemma_bindings_and_work_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            recipe = local.recipe_path("gemma4_31b")
            with self.assertRaises(PipelineError):
                local.pipeline_command("plan", recipe, workspace, workspace / "lock", False)
            (workspace / "gemma4_31b.paths.json").write_text("{}")
            command = local.pipeline_command("run", recipe, workspace, workspace / "lock", True)
            self.assertIn(str(workspace / "gemma4_31b.paths.json"), command)
            self.assertIn(str(workspace / "work/gemma4_31b"), command)
            self.assertIn("--refresh", command)


if __name__ == "__main__":
    unittest.main()
