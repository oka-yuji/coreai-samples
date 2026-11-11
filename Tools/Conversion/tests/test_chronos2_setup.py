import copy
from pathlib import Path
import sys
import tomllib
import unittest
from unittest.mock import patch

from Tools.Conversion import local
from Tools.Conversion.adapters.chronos2 import prepare, setup
from Tools.Conversion.recipe import load_recipe, source_paths
from Tools.Conversion.storage import digest, file_hash, read_json, tree_hash, write_json
from Tools.Conversion.tests.test_chronos2_support import ChronosFiles, RECIPE, REPO


class SetupTests(ChronosFiles, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.environment = {"python": "3.11.11", "binary": {"bytes": 10, "sha256": "a" * 64}, "packages": {}}
        self.toolchain = {"xcode": "Xcode 27.0", "macos_sdk": "27.0", "coreai_build_sha256": "b" * 64,
                          "system": "Darwin", "release": "26.0.0", "version": "test", "machine": "arm64"}
        self.stamp = self.root / "setup.json"
        self.record = {"format": "coreai-samples-chronos2-setup/1",
                       "environment_inputs": setup.environment_inputs(self.recipe),
                       "sources": setup.source_inputs(self.recipe), "environment": self.environment,
                       "toolchain": self.toolchain}
        write_json(self.stamp, self.record)

    def check(self):
        with patch.object(setup, "installed_environment", return_value=self.environment), \
                patch.object(setup, "toolchain_identity", return_value=self.toolchain):
            return setup.check_setup(self.recipe, self.stamp)

    def test_setup_stamp_invalidates_changed_lock_requirements_sources_and_toolchain(self):
        self.assertEqual(self.check(), self.record)
        for name in ("pyproject.toml", "uv.lock", "NOTICE", "model.json", "fixtures.json", "recipe.json"):
            with self.subTest(name=name):
                path = self.recipe / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaises(ValueError):
                    self.check()
                path.write_bytes(original)
        for name in ("sources", "toolchain", "environment"):
            with self.subTest(name=name):
                record = copy.deepcopy(self.record)
                record[name] = {}
                write_json(self.stamp, record)
                with self.assertRaises(ValueError):
                    self.check()

    def test_direct_dependency_versions_are_enforced(self):
        dependencies = dict(value.split("==") for value in tomllib.loads((RECIPE / "pyproject.toml").read_text())
                            ["project"]["dependencies"])
        with patch.object(setup.importlib.metadata, "version", side_effect=dependencies.__getitem__):
            observed = setup.installed_environment(RECIPE)
        self.assertEqual(observed["packages"], dependencies)
        for name in dependencies:
            with self.subTest(name=name):
                altered = {**dependencies, name: "0.0.0"}
                with patch.object(setup.importlib.metadata, "version", side_effect=altered.__getitem__), \
                        self.assertRaises(ValueError):
                    setup.installed_environment(RECIPE)

    def test_prepare_records_current_source_and_environment_without_inference(self):
        with patch.object(setup, "installed_environment", return_value=self.environment), \
                patch.object(setup, "toolchain_identity", return_value=self.toolchain):
            record = prepare.prepare(self.model_path, self.fixture_path, self.snapshot, self.stamp, self.root / "out")
        self.assertTrue(record["ready"])
        self.assertEqual(record["source_files"], self.contract["files"])
        self.assertEqual(record["setup"], self.record)
        self.assertEqual(record["identity"], self.identity)

    def fake_command(self, *args, **kwargs):
        if args[0] == "uv":
            self.assertIn("--frozen", args)
            self.assertIn("--no-dev", args)
            self.assertEqual(args[args.index("--project") + 1], self.recipe)
            environment = Path(kwargs["env"]["UV_PROJECT_ENVIRONMENT"])
            self.assertEqual(environment, self.root / "workspace/chronos2/.venv")
            (environment / "bin").mkdir(parents=True)
            (environment / "bin/python").write_bytes(b"test interpreter")
        else:
            self.assertEqual(args[1:3], ("-m", "Tools.Conversion.adapters.chronos2.setup"))
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            write_json(args[args.index("--probe") + 1], self.environment)

    def test_setup_uses_dedicated_environment_and_publishes_bindings_after_checks(self):
        with patch.object(local, "command", side_effect=self.fake_command) as command, \
                patch.object(setup, "toolchain_identity", return_value=self.toolchain):
            binding_path = setup.setup(self.root / "workspace", self.recipe / "recipe.json", self.snapshot,
                                       self.root / "heavy-job")
        bindings = read_json(binding_path)
        self.assertEqual(bindings, {"python": str(self.root / "workspace/chronos2/.venv/bin/python"),
                                    "snapshot": str(self.snapshot),
                                    "setup_stamp": str(self.root / "workspace/chronos2/setup.json")})
        self.assertEqual(read_json(bindings["setup_stamp"]), self.record)
        self.assertEqual(command.call_count, 2)
        self.assertFalse((self.root / "heavy-job").exists())

    def test_setup_changed_during_sync_publishes_no_stamp_or_bindings(self):
        def changed(*args, **kwargs):
            self.fake_command(*args, **kwargs)
            if args[0] == "uv":
                with (self.recipe / "uv.lock").open("a") as handle:
                    handle.write("\n")
        with patch.object(local, "command", side_effect=changed), \
                patch.object(setup, "toolchain_identity", return_value=self.toolchain), self.assertRaises(ValueError):
            setup.setup(self.root / "workspace", self.recipe / "recipe.json", self.snapshot, self.root / "heavy-job")
        self.assertFalse((self.root / "workspace/chronos2.paths.json").exists())
        self.assertFalse((self.root / "workspace/chronos2/setup.json").exists())

    def test_setup_rejects_symlink_profile_stamp_and_bindings_before_commands(self):
        workspace = self.root / "workspace"
        (workspace / "chronos2").mkdir(parents=True)
        target = self.root / "external"
        target.mkdir()
        for name in ("chronos2/.venv", "chronos2/setup.json", "chronos2.paths.json"):
            with self.subTest(name=name):
                link = workspace / name
                link.symlink_to(target, target_is_directory=True)
                with patch.object(local, "command") as command, self.assertRaises(ValueError):
                    setup.setup(workspace, self.recipe / "recipe.json", self.snapshot, self.root / "heavy-job")
                command.assert_not_called()
                link.unlink()


class RoutingTests(ChronosFiles, unittest.TestCase):
    def test_default_and_explicit_fp32_route_without_affecting_other_recipes(self):
        for precision in (None, "fp32"):
            self.assertEqual(local.recipe_path("chronos2", precision), RECIPE / "recipe.json")
        for precision in ("fp16_safe", "int4_int8", "ios", "w4", "bf16", ""):
            with self.subTest(precision=precision), self.assertRaises(ValueError):
                local.recipe_path("chronos2", precision)
        self.assertEqual(local.recipe_path("embeddinggemma2").name, "recipe.json")
        self.assertEqual(local.recipe_path("embeddinggemma2", "int4_int8").name, "recipe-int4-int8.json")
        with self.assertRaises(ValueError):
            local.recipe_path("qwen3_06b", "fp32")

    def test_setup_bypasses_qwen_contract_and_upstream_exporter(self):
        with patch.object(setup, "setup", return_value="bindings") as dedicated, \
                patch.object(local, "load_contract") as qwen, patch.object(local, "build_upstream") as upstream:
            result = local.setup(self.root, RECIPE / "recipe.json", self.snapshot, self.root / "lock")
        self.assertEqual(result, "bindings")
        dedicated.assert_called_once_with(self.root, RECIPE / "recipe.json", self.snapshot, self.root / "lock")
        qwen.assert_not_called()
        upstream.assert_not_called()

    def test_recipe_dependencies_locking_and_inputs_cover_the_five_stage_workflow(self):
        stamp = self.root / "setup.json"
        write_json(stamp, {})
        recipe = load_recipe(RECIPE / "recipe.json", {"python": sys.executable, "snapshot": str(self.snapshot),
                                                      "setup_stamp": str(stamp)}, REPO)
        self.assertEqual([stage["id"] for stage in recipe["steps"]],
                         ["prepare", "reference", "export", "verify", "bundle"])
        for index, stage in enumerate(recipe["steps"]):
            with self.subTest(stage=stage["id"]):
                self.assertTrue(stage["heavy"])
                self.assertEqual(stage["depends"], [prior["id"] for prior in recipe["steps"][:index]])
                paths = source_paths(stage, recipe["variables"])
                for required in (RECIPE, REPO / "Tools/Conversion/adapters/chronos2", stamp,
                                 REPO / "Tools/Conversion/storage.py", REPO / "Tools/Conversion/runner.py",
                                 REPO / "Tools/Conversion/local.py", REPO / "Tools/Conversion/adapters/pinned.py"):
                    self.assertIn(required, paths)
                self.assertTrue(all(not Path(output).is_absolute() for output in stage["outputs"]))
                if stage["id"] != "verify":
                    self.assertIn(self.snapshot, paths)
        self.assertFalse(recipe["steps"][0]["cache"])
        checks = recipe["steps"][3]["checks"]
        self.assertIn({"file": "comparison.json", "key": ["passed"], "equals": True}, checks)
        self.assertIn({"file": "comparison.json", "key": ["elements"], "equals": 6048}, checks)

    def test_environment_and_fixture_changes_change_inventory_identities(self):
        before = digest(tree_hash(self.recipe))
        for name in ("uv.lock", "fixtures.json", "NOTICE"):
            path = self.recipe / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            self.assertNotEqual(digest(tree_hash(self.recipe)), before)
            path.write_bytes(original)
        self.assertEqual(digest(tree_hash(self.recipe)), before)


if __name__ == "__main__":
    unittest.main()
