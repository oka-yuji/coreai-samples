import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from Tools.Conversion import local, summary
from Tools.Conversion.storage import PipelineError


class MaintenanceTests(unittest.TestCase):
    def test_shared_summary_drops_free_text_paths_arguments_environment_and_case_names(self):
        secret = "private-example-sentinel"
        report = {
            "version": 1, "status": "completed", "verification": "passed", "elapsed_s": 2.0,
            "recipe": secret, "host": {"environment": {"TOKEN": secret}}, "error": secret,
            "stages": [{"id": secret, "role": "verify", "status": "reused", "execute_s": 0,
                        "artifacts": secret, "command": [secret], "origin": secret}],
        }
        comparison = {"passed": True, "cases": [{"id": secret, "positions": 4, "top1_matches": 4,
                       "near_tie_mismatches": 0, "significant_mismatches": 0, "max_probability_error": 0.001}]}
        result = summary.project(report, comparison)
        self.assertNotIn(secret, json.dumps(result))
        self.assertEqual(result["stages"][0]["status"], "reused")
        self.assertEqual(result["comparison"]["positions"], 4)

    def test_shared_summary_preserves_failure_and_missing_verification(self):
        for status, verification in (("failed", "failed"), ("cancelled", "incomplete"), ("completed", "not_configured")):
            result = summary.project({"version": 1, "status": status, "verification": verification, "stages": []})
            self.assertEqual(result["status"], status)
            self.assertEqual(result["verification"], verification)
            self.assertNotIn("elapsed_s", result)

    def test_shared_summary_rejects_non_numeric_metrics_and_arbitrary_status_text(self):
        original = {"version": 1, "status": "completed", "verification": "passed", "stages": []}
        for value in (True, "private-example-sentinel", float("nan"), float("inf"), -1):
            with self.subTest(value=value), self.assertRaises(PipelineError):
                summary.project({**original, "elapsed_s": value})
        with self.assertRaises(PipelineError):
            summary.project({**original, "status": "/private/example"})

    def test_local_command_requires_setup_and_passes_paths_without_a_shell(self):
        with tempfile.TemporaryDirectory(prefix="conversion space ") as folder:
            workspace = Path(folder)
            recipe = local.recipe_path("qwen3_06b")
            with self.assertRaises(PipelineError):
                local.pipeline_command("run", recipe, workspace, workspace / "lock", False)
            (workspace / "qwen3_06b.paths.json").write_text("{}")
            command = local.pipeline_command("run", recipe, workspace, workspace / "lock", True)
            self.assertIn(str(workspace / "qwen3_06b.paths.json"), command)
            self.assertIn(str(workspace / "lock"), command)
            self.assertIn("--refresh", command)
            plan = local.pipeline_command("plan", recipe, workspace, workspace / "lock", False)
            self.assertNotIn("--work-dir", plan)
            with self.assertRaises(PipelineError):
                local.pipeline_command("plan", recipe, workspace, workspace / "lock", True)

    def test_recipe_selection_cannot_escape_the_maintained_recipes(self):
        for name in ("../qwen3_06b", "/tmp/qwen3_06b", "unknown"):
            with self.assertRaises(PipelineError):
                local.recipe_path(name)

    def test_mobile_candidates_keep_separate_recipes_without_changing_legacy_defaults(self):
        for name in ("qwen3_06b", "qwen_image21", "embeddinggemma2"):
            self.assertEqual(local.recipe_path(name), local.recipe_path(name, "fp16_safe"))
            with self.assertRaises(PipelineError):
                local.recipe_path(name, "w4")
        self.assertEqual(local.recipe_path("embeddinggemma2", "int4_int8").name, "recipe-int4-int8.json")
        for candidate in ("fp16", "bf16", "mixed", "w8", "w4"):
            path = local.recipe_path("qwen_image21_mobile", candidate)
            self.assertEqual(path.parent.name, "qwen_image21_mobile")
            expected = "recipe.json" if candidate == "w4" else f"recipe-{candidate}.json"
            self.assertEqual(path.name, expected)
        self.assertEqual(local.recipe_path("qwen_image21_mobile"), local.recipe_path("qwen_image21_mobile", "w4"))
        for unsupported in ("fp16_safe", "int4_int8"):
            with self.assertRaises(PipelineError):
                local.recipe_path("qwen_image21_mobile", unsupported)

    def test_setup_does_not_reset_a_modified_or_wrong_checkout(self):
        revision = "a" * 40
        for origin, head, dirty in ((local.UPSTREAM, revision, " M python/export.py"),
                                    ("https://example.invalid/repo.git", revision, ""),
                                    (local.UPSTREAM, "b" * 40, "")):
            with tempfile.TemporaryDirectory() as folder:
                results = [subprocess.CompletedProcess([], 0, stdout=value) for value in (origin, head, dirty)]
                with mock.patch.object(local, "command", side_effect=results) as command:
                    with self.assertRaises(PipelineError):
                        local.check_checkout(Path(folder), revision)
                    self.assertEqual(command.call_count, 3)

    def test_setup_commands_cannot_inherit_external_environment_targets(self):
        supplied = {"PATH": "/usr/bin", "HOME": "/test-home", "UV_PROJECT_ENVIRONMENT": "/outside/venv",
                    "VIRTUAL_ENV": "/outside/venv", "PYTHONPATH": "/outside/packages",
                    "UV_CONFIG_FILE": "/outside/uv.toml", "GIT_CONFIG_COUNT": "1"}
        with mock.patch.dict(local.os.environ, supplied, clear=True), \
                mock.patch.object(local.subprocess, "run") as run:
            local.command("uv", "sync", env={"UV_PROJECT_ENVIRONMENT": "/workspace/upstream/.venv"})
            environment = run.call_args.kwargs["env"]
            self.assertEqual(environment["UV_PROJECT_ENVIRONMENT"], "/workspace/upstream/.venv")
            self.assertEqual(environment["HOME"], "/test-home")
            for key in ("VIRTUAL_ENV", "PYTHONPATH", "UV_CONFIG_FILE", "GIT_CONFIG_COUNT"):
                self.assertNotIn(key, environment)

    def test_setup_requires_the_downloaded_snapshot_before_writing_a_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder) / "workspace"
            with mock.patch.object(local, "command") as command:
                with self.assertRaises(PipelineError):
                    local.setup(workspace, local.recipe_path("qwen3_06b"), Path(folder) / "missing")
                command.assert_not_called()
            self.assertFalse(workspace.exists())

    def test_setup_rejects_build_environment_and_ancestor_symlinks_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace, outside = root / "workspace", root / "outside"
            upstream = workspace / "upstream/coreai-models"
            upstream.mkdir(parents=True)
            outside.mkdir()
            local.check_workspace_paths(workspace, upstream)
            for name in (".build", ".venv"):
                path = upstream / name
                path.symlink_to(outside, target_is_directory=True)
                with self.assertRaises(PipelineError):
                    local.check_workspace_paths(workspace, upstream)
                path.unlink()
            upstream.rmdir()
            upstream.parent.rmdir()
            upstream.parent.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(PipelineError):
                local.check_workspace_paths(workspace, upstream)

    def test_shared_summary_refuses_non_object_inputs(self):
        with self.assertRaises(PipelineError):
            summary.project([])

    def test_default_lock_is_shared_across_worktrees_and_can_be_overridden(self):
        with mock.patch.dict(local.os.environ, {}, clear=True):
            self.assertEqual(local.default_lock(), Path.home() / "Library/Caches/coreai-samples/heavy-job")
        with mock.patch.dict(local.os.environ, {"COREAI_HEAVY_JOB_LOCK": "/tmp/shared-model-lock"}):
            self.assertEqual(local.default_lock(), Path("/tmp/shared-model-lock"))


if __name__ == "__main__":
    unittest.main()
