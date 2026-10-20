import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from Tools.Conversion.adapters.qwen_image21.evaluation import REVISION, SIGMAS
from Tools.Conversion.adapters.qwen_image21.fixtures import quality_axes


class ImageReferenceFailureTests(unittest.TestCase):
    def test_load_and_input_failure_keep_all_unreviewed_quality_rows(self):
        import torch
        from diffusers import QwenImage21Pipeline
        from Tools.Conversion.adapters.qwen_image21 import suite_reference

        axes = quality_axes()
        runs = [{"run_id": key, "fixture_id": key.rsplit("-", 1)[0], "seed": int(key.rsplit("-", 1)[1])}
                for key in axes]
        first = runs[0]
        for failure in ("load", "input"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixtures = root / "fixtures"
                (fixtures / first["fixture_id"]).mkdir(parents=True)
                (fixtures / "fixtures.json").write_text(json.dumps({"runs": runs}))
                (fixtures / first["fixture_id"] / "input.json").write_text(json.dumps({
                    "prompt": "fixture", "reference": None,
                }))
                output = root / "output"
                args = SimpleNamespace(snapshot=root / REVISION, fixtures=fixtures, output=output,
                                       run_ids=first["run_id"], precision="fp32", vae_precision=None, device="cpu")
                pipeline = SimpleNamespace(
                    config=SimpleNamespace(sample_sigmas=SIGMAS), transformer=SimpleNamespace(dtype=torch.float32),
                    text_encoder=SimpleNamespace(dtype=torch.float32),
                    vae=SimpleNamespace(dtype=torch.float32, decode=lambda *args: None),
                    prepare_latents=lambda *args: None, set_progress_bar_config=lambda **kwargs: None,
                )
                pipeline.to = lambda device: pipeline
                with patch("Tools.Conversion.adapters.qwen_image21.contracts.require_heavy_job"), \
                        patch.object(suite_reference, "validate_fixture_suite"), \
                        patch.object(QwenImage21Pipeline, "from_pretrained", return_value=pipeline,
                                     side_effect=RuntimeError("load failed") if failure == "load" else None), \
                        patch("Tools.Conversion.adapters.qwen_image21.preprocessing.prompt_inputs",
                              side_effect=ValueError("input failed")):
                    with self.assertRaises(RuntimeError if failure == "load" else ValueError):
                        suite_reference.execute(args)
                summary = json.loads((output / "summary.json").read_text())
                ratings = json.loads((output / "quality-ratings.json").read_text())["ratings"]
                self.assertFalse(summary["complete"])
                self.assertEqual(len(ratings), 60)
                self.assertEqual({row["run_id"] for row in ratings}, set(axes))
                self.assertTrue(all(row["status"] == "unreviewed" for row in ratings))
                self.assertTrue(all(all(value is None for value in row["axes"].values()) for row in ratings))
                failed = next(row for row in ratings if row["run_id"] == first["run_id"])
                self.assertEqual(failed["reference_status"], "not_run" if failure == "load" else "failed")
