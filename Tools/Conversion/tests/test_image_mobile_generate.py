import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import diffusers
    import transformers.vision_utils
except ImportError as error:
    raise unittest.SkipTest(f"Qwen image dependencies unavailable: {error}") from error

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from Tools.Conversion.adapters.qwen_image21.evaluation import SIGMAS
from Tools.Conversion.adapters.qwen_image21.measurement import PhaseRecorder
from Tools.Conversion.adapters.qwen_image21.mobile_generate import generate
from Tools.Conversion.adapters.qwen_image21.mobile_profile import make_profile


class ImageMobileGenerateTests(unittest.TestCase):
    def test_rollout_uses_own_preceding_latent_target_positions_and_low_precision_step_boundaries(self):
        for candidate, dtype in (("w4", torch.float16), ("bf16", torch.bfloat16)):
            with self.subTest(candidate=candidate), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "vae").mkdir()
                configuration = {"latents_mean": [0.1234567] * 64, "latents_std": [1.234567] * 64}
                (root / "vae/config.json").write_text(json.dumps(configuration))
                profile = make_profile(candidate, "a" * 64, {"test.py": "b" * 64})
                target = np.arange(26, 1050)
                captured = []
                live = set()
                decoder_inputs = []

                class Component:
                    def __init__(self, directory, profile, kind):
                        if live:
                            raise AssertionError("A prior component was not released")
                        live.add(kind)
                        self.kind, self.calls, self.load_seconds = kind, [], 0

                    def call(self, values):
                        self.calls.append(0)
                        if self.kind == "text":
                            return (np.zeros((1, 512, 4096), np.float32),)
                        if self.kind == "denoiser":
                            captured.append(values)
                            result = np.full((1, 2304, 64), 99, np.float32)
                            result[:, target] = np.float32(0.1234567)
                            return (result,)
                        decoder_inputs.append(values["value"])
                        return (np.zeros((1, 4, 1, 512, 512), np.float32),)

                    def close(self):
                        live.remove(self.kind)

                def make_scheduler():
                    scheduler = FlowMatchEulerDiscreteScheduler(shift=1)
                    scheduler.set_timesteps(sigmas=list(SIGMAS), device="cpu")
                    scheduler.set_begin_index(0)
                    return scheduler, scheduler.timesteps

                def physical(latents, text, timestep, structure):
                    return {"latents": latents.copy(), "text": text.copy(), "timestep": timestep}

                torch.manual_seed(216)
                noise = torch.randn(1, 1024, 64).numpy()
                encoded = {"input_ids": np.ones((1, 40), np.int64)}
                metadata = {"drop": 14, "operation": "generate", "image_token_id": 151655}
                with patch("Tools.Conversion.adapters.qwen_image21.mobile_generate.scheduler", side_effect=lambda snapshot: make_scheduler()), \
                     patch("Tools.Conversion.adapters.qwen_image21.mobile_generate.text_inputs", return_value={}), \
                     patch("Tools.Conversion.adapters.qwen_image21.mobile_generate.denoiser_layout", return_value={"target_indices": target}), \
                     patch("Tools.Conversion.adapters.qwen_image21.mobile_generate.denoiser_inputs", side_effect=physical), \
                     patch("Tools.Conversion.adapters.qwen_image21.mobile_generate.PhaseRecorder", side_effect=lambda: PhaseRecorder(capture=lambda: {})):
                    actual, _ = generate({kind: root for kind in profile["components"]}, profile, root, root / "result",
                                         encoded, metadata, noise, component_type=Component)
                scheduler, times = make_scheduler()
                sample = torch.from_numpy(noise).to(dtype)
                predicted = torch.full(sample.shape, 0.1234567).to(dtype)
                for index, timestep in enumerate(times):
                    np.testing.assert_array_equal(captured[index]["latents"], sample.float().numpy())
                    self.assertEqual(captured[index]["timestep"], float(timestep.to(dtype) / 1000))
                    sample = scheduler.step(predicted, timestep, sample, return_dict=False)[0].to(dtype)
                np.testing.assert_array_equal(actual, sample.float().numpy())
                vae_dtype = torch.float32 if candidate == "w4" else dtype
                mean = torch.tensor(configuration["latents_mean"]).reshape(1, 64, 1, 1, 1).to(vae_dtype)
                std = torch.tensor(configuration["latents_std"]).reshape(1, 64, 1, 1, 1).to(vae_dtype)
                expected_decoder = sample.transpose(1, 2).reshape(1, 64, 1, 32, 32).to(vae_dtype) * std + mean
                np.testing.assert_array_equal(decoder_inputs[0], expected_decoder.float().numpy())
                self.assertFalse(live)


if __name__ == "__main__":
    unittest.main()
