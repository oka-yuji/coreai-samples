from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from Tools.Conversion.adapters.qwen_image21.mobile_capture import NonFiniteSourceError, SourceTrace
from Tools.Conversion.adapters.qwen_image21.mobile_profile import make_profile


class DoubleValue(torch.nn.Module):
    def forward(self, value):
        return value * 2


class LanguageFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = DoubleValue()

    def forward(self, value):
        return self.norm(value)


class EncoderFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = LanguageFixture()
        self.model.visual = torch.nn.Identity()

    def forward(self, value):
        return self.model.language_model(value)


class TransformerFixture(torch.nn.Module):
    def forward(self, **values):
        return (values["hidden_states"] * 0.1,)


class VAEFixture:
    def encode(self, value):
        self.last_image = value.clone()
        latent = torch.ones(1, 64, 1, 32, 32, dtype=value.dtype)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(self, value, **kwargs):
        return (torch.zeros(1, 4, 1, 512, 512, dtype=value.dtype),)


class SchedulerFixture:
    def step(self, noise, timestep, sample, **kwargs):
        return (sample - noise * 0.25,)


class PipelineFixture:
    def __init__(self):
        self.text_encoder = EncoderFixture()
        self.transformer = TransformerFixture()
        self.vae = VAEFixture()
        self.scheduler = SchedulerFixture()

    def prepare_latents(self, images, batch, channels, height, width, dtype, device, generator, latents=None):
        return torch.ones(1, 1024, 64, dtype=dtype), None

    def _encode_vae_image(self, value, generator):
        return self.vae.encode(value).latent_dist.mode()

    def _pack_latents(self, value, batch, channels, height, width):
        return value.reshape(batch, channels, height * width).transpose(1, 2)


class ImageMobileCaptureTests(unittest.TestCase):
    def profile(self):
        return make_profile("w4", "a" * 64, {"capture.py": "b" * 64})

    def test_all_eight_post_boundary_inputs_and_original_pre_norm_values_are_preserved(self):
        pipeline = PipelineFixture()
        original = pipeline.transformer.forward
        neutral = pipeline.text_encoder.model.language_model.norm.register_forward_hook(lambda module, args, value: args[0])
        with tempfile.TemporaryDirectory() as folder:
            with SourceTrace(pipeline, self.profile()) as trace:
                directory = Path(folder) / "generate"
                trace.begin(directory, "generate")
                text = torch.full((1, 2, 4), 1.234567, dtype=torch.float32)
                torch.testing.assert_close(pipeline.text_encoder(text), text, atol=0, rtol=0)
                sample, _ = pipeline.prepare_latents(None, 1, 64, 512, 512, torch.float32, "cpu", None)
                for index in range(8):
                    timestep = torch.tensor(1 - index * 0.1)
                    noise, = pipeline.transformer(hidden_states=sample, encoder_hidden_states=text, timestep=timestep,
                                                   img_shapes=[[(1, 32, 32)]], img_mask=torch.ones(1, 256, dtype=torch.bool),
                                                   kv_cache=None, kv_cache_mode=None)
                    sample, = pipeline.scheduler.step(noise, timestep, sample, return_dict=False)
                    values = {"latents": sample}
                    self.assertIs(trace.callback(pipeline, index, timestep, values), values)
                pipeline.vae.decode(sample.float().reshape(1, 64, 1, 32, 32), return_dict=False)
                record = trace.finish()
                self.assertEqual(len(record["steps"]), 8)
                self.assertEqual(record["tensors"]["language-reference.npy"]["original_dtype"], "torch.float32")
                np.testing.assert_array_equal(np.load(directory / "language-reference.npy"), text.numpy())
                with np.load(directory / "transformer-0-inputs.npz") as first, np.load(directory / "transformer-7-inputs.npz") as last:
                    np.testing.assert_array_equal(first["encoder_hidden_states"], text.half().float().numpy())
                    self.assertFalse(np.array_equal(first["hidden_states"], last["hidden_states"]))
                self.assertEqual(record["not_applicable"], [
                    {"component": name, "reason": "generation has no reference image"} for name in ("vision", "vae_encoder")])
            self.assertEqual(pipeline.transformer.forward, original)
        neutral.remove()

    def test_mixed_prepare_rounds_latents_without_rounding_the_float32_vae_image(self):
        pipeline = PipelineFixture()
        with tempfile.TemporaryDirectory() as folder, SourceTrace(pipeline, self.profile()) as trace:
            trace.begin(Path(folder) / "edit", "edit")
            image = torch.full((1, 4, 1, 512, 512), 0.1234567)
            target, condition = pipeline.prepare_latents([image], 1, 64, 512, 512, torch.float32, "cpu", None)
            self.assertEqual(target.dtype, torch.float16)
            self.assertEqual(condition.dtype, torch.float16)
            torch.testing.assert_close(pipeline.vae.last_image, image, atol=0, rtol=0)
            with self.assertRaises(ValueError):
                trace.finish()

    def test_nonfinite_output_is_retained_and_never_marked_complete(self):
        pipeline = PipelineFixture()
        with tempfile.TemporaryDirectory() as folder, SourceTrace(pipeline, self.profile()) as trace:
            directory = Path(folder) / "failed"
            trace.begin(directory, "generate")
            with self.assertRaises(NonFiniteSourceError):
                trace.output("bad-output", torch.tensor([1.0, float("nan"), float("inf")]))
            record = trace.snapshot()
            self.assertFalse(record["complete"])
            self.assertEqual(record["tensors"]["bad-output.npy"]["nonfinite_elements"], 2)
            self.assertEqual(np.count_nonzero(~np.isfinite(np.load(directory / "bad-output.npy"))), 2)
            with self.assertRaises(ValueError):
                trace.finish()

    def test_original_encoder_input_must_match_the_frozen_tensor_before_forward(self):
        pipeline = PipelineFixture()
        with tempfile.TemporaryDirectory() as folder, SourceTrace(pipeline, self.profile()) as trace:
            trace.begin(Path(folder) / "mismatch", "generate", {"input_ids": np.array([[1, 2]], dtype=np.int64)})
            with self.assertRaises(ValueError):
                trace.inputs("encoder", (), {"input_ids": torch.tensor([[1, 3]])})

    def test_exit_releases_the_bound_pipeline_methods_even_after_failure(self):
        import gc
        import weakref
        pipeline = PipelineFixture()
        reference = weakref.ref(pipeline)
        trace = SourceTrace(pipeline, self.profile())
        with self.assertRaises(RuntimeError), trace:
            raise RuntimeError("inference failed")
        del pipeline
        gc.collect()
        self.assertIsNone(reference())

    def test_failed_directory_creation_does_not_publish_uninitialized_trace_state(self):
        trace = SourceTrace(PipelineFixture(), self.profile())
        with patch.object(Path, "mkdir", side_effect=OSError("No space left")), self.assertRaises(OSError):
            trace.begin(Path("unused"), "generate")
        self.assertIsNone(trace.directory)


if __name__ == "__main__":
    unittest.main()
