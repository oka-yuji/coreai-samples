from contextlib import contextmanager
from itertools import chain
from pathlib import Path
import builtins
import runpy
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLVisionModel
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
except ImportError as error:
    DEPENDENCY_ERROR = str(error)
else:
    DEPENDENCY_ERROR = None

if DEPENDENCY_ERROR is None:
    from Tools.Conversion.adapters.qwen_image21 import encoders


@unittest.skipIf(DEPENDENCY_ERROR is not None, f"Qwen image encoder dependencies unavailable: {DEPENDENCY_ERROR}")
class ImageEncoderDependencyTests(unittest.TestCase):
    def test_missing_vision_utilities_skips_optional_encoder_tests(self):
        original_import = builtins.__import__

        def without_vision_utilities(name, *args, **kwargs):
            if name == "transformers.vision_utils":
                raise ModuleNotFoundError("No module named 'transformers.vision_utils'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_vision_utilities):
            namespace = runpy.run_path(str(Path(__file__)))
        self.assertIn("transformers.vision_utils", namespace["DEPENDENCY_ERROR"])
        self.assertTrue(namespace["ImageEncoderTests"].__unittest_skip__)


@unittest.skipIf(DEPENDENCY_ERROR is not None, f"Qwen image encoder dependencies unavailable: {DEPENDENCY_ERROR}")
class ImageEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261018)
            text_config = Qwen3VLTextConfig(
                hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=64,
                rope_parameters={
                    "rope_type": "default", "rope_theta": 5000000,
                    "mrope_section": [2, 1, 1], "mrope_interleaved": True,
                },
            )
            vision_config = Qwen3VLVisionConfig(
                hidden_size=32, intermediate_size=64, depth=4, num_heads=4,
                out_hidden_size=32, deepstack_visual_indexes=[0, 2, 3],
            )
            text_config._attn_implementation = "sdpa"
            vision_config._attn_implementation = "sdpa"
            self.config = Qwen3VLConfig(text_config=text_config, vision_config=vision_config)
            self.text = Qwen3VLTextModel(text_config).eval()
            self.vision = Qwen3VLVisionModel(vision_config).eval()
            self.ids = torch.randint(0, 64, (1, 24), dtype=torch.int32)
            self.visual_mask = torch.zeros(1, 24, dtype=torch.bool)
            self.visual_mask[:, [3, 4, 10, 11]] = True
            self.visual = torch.randn(1, 24, 32)
            self.deepstack = torch.zeros(3, 1, 24, 32)
            for index in range(3):
                self.deepstack[index][self.visual_mask] = torch.randn(4, 32) * (index + 1)
            self.pixels = torch.randn(1024, 1536)
        self.positions = torch.arange(24)[None, None].expand(3, 1, -1)
        self.mask = torch.ones(1, 1, 24, 24, dtype=torch.bool).tril()
        self.grid = torch.tensor([[1, 32, 32]], dtype=torch.int64)

    def assert_numerically_equal(self, reference, actual):
        self.assertEqual(actual.shape, reference.shape)
        self.assertTrue(torch.isfinite(reference).all().item())
        self.assertTrue(torch.isfinite(actual).all().item())
        difference = reference - actual
        maximum = difference.abs().max() / reference.abs().max().clamp_min(1e-12)
        rmse = difference.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-12)
        self.assertLessEqual(maximum.item(), 1e-3)
        self.assertLessEqual(rmse.item(), 1e-4)

    def text_reference(self):
        captured = []
        hook = self.text.norm.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone())
        )
        try:
            with torch.no_grad():
                hidden = self.text.embed_tokens(self.ids).clone()
                hidden[self.visual_mask] = self.visual[self.visual_mask]
                output = self.text(
                    inputs_embeds=hidden,
                    attention_mask=self.mask,
                    position_ids=self.positions,
                    use_cache=False,
                    visual_pos_masks=self.visual_mask,
                    deepstack_visual_embeds=[
                        self.deepstack[index][self.visual_mask] for index in range(3)
                    ],
                ).last_hidden_state
                cosine, sine = self.text.rotary_emb(hidden, self.positions)
        finally:
            hook.remove()
        self.assertEqual(len(captured), 1)
        return captured[0], output, cosine, sine

    def test_text_matches_source_visual_injection_and_excludes_final_norm(self):
        reference, normalized, cosine, sine = self.text_reference()
        with torch.no_grad():
            actual = encoders.TextEncoder(self.text).eval()(
                self.ids, self.visual, self.deepstack, cosine, sine, self.mask, self.visual_mask
            )
        self.assert_numerically_equal(reference, actual)
        self.assertFalse(torch.allclose(normalized, actual, rtol=1e-3, atol=1e-4))

    def test_text_without_an_image_ignores_dense_visual_values(self):
        self.visual_mask.zero_()
        self.deepstack.zero_()
        self.visual.add_(100)
        reference, _, cosine, sine = self.text_reference()
        with torch.no_grad():
            actual = encoders.TextEncoder(self.text).eval()(
                self.ids, self.visual, self.deepstack, cosine, sine, self.mask, self.visual_mask
            )
        self.assert_numerically_equal(reference, actual)

    def test_vision_matches_source_merge_and_three_nonconsecutive_taps(self):
        with torch.no_grad():
            original = self.vision(self.pixels, self.grid)
            actual = encoders.VisionEncoder(self.vision).eval()(self.pixels)
        self.assertEqual(len(original.deepstack_features), 3)
        self.assertEqual(len(actual), 4)
        references = (original.pooler_output, *original.deepstack_features)
        for index, (reference, output) in enumerate(zip(references, actual, strict=True)):
            with self.subTest(output=index):
                self.assertEqual(tuple(output.shape), (256, 32))
                self.assert_numerically_equal(reference, output)

    def snapshot_state(self):
        return {
            **{
                "model.language_model." + name: value.bfloat16()
                for name, value in self.text.state_dict().items()
            },
            **{
                "model.visual." + name: value.bfloat16()
                for name, value in self.vision.state_dict().items()
            },
            "lm_head.weight": torch.zeros(64, 32, dtype=torch.bfloat16),
        }

    def write_snapshot(self, root, state):
        directory = root / "text_encoder"
        directory.mkdir()
        self.config.save_pretrained(directory)
        save_file(state, directory / "model.safetensors")

    def test_loader_restores_nonpersistent_buffers_and_reads_only_selected_weights(self):
        state = self.snapshot_state()
        accessed = []

        @contextmanager
        def recording_source(*args, **kwargs):
            with safe_open(*args, **kwargs) as source:
                class Reader:
                    def keys(self):
                        return source.keys()

                    def get_tensor(self, key):
                        accessed.append(key)
                        return source.get_tensor(key)

                yield Reader()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_snapshot(root, state)
            for kind, prefix in (("text", "model.language_model."), ("vision", "model.visual.")):
                with self.subTest(kind=kind), patch.object(encoders, "safe_open", recording_source):
                    accessed.clear()
                    model = encoders.load_encoder(root, kind)
                    self.assertEqual(set(accessed), {name for name in state if name.startswith(prefix)})
                    self.assertFalse(model.training)
                    for _, value in chain(model.named_parameters(), model.named_buffers()):
                        self.assertEqual(value.device.type, "cpu")
                        self.assertEqual(value.dtype, torch.float32)
                        self.assertFalse(value.is_meta)
                    for name, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, state[prefix + name].float()))
                    rotary = model.rotary_emb if kind == "text" else model.rotary_pos_emb
                    self.assertTrue(torch.equal(rotary.inv_freq, rotary.original_inv_freq))
                    original = self.text.rotary_emb if kind == "text" else self.vision.rotary_pos_emb
                    torch.testing.assert_close(rotary.inv_freq, original.inv_freq, rtol=0, atol=0)
                    with torch.no_grad():
                        if kind == "text":
                            output = model(input_ids=self.ids, use_cache=False).last_hidden_state
                        else:
                            output = model(self.pixels, self.grid).pooler_output
                    self.assertTrue(torch.isfinite(output).all().item())

    def test_loader_rejects_an_incomplete_component(self):
        for kind, prefix in (("text", "model.language_model."), ("vision", "model.visual.")):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                state = self.snapshot_state()
                missing = next(name for name in state if name.startswith(prefix))
                del state[missing]
                root = Path(directory)
                self.write_snapshot(root, state)
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    encoders.load_encoder(root, kind)


if __name__ == "__main__":
    unittest.main()
