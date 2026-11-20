import hashlib
import io
import json
import wave
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from Tools.Conversion.adapters.voxcpm2 import reference_encoder as module


class NormalizedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(1, 64, 1, stride=640, bias=False))

    def forward(self, pcm):
        mu = self.conv(pcm)
        return {"mu": mu, "logvar": mu + 1000}


def arguments(output):
    return ["--snapshot", "missing", "--upstream", "missing", "--contract", "missing", "--voices", "missing", "--output", str(output)]


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, value):
        mu = F.avg_pool1d(value, 640, 640).repeat(1, 64, 1) * self.scale
        return {"mu": mu, "logvar": mu + 1000}

class ReferenceEncoderTests(unittest.TestCase):
    def test_padding_boundaries(self):
        for size, patches in [(1, 1), (2559, 1), (2560, 1), (2561, 2), (80000, 32), (81919, 32), (81920, 32), (81921, 33), (160000, 63), (161280, 63)]:
            samples = np.arange(size, dtype=np.float32) / 200000
            before = samples.copy()
            padded, count = module.prepare_pcm(samples)
            self.assertEqual(padded.shape, (1, 1, 161280))
            self.assertEqual(padded.dtype, np.float32)
            self.assertEqual(count, patches)
            np.testing.assert_array_equal(padded[0, 0, :size], before)
            np.testing.assert_array_equal(samples, before)
            self.assertTrue(np.all(padded[0, 0, size:] == 0))
            self.assertTrue(padded.flags.c_contiguous)

    def test_rejects_invalid_pcm(self):
        for samples in [np.array([], dtype=np.float32), np.zeros(161281, dtype=np.float32), np.zeros((1, 4), dtype=np.float32), np.array([np.nan], dtype=np.float32), np.array([np.inf], dtype=np.float32), np.array([-np.inf], dtype=np.float32), np.array([1], dtype=np.int32)]:
            with self.subTest(shape=samples.shape, dtype=samples.dtype):
                with self.assertRaises(ValueError):
                    module.prepare_pcm(samples)

    def test_axis_order(self):
        mu = (np.arange(64, dtype=np.float32)[:, None] * 10000 + np.arange(252, dtype=np.float32)[None, :])[None, :, :]
        for patches in [1, 32, 63]:
            actual = module.patches_from_mu(mu, patches)
            self.assertEqual(actual.shape, (patches, 4, 64))
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(actual.flags.c_contiguous)
            for patch in range(patches):
                for frame in range(4):
                    for channel in range(64):
                        self.assertEqual(actual[patch, frame, channel], 10000 * channel + 4 * patch + frame)

    def test_rejects_invalid_mu(self):
        good = np.zeros((1, 64, 252), dtype=np.float32)
        for patches in [0, -1, 64, 1.5, True]:
            with self.assertRaises(ValueError):
                module.patches_from_mu(good, patches)
        for mu in [np.zeros((1, 252, 64), dtype=np.float32), np.zeros((64, 252), dtype=np.float32), np.full((1, 64, 252), np.nan, dtype=np.float32), np.full((1, 64, 252), np.inf, dtype=np.float32), np.zeros((1, 64, 252), dtype=np.int32)]:
            with self.assertRaises(ValueError):
                module.patches_from_mu(mu, 1)

    def test_wrapper_is_independent_and_uses_mu(self):
        source = FakeEncoder().eval()
        wrapper = module.ReferenceEncoder(SimpleNamespace(encoder=source)).eval()
        samples = torch.arange(161280, dtype=torch.float32).reshape(1, 1, -1) / 200000
        expected = source(samples)["mu"].detach().clone()
        original = source.state_dict()["scale"].clone()
        torch.testing.assert_close(wrapper(samples), expected, rtol=0, atol=0)
        torch.testing.assert_close(source.state_dict()["scale"], original, rtol=0, atol=0)
        with torch.no_grad():
            source.scale.fill_(9.0)
        torch.testing.assert_close(wrapper(samples), expected, rtol=0, atol=0)
        for patches in [1, 32, 63]:
            exact = samples[:, :, :patches * 2560]
            torch.testing.assert_close(wrapper(exact), wrapper(samples)[:, :, :patches * 4], rtol=0, atol=0)

    def test_weight_norm_copy_does_not_mutate_or_share_the_original(self):
        source = NormalizedEncoder().eval()
        pcm = torch.ones(1, 1, 2560)
        expected = source(pcm)["mu"].detach().clone()
        state = {name: value.detach().clone() for name, value in source.state_dict().items()}
        hooks = tuple(source.conv._forward_pre_hooks.values())
        wrapper = module.ReferenceEncoder(SimpleNamespace(encoder=source))
        self.assertEqual(tuple(source.conv._forward_pre_hooks.values()), hooks)
        self.assertEqual(list(source.state_dict()), list(state))
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        self.assertFalse({value.data_ptr() for value in source.parameters()} & {value.data_ptr() for value in wrapper.parameters()})
        torch.testing.assert_close(wrapper(pcm), expected, rtol=0, atol=0)
        with torch.no_grad():
            source.conv.weight_v.add_(1)
        torch.testing.assert_close(wrapper(pcm), expected, rtol=0, atol=0)

    def test_nonfinite_outputs_fail_the_numeric_gate(self):
        reference = np.zeros((1, 4, 64), dtype=np.float32)
        for value in [np.nan, np.inf, -np.inf]:
            mu = np.full((1, 64, 252), value, dtype=np.float32)
            prefix, result = module.measure(reference, mu, 1, 0.001)
            self.assertIsNone(prefix)
            self.assertFalse(result["passed"])
            self.assertEqual(result["nonfinite"], 64 * 252)

    def test_existing_output_and_symlink_are_refused_before_model_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "existing"
            existing.mkdir()
            marker = existing / "marker"
            marker.write_bytes(b"keep")
            link = root / "link"
            link.symlink_to(existing, target_is_directory=True)
            dangling = root / "dangling"
            dangling.symlink_to(root / "absent", target_is_directory=True)
            for output in [existing, link, dangling]:
                with mock.patch.object(module, "run") as run:
                    with self.assertRaises(FileExistsError):
                        module.main(arguments(output))
                    run.assert_not_called()
            self.assertEqual(marker.read_bytes(), b"keep")

    def test_failure_keeps_an_incomplete_report_without_local_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            error = ValueError("Cannot use " + str(output / "asset"))
            with mock.patch.object(module, "run", side_effect=error):
                with self.assertRaises(ValueError):
                    module.main(arguments(output))
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["passed"])
            self.assertFalse(report["complete"])
            self.assertEqual(report["error"]["type"], "ValueError")
            self.assertNotIn(str(output), report["error"]["message"])


    def test_contract_identity_is_fixed_before_voices_or_models(self):
        original = json.loads((module.REPOSITORY / "Tools/Conversion/recipes/voxcpm2/model.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = root / "contract.json"
            for field in ["revision", "upstream_revision", "files"]:
                wrong = json.loads(json.dumps(original))
                if field == "files":
                    wrong["files"]["audiovae.pth"]["sha256"] = "0" * 64
                else:
                    wrong[field] = "0" * 40
                contract.write_text(json.dumps(wrong))
                args = arguments(root / field)
                args[args.index("--contract") + 1] = str(contract)
                with mock.patch.object(module, "runtime_check", return_value={}), mock.patch.object(module, "prepare", side_effect=AssertionError("Wrong contract reached voice preparation")), mock.patch.object(module, "load_model") as load:
                    with self.assertRaisesRegex(ValueError, "pinned"):
                        module.main(args)
                    load.assert_not_called()
            contract.write_text(json.dumps(original, indent=4))
            self.assertEqual(module.fixed_contract(contract), original)

    def test_prepared_cases_keep_the_audio_duration_separate_from_timings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            voices, output = root / "voices", root / "result"
            voices.mkdir()
            output.mkdir()
            readme = b"Synthetic test fixture"
            (voices / "README.md").write_bytes(readme)
            data = io.BytesIO()
            with wave.open(data, "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(24000)
                stream.writeframes(np.zeros(240000, dtype="<i2").tobytes())
            raw = data.getvalue()
            (voices / "voice.wav").write_bytes(raw)
            manifest = {"readme_sha256": hashlib.sha256(readme).hexdigest(), "voices": [{"id": "fixture", "path": "voice.wav", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "pcm_f32le_sha256": {str(seconds): hashlib.sha256(np.zeros(seconds * 16000, dtype="<f4").tobytes()).hexdigest() for seconds in [5, 10]}}]}
            with mock.patch.object(module, "canonical_pcm", side_effect=lambda pcm: np.zeros(pcm.size * 2 // 3, dtype=np.float32)):
                cases, _ = module.prepare(manifest, voices, output)
            self.assertEqual([case["row"]["audio_seconds"] for case in cases], [5, 10])
            self.assertEqual([case["row"]["patches"] for case in cases], [32, 63])
            self.assertTrue(all("seconds" not in case["row"] for case in cases))


if __name__ == "__main__":
    unittest.main()
