import copy
import json
from pathlib import Path
import unittest

import numpy as np

from Tools.Conversion.adapters.voxcpm2 import host, synthesis

CONTRACT = json.loads((Path(synthesis.__file__).resolve().parents[2] / "recipes/voxcpm2/model.json").read_text())


def contract():
    result = copy.deepcopy(CONTRACT)
    result["execution"].update(prefill_length=12, capacity=32)
    result["generation"].update(inference_timesteps=2, zero_init_steps=1, dit_calls_per_patch=1, min_len=0)
    return result


def reference():
    values = (np.arange(2, dtype=np.float32)[:, None, None] * 10000
              + np.arange(4, dtype=np.float32)[None, :, None] * 100
              + np.arange(64, dtype=np.float32)[None, None, :])
    return values.transpose(0, 2, 1).copy().transpose(0, 2, 1)


class ObservedAssets:
    def __init__(self):
        self.calls = []
        self.state_calls = []

    def state(self, asset, function):
        self.state_calls.append((asset, function))
        return {"tag": len(self.state_calls)}

    @staticmethod
    def embedding(ids):
        return ids.astype(np.float32)[:, :, None] + np.arange(2048, dtype=np.float32)[None, None, :] / np.float32(10000)

    @staticmethod
    def encoding(patch):
        return np.full((1, 2048), np.sum(patch, dtype=np.float32), dtype=np.float32) + np.arange(2048, dtype=np.float32)[None, :]

    def call(self, asset, function, values, state=None):
        self.calls.append((asset, function, {name: value.copy() for name, value in values.items()}, state))
        key = (asset, function)
        if key == ("tslm", "embed"):
            return {"embeds": self.embedding(values["ids"])}
        if key == ("tslm", "prefill"):
            return {"hidden": values["embeds"] + np.float32(5)}
        if key == ("ralm", "prefill"):
            return {"hidden": values["embeds"] + np.float32(13)}
        if key == ("proj", "prefill_fuse"):
            return {"residual_inputs": values["hidden"] + np.float32(11)}
        if key == ("proj", "fuse"):
            return {"lm_hidden": values["lm_raw"] + np.float32(101), "residual_input": values["lm_raw"] + values["curr_embed"] + np.float32(211)}
        if key == ("proj", "heads"):
            return {"dit_hidden": np.zeros((1, 2048), np.float32), "stop_logits": np.array([[0, 1]], np.float32)}
        if key == ("dit", "main"):
            return {"velocity": np.zeros((2, 64, 4), np.float32)}
        if key == ("locenc", "main"):
            if not values["patch"].flags.c_contiguous:
                raise AssertionError("locenc patch must be contiguous")
            return {"curr_embed": self.encoding(values["patch"])}
        if key == ("vae", "main"):
            return {"audio": np.full((1, 1, 7680), np.mean(values["latent"]), np.float32)}
        if key == ("tslm", "step"):
            return {"hidden": values["embed"] + np.float32(17)}
        if key == ("ralm", "step"):
            return {"hidden": values["embed"] + np.float32(19)}
        raise AssertionError(key)

    def inputs(self, asset, function, name):
        return [values[name] for a, f, values, _ in self.calls if (a, f) == (asset, function)]


class ReferenceHostTests(unittest.TestCase):
    def test_reference_rows_and_axes_follow_the_upstream_contract(self):
        ids = [0, 103, 104, 101]
        patches = reference()
        before = patches.copy()
        assets = ObservedAssets()
        result = synthesis.generate(assets, contract(), ids, 1708, limit=2, reference_patches=patches)
        expected_ids = [103, 0, 0, 104, 0, 103, 104, 101]
        self.assertEqual(result.ids, expected_ids)
        self.assertTrue(all(type(value) is int for value in result.ids))
        self.assertEqual(ids, [0, 103, 104, 101])
        np.testing.assert_array_equal(patches, before)
        padded = np.zeros((1, 12), np.int32)
        padded[0, :8] = expected_ids
        np.testing.assert_array_equal(assets.inputs("tslm", "embed", "ids")[0], padded)
        embeds = assets.embedding(padded)
        locenc = assets.inputs("locenc", "main", "patch")
        encoded = []
        for index in range(2):
            expected = np.zeros((1, 64, 4), np.float32)
            for channel in range(64):
                for frame in range(4):
                    expected[0, channel, frame] = 10000 * index + 100 * frame + channel
            np.testing.assert_array_equal(locenc[index], expected)
            encoded.append(assets.encoding(expected))
            embeds[:, index + 1, :] = encoded[-1]
        np.testing.assert_array_equal(assets.inputs("tslm", "prefill", "embeds")[0], embeds)
        np.testing.assert_array_equal(assets.inputs("tslm", "prefill", "length")[0], [8])
        hidden = embeds + np.float32(5)
        residual = hidden + np.float32(11)
        raw = assets.inputs("proj", "fuse", "lm_raw")
        current = assets.inputs("proj", "fuse", "curr_embed")
        for index in range(2):
            np.testing.assert_array_equal(raw[index], hidden[:, index + 1, :])
            np.testing.assert_array_equal(current[index], encoded[index])
            residual[:, index + 1, :] = hidden[:, index + 1, :] + encoded[index] + np.float32(211)
        np.testing.assert_array_equal(assets.inputs("ralm", "prefill", "embeds")[0], residual)
        np.testing.assert_array_equal(assets.inputs("proj", "heads", "lm_hidden")[0], hidden[:, 7, :])
        np.testing.assert_array_equal(assets.inputs("proj", "heads", "residual_hidden")[0], residual[:, 7, :] + np.float32(13))
        np.testing.assert_array_equal(assets.inputs("dit", "main", "cond")[0], np.zeros((2, 64, 4), np.float32))
        np.testing.assert_array_equal(assets.inputs("dit", "main", "cond")[1], np.concatenate([result.latents[0], result.latents[0]]))
        np.testing.assert_array_equal(assets.inputs("tslm", "step", "position")[0], [8])
        np.testing.assert_array_equal(assets.inputs("ralm", "step", "position")[0], [8])
        self.assertEqual(result.finish, "stop")
        self.assertEqual(len(result.audio), 2)
        self.assertEqual(result.waveform().size, 2 * 7680)
        self.assertEqual(len(locenc), 4)

    def test_patch_limit_uses_text_count_and_actual_prompt_length(self):
        generation = contract()["generation"]
        for capacity, expected in [(64, 22), (17, 10), (8, 1)]:
            self.assertEqual(host.patch_limit(2, capacity, generation, prompt_length=7), expected)
        for capacity in [7, 0, -1]:
            with self.assertRaises(ValueError):
                host.patch_limit(2, capacity, generation, prompt_length=7)
        assets = ObservedAssets()
        value = contract()
        value["execution"]["capacity"] = 17
        result = synthesis.generate(assets, value, [11, 22, 101], 1708, reference_patches=reference())
        self.assertEqual(result.limit, 10)

    def test_invalid_references_are_refused_before_state_and_calls(self):
        samples = [np.zeros((0, 4, 64), np.float32), np.zeros((1, 64, 4), np.float32), np.zeros((1, 4, 64), np.float64), np.full((1, 4, 64), np.nan, np.float32), np.full((1, 4, 64), np.inf, np.float32), np.zeros((64, 4, 64), np.float32)]
        for sample in samples:
            with self.subTest(shape=sample.shape, dtype=sample.dtype):
                assets = ObservedAssets()
                before = sample.tobytes()
                with self.assertRaises(ValueError):
                    synthesis.generate(assets, contract(), [11, 22, 101], 1708, reference_patches=sample)
                self.assertEqual(assets.calls, [])
                self.assertEqual(assets.state_calls, [])
                self.assertEqual(sample.tobytes(), before)

    def test_reference_prefill_exact_fit_succeeds_and_overflow_refuses_calls(self):
        value = contract()
        value["execution"]["prefill_length"] = 7
        result = synthesis.generate(ObservedAssets(), value, [11, 22, 101], 1708, limit=2, reference_patches=reference())
        self.assertEqual(len(result.ids), 7)
        value["execution"]["prefill_length"] = 6
        assets = ObservedAssets()
        with self.assertRaises(ValueError):
            synthesis.generate(assets, value, [11, 22, 101], 1708, reference_patches=reference())
        self.assertEqual(assets.calls, [])
        self.assertEqual(assets.state_calls, [])


    def test_contiguous_reference_and_numpy_ids_keep_the_public_contract(self):
        ids = np.array([11, 22, 101], dtype=np.int32)
        before = ids.copy()
        result = synthesis.generate(ObservedAssets(), contract(), ids, 1708, limit=2, reference_patches=np.ascontiguousarray(reference()))
        self.assertTrue(all(type(value) is int for value in result.ids))
        json.dumps(result.ids)
        np.testing.assert_array_equal(ids, before)
        self.assertEqual(result.finish, "stop")


    def test_variable_reference_counts_use_only_generated_audio(self):
        for count in [1, 32, 63]:
            value = contract()
            value["execution"].update(prefill_length=145, capacity=1024)
            patches = np.arange(count * 4 * 64, dtype=np.float32).reshape(count, 4, 64)
            assets = ObservedAssets()
            result = synthesis.generate(assets, value, [11, 22, 101], 1708, limit=2, reference_patches=patches)
            self.assertEqual(len(result.ids), count + 5)
            self.assertEqual(len(assets.inputs("locenc", "main", "patch")), count + 2)
            self.assertEqual(result.waveform().size, 2 * 7680)


if __name__ == "__main__":
    unittest.main()
