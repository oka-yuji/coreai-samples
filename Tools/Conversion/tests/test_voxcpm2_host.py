import base64
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

from Tools.Conversion.adapters.voxcpm2.host import Solver, patch_limit, patch_step, prompt_ids, time_span

BASE = Path(__file__).resolve().parents[3]

REFERENCE = json.loads((BASE / "CoreAISamplesKit/Tests/VoxCPMCoreTests/Resources/reference.json").read_text(encoding="utf-8"))
CONTRACT = json.loads((BASE / "Tools/Conversion/recipes/voxcpm2/model.json").read_text(encoding="utf-8"))
GENERATION = CONTRACT["generation"]
AUDIO_START = CONTRACT["execution"]["audio_start_token"]
INPUT_LIMIT = 1e-6
LATENT_LIMIT = 1e-5


def shapes(asset, function, kind):
    return {name: tuple(shape) for name, _, shape in CONTRACT["assets"][asset]["functions"][function][kind]}


DIT_INPUTS = shapes("dit", "main", "inputs")
VELOCITY = shapes("dit", "main", "outputs")["velocity"]
PATCH = shapes("locenc", "main", "inputs")["patch"]
HIDDEN = shapes("proj", "heads", "outputs")["dit_hidden"]


def floats(text, shape):
    data = base64.b64decode(text, validate=True)
    if len(data) % 4:
        raise ValueError("A float array must hold whole little-endian binary32 values")
    return np.frombuffer(data, dtype="<f4").astype(np.float32).reshape(shape)


def maximum_normalized_error(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape or not np.all(np.isfinite(actual)):
        return math.inf
    return float(np.abs(actual - reference).max()) / max(float(np.abs(reference).max()), 1e-8)


def vocabulary():
    return {entry["id"]: entry["token"] for entry in REFERENCE["vocabulary"]}


def solver_patch(number):
    return next(patch for patch in REFERENCE["solver"] if patch["patch"] == number)


def solver_inputs(patch):
    noise = floats(REFERENCE["noise"][patch["patch"]], PATCH)
    return noise, floats(patch["ditHidden"], HIDDEN), floats(patch["condition"], PATCH)


def stop_logits():
    return [floats(text, (1, 2)) for text in REFERENCE["stopLogits"]]


class HostTestCase(unittest.TestCase):
    def setUp(self):
        settings = REFERENCE["source"]["settings"]
        self.assertEqual(REFERENCE["source"]["upstream"]["commit"], CONTRACT["upstream_revision"])
        self.assertEqual(REFERENCE["source"]["weights"]["revision"], CONTRACT["revision"])
        self.assertEqual(settings["inferenceTimesteps"], GENERATION["inference_timesteps"])
        self.assertEqual(settings["cfgValue"], GENERATION["cfg_value"])
        self.assertEqual(settings["swaySamplingCoefficient"], GENERATION["sway_sampling_coef"])
        self.assertEqual(settings["zeroInitSteps"], GENERATION["zero_init_steps"])
        self.assertEqual(settings["minLen"], GENERATION["min_len"])
        self.assertEqual(settings["temperature"], 1.0)
        self.assertTrue(GENERATION["cfg_zero_star"])
        self.assertEqual(GENERATION["dit_calls_per_patch"], GENERATION["inference_timesteps"] - GENERATION["zero_init_steps"])

    def assertBits(self, actual, expected, msg=None):
        actual = np.asarray(actual)
        self.assertEqual(actual.dtype, np.dtype(np.float32), msg)
        self.assertEqual(actual.shape, expected.shape, msg)
        self.assertTrue(np.array_equal(np.ascontiguousarray(actual).view(np.uint32), expected.view(np.uint32)), msg)

    def assertClose(self, actual, expected, limit, msg=None):
        actual = np.asarray(actual)
        self.assertEqual(actual.dtype, np.dtype(np.float32), msg)
        self.assertEqual(actual.shape, expected.shape, msg)
        self.assertLessEqual(maximum_normalized_error(actual, expected), limit, msg)


class VoxCPM2HostPromptTests(HostTestCase):
    def test_fixture_sentences_give_the_original_ids(self):
        self.assertEqual(len(REFERENCE["prompts"]), 4)
        for prompt in REFERENCE["prompts"]:
            with self.subTest(prompt=prompt["name"]):
                ids = prompt_ids(prompt["tokenIDs"], vocabulary(), AUDIO_START)
                self.assertEqual([int(value) for value in ids], prompt["expectedIDs"])

    def test_ids_or_characters_missing_from_the_vocabulary_are_rejected(self):
        tokens = vocabulary()
        word = next(identifier for identifier, token in sorted(tokens.items())
                    if len(token) >= 2 and all(0x4E00 <= ord(character) <= 0x9FFF for character in token))
        without_character = {identifier: token for identifier, token in tokens.items() if token != tokens[word][0]}
        with self.assertRaises(ValueError):
            prompt_ids([word], without_character, AUDIO_START)
        with self.assertRaises(ValueError):
            prompt_ids([max(tokens) + 1], tokens, AUDIO_START)


class VoxCPM2HostPatchLimitTests(HostTestCase):
    def test_limit_follows_the_contract_formula(self):
        rows = [(1024, 28, 178), (1024, 144, 874), (1024, 145, 878), (1024, 1022, 1), (64, 7, 52), (64, 8, 55),
                (sys.maxsize, sys.maxsize // 2, 4096)]
        for capacity, text_tokens, limit in rows:
            with self.subTest(capacity=capacity, text_tokens=text_tokens):
                self.assertEqual(patch_limit(text_tokens, capacity, GENERATION), limit)

    def test_texts_without_room_are_rejected(self):
        rows = [(1024, -1), (1024, 1023), (1024, 4096), (64, 63), (1024, sys.maxsize), (-sys.maxsize - 1, 0)]
        for capacity, text_tokens in rows:
            with self.subTest(capacity=capacity, text_tokens=text_tokens):
                with self.assertRaises(ValueError):
                    patch_limit(text_tokens, capacity, GENERATION)


class VoxCPM2HostTimeSpanTests(HostTestCase):
    def test_time_span_matches_the_fixture_bit_for_bit(self):
        expected = floats(REFERENCE["timeSpan"], (GENERATION["inference_timesteps"] + 1,))
        self.assertBits(time_span(GENERATION), expected)


class VoxCPM2HostSolverTests(HostTestCase):
    def test_first_step_uses_no_estimator_output(self):
        patch = solver_patch(0)
        noise, hidden, cond = solver_inputs(patch)
        times = floats(REFERENCE["timeSpan"], (GENERATION["inference_timesteps"] + 1,))
        solver = Solver(noise, hidden, cond, GENERATION)
        self.assertBits(solver.latent, noise)
        first = solver.inputs()
        self.assertIsNotNone(first)
        self.assertEqual(set(first), set(DIT_INPUTS))
        self.assertBits(first["x"], np.concatenate([noise, noise]))
        self.assertBits(first["t"], np.full(DIT_INPUTS["t"], times[GENERATION["zero_init_steps"]], dtype=np.float32))
        requests = 0
        for call in patch["calls"]:
            if solver.inputs() is None:
                break
            requests += 1
            solver.advance(floats(call["velocity"], VELOCITY))
        self.assertEqual(requests, GENERATION["dit_calls_per_patch"])
        self.assertIsNone(solver.inputs())

    def test_estimator_inputs_match_the_fixture_calls(self):
        for patch in REFERENCE["solver"]:
            with self.subTest(patch=patch["patch"]):
                noise, hidden, cond = solver_inputs(patch)
                solver = Solver(noise, hidden, cond, GENERATION)
                for index, call in enumerate(patch["calls"]):
                    inputs = solver.inputs()
                    message = f"call {index}"
                    self.assertIsNotNone(inputs, message)
                    self.assertEqual(set(inputs), set(DIT_INPUTS), message)
                    expected_x = floats(call["x"], DIT_INPUTS["x"])
                    if index == 0:
                        self.assertBits(inputs["x"], expected_x, message)
                    else:
                        self.assertClose(inputs["x"], expected_x, INPUT_LIMIT, message)
                    self.assertBits(inputs["mu"], np.concatenate([hidden, np.zeros_like(hidden)]), message)
                    self.assertBits(inputs["t"], floats(call["t"], DIT_INPUTS["t"]), message)
                    self.assertBits(inputs["cond"], np.concatenate([cond, cond]), message)
                    self.assertBits(inputs["dt"], floats(call["dt"], DIT_INPUTS["dt"]), message)
                    solver.advance(floats(call["velocity"], VELOCITY))

    def test_latent_matches_the_fixture(self):
        for patch in REFERENCE["solver"]:
            with self.subTest(patch=patch["patch"]):
                noise, hidden, cond = solver_inputs(patch)
                solver = Solver(noise, hidden, cond, GENERATION)
                for call in patch["calls"]:
                    self.assertIsNotNone(solver.inputs())
                    solver.advance(floats(call["velocity"], VELOCITY))
                self.assertIsNone(solver.inputs())
                self.assertClose(solver.latent, floats(patch["latent"], PATCH), LATENT_LIMIT)

    def test_wrong_shapes_and_extra_velocities_are_rejected(self):
        noise = np.zeros(PATCH, dtype=np.float32)
        hidden = np.zeros(HIDDEN, dtype=np.float32)
        velocity = np.zeros(VELOCITY, dtype=np.float32)
        for arguments in ((noise[..., :-1], hidden, noise), (noise, hidden[..., :-1], noise), (noise, hidden, noise[..., :-1])):
            with self.subTest(shapes=[value.shape for value in arguments]):
                with self.assertRaises(ValueError):
                    Solver(*arguments, GENERATION)
        solver = Solver(noise, hidden, noise, GENERATION)
        with self.assertRaises(ValueError):
            solver.advance(velocity[..., :-1])
        advanced = 0
        for _ in range(GENERATION["dit_calls_per_patch"] + 1):
            try:
                solver.advance(velocity)
            except ValueError:
                break
            advanced += 1
        self.assertEqual(advanced, GENERATION["dit_calls_per_patch"])


class VoxCPM2HostPatchStepTests(HostTestCase):
    def test_fixture_stop_logits_stop_at_patch_33(self):
        logits = stop_logits()
        self.assertEqual(len(logits), 34)
        steps = [patch_step(index, 178, value, GENERATION) for index, value in enumerate(logits)]
        self.assertEqual(steps, ["next"] * 33 + ["stopped"])

    def test_limit_is_reported_apart_from_a_stop(self):
        logits = stop_logits()
        steps = [patch_step(index, 10, logits[index], GENERATION) for index in range(10)]
        self.assertEqual(steps, ["next"] * 9 + ["limit"])
        self.assertEqual(patch_step(33, 34, logits[33], GENERATION), "stopped")

    def test_stops_up_to_min_len_and_ties_are_ignored(self):
        stop = np.array([[0.0, 1.0]], dtype=np.float32)
        steps = [patch_step(index, 10, stop, GENERATION) for index in range(GENERATION["min_len"] + 2)]
        self.assertEqual(steps, ["next"] * (GENERATION["min_len"] + 1) + ["stopped"])
        self.assertEqual(patch_step(GENERATION["min_len"] + 1, 10, np.array([[0.5, 0.5]], dtype=np.float32), GENERATION), "next")

    def test_wrong_logit_counts_and_indices_are_rejected(self):
        cases = [(3, 10, np.zeros((1, 1), dtype=np.float32)), (3, 10, np.zeros((1, 3), dtype=np.float32)),
                 (10, 10, np.zeros((1, 2), dtype=np.float32)), (-1, 10, np.zeros((1, 2), dtype=np.float32))]
        for index, limit, logits in cases:
            with self.subTest(index=index, limit=limit, count=logits.size):
                with self.assertRaises(ValueError):
                    patch_step(index, limit, logits, GENERATION)


if __name__ == "__main__":
    unittest.main()
