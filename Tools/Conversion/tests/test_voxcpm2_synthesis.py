import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from Tools.Conversion.adapters.voxcpm2.source import load_contract
from Tools.Conversion.adapters.voxcpm2.synthesis import (Alignment, Assets, GenerationError, Result, Tokenizer, Trace, describe,
                                                         fixture_verdict, generate, limit_verdict, load_result, overall_verdict,
                                                         pcm16, replacement_summary, save_result, write_listening, write_wav)
from Tools.Conversion.storage import file_hash

RECIPE = Path(__file__).resolve().parents[1] / "recipes" / "voxcpm2"
CONTRACT = load_contract(RECIPE / "model.json")
IDS = [11, 12, 13, 14, 15, CONTRACT["execution"]["audio_start_token"]]
LIMIT = CONTRACT["criteria"]["normalized_max_limit"]


class FakeAssets:
    def __init__(self, stop_at=4, fuse=None, failing=None):
        self.calls, self.states = [], []
        self.stop_at, self.fuse, self.failing = stop_at, fuse, failing
        self.heads = 0

    def state(self, asset, function):
        state = {"asset": asset, "serial": len(self.states)}
        self.states.append(state)
        return state

    def call(self, asset, function, values, state=None):
        self.calls.append((asset, function, {name: np.array(value) for name, value in values.items()}, state))
        if (asset, function) == self.failing:
            raise GenerationError({"reason": "nonfinite_output", "call": f"{asset}.{function}"})
        outputs = {name: np.zeros(shape, np.float32) for name, _, shape in CONTRACT["assets"][asset]["functions"][function]["outputs"]}
        if (asset, function) == ("proj", "heads"):
            outputs["stop_logits"][0] = (0.0, 1.0) if self.heads >= self.stop_at else (1.0, 0.0)
            self.heads += 1
        if (asset, function) == ("proj", "fuse") and self.fuse is not None:
            outputs = self.fuse(sum(call[:2] == ("proj", "fuse") for call in self.calls) - 1)
        return outputs

    def inputs(self, asset, function, name):
        return [values[name] for called_asset, called_function, values, _ in self.calls
                if (called_asset, called_function) == (asset, function)]


class FakeReference:
    def __init__(self, arrays):
        self.arrays = arrays

    def load(self, name):
        return self.arrays[name]


def described(patches=4, reference_patches=4, limit=10, finish="stop", latent_shift=None, audio_shift=0.0, replaced=()):
    generator = np.random.default_rng(11)
    latents = generator.normal(size=(reference_patches, 1, 64, 4)).astype(np.float32)
    audio = generator.normal(size=(reference_patches, 1, 1, 7680)).astype(np.float32)
    stops = generator.normal(size=(reference_patches, 1, 2)).astype(np.float32)
    result = Result(IDS, limit)
    for index in range(patches):
        latent = latents[index] + np.float32(latent_shift[1] if latent_shift and latent_shift[0] == index else 0.0)
        result.noise.append(np.zeros((1, 64, 4), np.float32))
        result.latents.append(latent)
        result.stop_logits.append(stops[index])
        result.audio.append(audio[index] + np.float32(audio_shift))
    result.finish = finish
    reference = FakeReference({"patch": latents, "stop_logits": stops, "stream_audio": audio})
    return describe(result, reference, LIMIT, CONTRACT["architecture"]["sample_rate"], replaced)


class FakeComponents:
    def __init__(self, value):
        self.value, self.seen = value, []

    def call(self, asset, function, values, state=None):
        self.seen.append(values)
        return {"hidden": np.full((1, 2048), self.value, np.float16)}


class VoxCPM2SynthesisRuleTests(unittest.TestCase):
    def test_replaced_patches_count_once_and_are_limited_by_the_reference_patches(self):
        settings = {"per_sentence": 1, "total_fraction": 0.02}
        both = [{"patch": 9, "kind": "fuse"}, {"patch": 9, "kind": "latent"}]
        summary = replacement_summary([both, [{"patch": 30, "kind": "fuse"}], []], 239, settings)
        self.assertEqual((summary["replaced_patches"], summary["kinds"], summary["within_limit"]), (2, {"fuse": 2, "latent": 1}, True))
        four = [[{"patch": index, "kind": "latent"}] for index in range(4)]
        self.assertTrue(replacement_summary(four, 239, settings)["within_limit"])
        self.assertFalse(replacement_summary([*four, [{"patch": 1, "kind": "fuse"}]], 239, settings)["within_limit"])

    def test_assets_cast_inputs_by_precision_and_reject_non_finite_values(self):
        assets = Assets.__new__(Assets)
        assets.precisions, assets.loaded = {"tslm": "fp16"}, {"fp16": FakeComponents(0.5)}
        values = {"embed": np.ones((1, 2048), np.float32), "position": np.array([7], np.int32)}
        out = assets.call("tslm", "step", values)
        seen = assets.loaded["fp16"].seen[0]
        self.assertEqual((seen["embed"].dtype, seen["position"].dtype, out["hidden"].dtype), (np.float16, np.int32, np.float32))
        with self.assertRaises(GenerationError) as raised:
            assets.call("tslm", "step", {**values, "embed": np.full((1, 2048), np.inf, np.float32)})
        self.assertEqual((raised.exception.details["reason"], len(assets.loaded["fp16"].seen)), ("nonfinite_input", 1))
        assets.loaded["fp16"] = FakeComponents(np.nan)
        with self.assertRaises(GenerationError) as raised:
            assets.call("tslm", "step", values)
        self.assertEqual(raised.exception.details["reason"], "nonfinite_output")

    @unittest.skipUnless(importlib.util.find_spec("tokenizers"), "tokenizers is required")
    def test_the_tokenizer_file_must_match_the_contract(self):
        from tokenizers import Tokenizer as Backend
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            backend = Backend(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            backend.save(str(path))
            contract = copy.deepcopy(CONTRACT)
            contract["files"]["tokenizer.json"] = file_hash(path)
            self.assertEqual(Tokenizer(path, contract)("hello \n world"), [1, 2, contract["execution"]["audio_start_token"]])
            contract["files"]["tokenizer.json"] = {**contract["files"]["tokenizer.json"], "sha256": "0" * 64}
            with self.assertRaises(ValueError):
                Tokenizer(path, contract)

    def test_worker_results_round_trip_and_keep_audio_only_for_stopped_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            for finish, patches in (("stop", 3), ("limit", 3), ("error", 0)):
                result = Result(IDS, 5)
                for index in range(patches):
                    result.noise.append(np.full((1, 64, 4), index, np.float32))
                    result.latents.append(np.full((1, 64, 4), -index, np.float32))
                    result.stop_logits.append(np.array([[index, 0.5]], np.float32))
                    result.audio.append(np.full((1, 1, 7680), index / 10, np.float32))
                result.finish, result.error, result.seconds = finish, {"reason": "test"} if finish == "error" else None, 1.5
                folder = Path(directory) / finish
                folder.mkdir()
                save_result(result, folder, {"positions": [], "fuse_calls": patches})
                loaded, info = load_result(folder)
                with self.subTest(finish=finish):
                    self.assertEqual((loaded.finish, loaded.error, loaded.patches, loaded.ids, loaded.limit, info["fuse_calls"]),
                                     (finish, result.error, patches, IDS, 5, patches))
                    for name in ("noise", "latents", "stop_logits"):
                        np.testing.assert_array_equal(np.array(getattr(loaded, name)), np.array(getattr(result, name)))
                    if finish == "stop":
                        np.testing.assert_array_equal(loaded.waveform(), result.waveform())
                    else:
                        with np.load(folder / "result.npz") as arrays:
                            self.assertNotIn("audio", arrays.files)
                        with self.assertRaises(ValueError):
                            loaded.waveform()

    def test_assets_reject_inputs_that_overflow_the_asset_precision(self):
        assets = Assets.__new__(Assets)
        assets.precisions, assets.loaded = {"tslm": "fp16"}, {"fp16": FakeComponents(0.5)}
        with np.errstate(over="ignore"), self.assertRaises(GenerationError) as raised:
            assets.call("tslm", "step", {"embed": np.full((1, 2048), 1e5, np.float32), "position": np.array([7], np.int32)})
        details = raised.exception.details
        self.assertEqual((details["reason"], details["precision"], details["names"]), ("nonfinite_input", "fp16", ["embed"]))
        self.assertEqual(assets.loaded["fp16"].seen, [])

    def test_a_complete_generation_passes_only_with_every_condition(self):
        self.assertTrue(described()["passed"])
        cases = {"audio outside the criterion": {"audio_shift": 0.5}, "another stop patch": {"patches": 3},
                 "an end other than stop": {"finish": "limit"}, "a latent outside the criterion": {"latent_shift": (2, 0.5)},
                 "an original that reached its patch limit": {"limit": 4}}
        for name, change in cases.items():
            with self.subTest(name):
                self.assertFalse(described(**change)["passed"])

    def test_a_replaced_latent_is_excluded_after_its_original_error_is_recorded(self):
        description = described(latent_shift=(2, 0.5), replaced=(2,))
        self.assertTrue(description["passed"])
        self.assertGreater(description["latent_errors"][2], LIMIT)
        self.assertEqual(description["first_latent_out_of_criterion"], 2)
        self.assertFalse(described(latent_shift=(2, 0.5), replaced=(1,))["passed"])
        self.assertTrue(described()["reference_natural_stop"])
        self.assertFalse(described(limit=4)["reference_natural_stop"])

    def test_token_ids_and_noise_must_equal_the_original(self):
        noise = [np.full((1, 64, 4), index, np.float32) for index in range(3)]
        good = {"passed": True}
        self.assertTrue(fixture_verdict(IDS, np.array(IDS, np.int32), noise, np.stack(noise), good)["passed"])
        changed = [*noise[:2], np.nextafter(noise[2], np.float32(3))]
        for name, arguments in {"token IDs": (IDS[:-1] + [0], np.array(IDS, np.int32), noise, np.stack(noise), good),
                                "noise": (IDS, np.array(IDS, np.int32), changed, np.stack(noise), good),
                                "generation": (IDS, np.array(IDS, np.int32), noise, np.stack(noise), {"passed": False})}.items():
            with self.subTest(name):
                self.assertFalse(fixture_verdict(*arguments)["passed"])

    def test_the_limit_check_requires_an_unfinished_generation_without_audio(self):
        result = Result(IDS, 8)
        result.latents, result.finish = [np.zeros((1, 64, 4), np.float32)] * 8, "limit"
        self.assertTrue(limit_verdict(result, 8, 15, False))
        self.assertFalse(limit_verdict(result, 8, 15, True))
        self.assertFalse(limit_verdict(result, 8, 8, False))
        result.finish = "stop"
        self.assertFalse(limit_verdict(result, 8, 15, False))

    def test_the_stage_passes_only_when_every_part_passes(self):
        fixtures = {"a": {"passed": True}, "b": {"passed": True}}
        self.assertTrue(overall_verdict(fixtures, {"within_limit": True}, {"passed": True}))
        self.assertFalse(overall_verdict({**fixtures, "b": {"passed": False}}, {"within_limit": True}, {"passed": True}))
        self.assertFalse(overall_verdict(fixtures, {"within_limit": False}, {"passed": True}))
        self.assertFalse(overall_verdict(fixtures, {"within_limit": True}, {"passed": False}))
        self.assertFalse(overall_verdict({}, {"within_limit": True}, {"passed": True}))

    def test_listening_files_are_written_in_label_order_with_one_time(self):
        labels = {"one": {"A": "fp16", "B": "fp32", "C": "mixed"}, "two": {"A": "fp32", "B": "mixed", "C": "fp16"}}
        keys = [(sentence, seed, name) for sentence in labels for seed in (1, 2) for name in ("fp32", "fp16", "mixed")]
        takes = {key: np.full(480, index / 100, np.float32) for index, key in enumerate(keys)}
        with tempfile.TemporaryDirectory() as directory:
            paths = write_listening(Path(directory), takes, labels, 1, CONTRACT["architecture"]["sample_rate"])
            names = [path.relative_to(directory).as_posix() for path in paths]
            self.assertEqual(names[:4], ["listen/one/A.wav", "listen/one/B.wav", "listen/one/C.wav", "listen-extra/2/one/A.wav"])
            self.assertEqual(len(names), 12)
            self.assertEqual(len({path.stat().st_mtime_ns for path in paths}), 1)
            with wave.open(str(paths[0]), "rb") as handle:
                frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
            np.testing.assert_array_equal(frames, pcm16(takes["one", 1, "fp16"])[0])

    def test_samples_beyond_full_scale_are_counted_when_rounded_to_16_bits(self):
        samples, clipped = pcm16([1.5, -1.5, 0.25, 1.0, -1.0])
        self.assertEqual((samples.tolist(), clipped), ([32767, -32768, 8192, 32767, -32767], 2))


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for the solver noise")
class VoxCPM2SynthesisHostTests(unittest.TestCase):
    def test_steps_use_the_prompt_length_plus_the_patch_index(self):
        assets = FakeAssets(stop_at=4)
        result = generate(assets, CONTRACT, IDS, seed=3)
        self.assertEqual((result.finish, result.patches, result.stopped_patch), ("stop", 5, 4))
        length = len(IDS)
        for asset in ("tslm", "ralm"):
            self.assertEqual([int(value[0]) for value in assets.inputs(asset, "step", "position")], [length + i for i in range(4)])
            self.assertEqual([int(value[0]) for value in assets.inputs(asset, "prefill", "length")], [length])
        ids = assets.inputs("tslm", "embed", "ids")[0]
        self.assertEqual(ids[0, :length].tolist(), IDS)
        self.assertTrue(np.all(ids[0, length:] == CONTRACT["execution"]["pad_token"]))
        self.assertEqual(len(assets.inputs("dit", "main", "x")), 5 * CONTRACT["generation"]["dit_calls_per_patch"])
        self.assertEqual(generate(FakeAssets(stop_at=0), CONTRACT, IDS, seed=3).stopped_patch, 3)

    def test_each_request_starts_from_new_states(self):
        assets = FakeAssets(stop_at=3)
        generate(assets, CONTRACT, IDS, seed=1)
        first = len(assets.calls)
        generate(assets, CONTRACT, IDS, seed=1)
        for calls in (assets.calls[:first], assets.calls[first:]):
            used = {}
            for asset, function, _, state in calls:
                if state is not None:
                    used.setdefault(asset, set()).add(state["serial"])
            self.assertEqual(sorted(len(serials) for serials in used.values()), [1, 1, 1])
        serials = [state["serial"] for _, _, _, state in assets.calls if state is not None]
        self.assertTrue(set(serials[:3]).isdisjoint(serials[-3:]))
        self.assertEqual(len(assets.states), 6)

    def test_reaching_the_limit_is_not_a_success_and_writes_no_audio(self):
        result = generate(FakeAssets(stop_at=100), CONTRACT, IDS, seed=1, limit=5)
        self.assertEqual((result.finish, result.patches, result.succeeded, result.stopped_patch), ("limit", 5, False, None))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            with self.assertRaises(ValueError):
                write_wav(path, result.waveform(), CONTRACT["architecture"]["sample_rate"])
            self.assertFalse(path.exists())
        stopped = generate(FakeAssets(stop_at=4), CONTRACT, IDS, seed=1, limit=5)
        self.assertEqual((stopped.finish, stopped.patches), ("stop", 5))

    def test_texts_that_do_not_fit_are_rejected_before_any_call(self):
        size = CONTRACT["execution"]["prefill_length"]
        small = copy.deepcopy(CONTRACT)
        small["execution"]["capacity"] = len(IDS)
        for contract, ids, limit in ((CONTRACT, IDS[-1:], None), (CONTRACT, list(range(1, size + 1)) + IDS[-1:], None),
                                     (small, IDS, None), (CONTRACT, IDS, 0)):
            assets = FakeAssets()
            with self.subTest(length=len(ids), capacity=contract["execution"]["capacity"], limit=limit):
                with self.assertRaises(ValueError):
                    generate(assets, contract, ids, seed=1, limit=limit)
                self.assertEqual((assets.calls, assets.states), ([], []))

    def test_non_finite_outputs_end_the_generation_as_an_error(self):
        result = generate(FakeAssets(failing=("locenc", "main")), CONTRACT, IDS, seed=1)
        self.assertEqual((result.finish, result.error["call"], result.patches, result.succeeded), ("error", "locenc.main", 1, False))
        with self.assertRaises(ValueError):
            result.waveform()

    def test_the_trace_keeps_every_call_in_order(self):
        trace = Trace(FakeAssets(stop_at=3))
        result = generate(trace, CONTRACT, IDS, seed=1)
        calls = CONTRACT["generation"]["dit_calls_per_patch"]
        order, arrays = trace.order(), trace.arrays()
        self.assertEqual(order[:6], [["tslm.embed", 0], ["tslm.prefill", 0], ["proj.prefill_fuse", 0], ["ralm.prefill", 0],
                                     ["proj.heads", 0], ["dit.main", 0]])
        self.assertEqual((len(order), order[-1]), (len(trace.calls), ["vae.main", result.patches - 1]))
        self.assertEqual(arrays["dit.main.in.x"].shape, (result.patches * calls, 2, 64, 4))
        self.assertEqual(arrays["dit.main.out.velocity"].shape, (result.patches * calls, 2, 64, 4))
        self.assertEqual(arrays["proj.heads.out.dit_hidden"].shape[0], result.patches)
        self.assertEqual(arrays["tslm.step.in.position"].shape[0], result.patches - 1)


def noise_sequence(seed, count):
    import torch

    generator = torch.Generator().manual_seed(seed)
    return np.stack([torch.randn((1, 64, 4), generator=generator, dtype=torch.float32).numpy() for _ in range(count)])


def matching(step, reference, alternative):
    return {"lm_hidden": reference["fsq_out"][step].copy(), "residual_input": reference["fuse_out"][step].copy()}


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for the solver noise")
class VoxCPM2ReplacementTests(unittest.TestCase):
    def reference(self, shifted=()):
        generator = np.random.default_rng(5)
        fsq, fused = (generator.normal(size=(4, 1, 2048)).astype(np.float32) for _ in range(2))
        noise = noise_sequence(2, 5)
        patch = noise.copy()
        for index in shifted:
            patch[index] = patch[index] + np.float32(0.5)
        reference = {"fsq_out": fsq, "fuse_out": fused, "noise": noise, "patch": patch, "mu": np.zeros((5, 1, 2048), np.float32),
                     "cond": np.concatenate([np.zeros_like(patch[:1]), patch[:-1]])}
        alternative = (fsq[1] + np.float32(0.5), fused[1] - np.float32(0.5))
        candidates = {1: [{"dim": 7, "distance": 4e-7, "code": 3.0, "candidate_code": 2.0, "outputs": alternative}]}
        return reference, candidates, alternative

    def generate(self, outputs=matching, stop_at=4, shifted=(), solve=None, change=None, rerun_shift=0.0):
        reference, candidates, alternative = self.reference(shifted)
        if change:
            change(reference)
        solved = []

        def original(cases):
            solved.append([tuple(value.copy() for value in case) for case in cases])
            index = next(index for index, noise in enumerate(reference["noise"]) if np.array_equal(noise, cases[1][0]))
            return [solve(*cases[0]) if solve else cases[0][0].copy(), reference["patch"][index] + np.float32(rerun_shift)]

        alignment = Alignment(reference, candidates, LIMIT, original, 1, True)
        assets = FakeAssets(stop_at=stop_at, fuse=lambda step: outputs(step, reference, alternative))
        return generate(assets, CONTRACT, IDS, seed=2, align=alignment), alignment, assets, reference, solved

    def test_values_within_the_criterion_are_kept_without_the_original_solver(self):
        def outputs(step, reference, alternative):
            return {"lm_hidden": reference["fsq_out"][step] * np.float32(1 + 1e-5), "residual_input": reference["fuse_out"][step].copy()}

        result, alignment, assets, reference, solved = self.generate(outputs)
        self.assertEqual((result.finish, alignment.positions, alignment.calls, solved), ("stop", [], 4, []))
        for step, value in enumerate(assets.inputs("proj", "heads", "lm_hidden")[1:]):
            np.testing.assert_array_equal(value, reference["fsq_out"][step] * np.float32(1 + 1e-5))

    def test_a_latent_with_matching_inputs_is_replaced_before_locenc_vae_and_the_next_cond(self):
        result, alignment, assets, reference, solved = self.generate(shifted=(2,))
        self.assertEqual(result.finish, "stop")
        self.assertEqual([(row["patch"], row["kind"], row["a"]) for row in alignment.positions], [(2, "latent", 0.0)])
        record = alignment.positions[0]
        self.assertGreater(record["error"]["normalized_max"], LIMIT)
        self.assertGreater(record["b_reference_scale"], LIMIT)
        self.assertTrue(record["inputs"]["noise_bitwise"] and record["inputs"]["cond"]["passed"])
        np.testing.assert_array_equal(result.latents[2], reference["noise"][2])
        np.testing.assert_array_equal(assets.inputs("locenc", "main", "patch")[2], reference["patch"][2])
        np.testing.assert_array_equal(assets.inputs("vae", "main", "latent")[2], reference["patch"][2])
        calls = CONTRACT["generation"]["dit_calls_per_patch"]
        np.testing.assert_array_equal(assets.inputs("dit", "main", "cond")[3 * calls][:1], reference["patch"][2])
        self.assertEqual((len(solved), len(solved[0]), record["original_reproduces_reference"]), (1, 2, True))
        np.testing.assert_array_equal(solved[0][0][0], reference["noise"][2])
        np.testing.assert_array_equal(solved[0][0][2], reference["noise"][1])
        for value, name in zip(solved[0][1], ("noise", "mu", "cond"), strict=True):
            np.testing.assert_array_equal(value, reference[name][2])

    def test_the_original_solver_must_reproduce_the_original_latent_from_the_original_inputs(self):
        result, alignment, _, _, _ = self.generate(shifted=(2,), rerun_shift=1e-6)
        self.assertEqual((result.finish, result.error["reason"], result.error["original_reproduces_reference"]),
                         ("error", "latent_mismatch", False))
        self.assertEqual(alignment.positions, [])

    def test_each_input_condition_stops_a_latent_replacement(self):
        noise = noise_sequence(4, 3)
        patch = noise.copy()
        patch[1] = patch[1] + np.float32(0.5)
        reference = {"noise": noise, "patch": patch, "mu": np.zeros((3, 1, 2048), np.float32),
                     "cond": np.concatenate([np.zeros_like(patch[:1]), patch[:-1]])}
        inputs = {"noise": noise[1], "dit_hidden": np.zeros((1, 2048), np.float32), "cond": reference["cond"][1]}
        cases = {"noise": ({"noise": noise[1] + np.float32(1e-6)}, True), "cond": ({"cond": reference["cond"][1] + np.float32(0.5)}, True),
                 "valid": ({"dit_hidden": np.zeros((1, 2048), np.float64)}, True), "schedule": ({}, False)}
        for name, (change, schedule) in {"control": ({}, True), **cases}.items():
            alignment = Alignment(reference, {}, LIMIT, lambda solved: [solved[0][0].copy(), patch[1]], 1, schedule)
            with self.subTest(name):
                arguments = {**inputs, **change}
                if name == "control":
                    np.testing.assert_array_equal(alignment.latent(1, noise[1], **arguments), patch[1])
                    continue
                with self.assertRaises(GenerationError) as raised:
                    alignment.latent(1, noise[1], **arguments)
                self.assertEqual(raised.exception.details["reason"], "latent_mismatch")
                self.assertEqual(alignment.positions, [])

    def test_a_latent_whose_inputs_differ_from_the_reference_stops(self):
        def change(reference):
            reference["mu"][2] = 1.0

        result, alignment, assets, _, _ = self.generate(shifted=(2,), change=change)
        self.assertEqual((result.finish, result.error["reason"], result.error["patch"]), ("error", "latent_mismatch", 2))
        self.assertFalse(result.error["inputs"]["dit_hidden"]["passed"])
        self.assertEqual((alignment.positions, len(assets.inputs("locenc", "main", "patch"))), ([], 2))

    def test_a_latent_that_the_original_solver_does_not_reproduce_stops(self):
        result, alignment, _, _, _ = self.generate(shifted=(2,), solve=lambda noise, dit_hidden, cond: noise + np.float32(0.5))
        self.assertEqual((result.finish, result.error["reason"]), ("error", "latent_mismatch"))
        self.assertGreater(result.error["a"], LIMIT)
        self.assertEqual(alignment.positions, [])

    def test_one_patch_per_sentence_counts_fuse_and_latent_together(self):
        def outputs(step, reference, alternative):
            chosen = alternative if step == 1 else (reference["fsq_out"][step], reference["fuse_out"][step])
            return {"lm_hidden": chosen[0].copy(), "residual_input": chosen[1].copy()}

        result, alignment, _, _, _ = self.generate(outputs, shifted=(2, 3))
        self.assertEqual([(row["patch"], row["kind"]) for row in alignment.positions], [(2, "fuse"), (2, "latent")])
        self.assertEqual((result.finish, result.error["reason"], result.error["patch"]), ("error", "replacement_limit", 3))

    def test_a_boundary_consistent_fuse_call_continues_with_the_reference_outputs(self):
        def outputs(step, reference, alternative):
            chosen = alternative if step == 1 else (reference["fsq_out"][step], reference["fuse_out"][step])
            return {"lm_hidden": chosen[0].copy(), "residual_input": chosen[1].copy()}

        result, alignment, assets, reference, _ = self.generate(outputs)
        self.assertEqual(result.finish, "stop")
        self.assertEqual([(row["patch"], row["step"], row["dim"], row["code"], row["candidate_code"]) for row in alignment.positions],
                         [(2, 1, 7, 3.0, 2.0)])
        np.testing.assert_array_equal(assets.inputs("ralm", "step", "embed")[1], reference["fuse_out"][1])
        np.testing.assert_array_equal(assets.inputs("proj", "heads", "lm_hidden")[2], reference["fsq_out"][1])

    def test_a_fuse_disagreement_that_is_not_boundary_consistent_stops(self):
        def outputs(step, reference, alternative):
            return {"lm_hidden": reference["fsq_out"][step] + np.float32(step == 2), "residual_input": reference["fuse_out"][step].copy()}

        result, alignment, assets, _, _ = self.generate(outputs)
        self.assertEqual((result.finish, result.error["reason"], result.error["step"]), ("error", "fuse_mismatch", 2))
        self.assertEqual((result.patches, len(assets.inputs("ralm", "step", "embed")), result.succeeded), (3, 2, False))
        self.assertEqual(alignment.positions, [])

    def test_a_generation_that_runs_past_the_reference_stop_fails(self):
        def outputs(step, reference, alternative):
            return matching(min(step, 3), reference, alternative)

        result, _, _, _, _ = self.generate(outputs, stop_at=10)
        self.assertEqual((result.finish, result.error["reason"], result.error["step"]), ("error", "past_reference_stop", 4))


if __name__ == "__main__":
    unittest.main()
