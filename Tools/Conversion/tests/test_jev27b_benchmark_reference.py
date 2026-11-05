import copy
import unittest

from Tools.Conversion.adapters.jev27b.benchmark_reference import measure, prepare_rows, token_hash, validate_lora, wait_nominal
from Tools.Conversion.adapters.jev27b.performance import LENGTHS


class BenchmarkReferenceTests(unittest.TestCase):
    def fixture(self):
        rows = []
        manifest = []
        for count, kind in LENGTHS:
            options = ["false", "true"] if kind == "noul" else [str(n) for n in range(6)] if kind == "score" else ["x", "y"]
            row = {"id": str(count), "kind": kind, "state": f"length{count}", "question": "Q", "options": options}
            rows.append(row)
            manifest.append({"tokens": count, "kind": kind, "n_options": len(options),
                             "tokens_sha256": token_hash([count] * count)})

        def tokenizer(text, add_special_tokens):
            self.assertFalse(add_special_tokens)
            count = next(count for count, _ in LENGTHS if f"length{count}" in text)
            return {"input_ids": [count] * count}

        return {"rows": rows * 4}, {"rows": manifest}, tokenizer

    def test_preparation_requires_the_frozen_sequence_kind_tokens_and_option_count(self):
        document, manifest, tokenizer = self.fixture()
        rows = prepare_rows(document, manifest, tokenizer)
        self.assertEqual(sum(row["warmup"] for row in rows), 4)
        self.assertEqual([row["round"] for row in rows], [0]*4 + [1]*4 + [2]*4 + [3]*4)
        changed = copy.deepcopy(document)
        changed["rows"][0], changed["rows"][1] = changed["rows"][1], changed["rows"][0]
        with self.assertRaises(ValueError):
            prepare_rows(changed, manifest, tokenizer)
        with self.assertRaises(ValueError):
            prepare_rows({"rows": document["rows"][:-1]}, manifest, tokenizer)
        broken = copy.deepcopy(manifest)
        broken["rows"][0]["tokens_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            prepare_rows(document, broken, tokenizer)

    def test_timing_includes_encoding_inference_normalization_and_final_synchronization(self):
        document, manifest, tokenizer = self.fixture()
        row = prepare_rows(document, manifest, tokenizer)[0]
        events = []
        times = iter([10.0, 12.0])

        def encode(text, add_special_tokens):
            events.append("encode")
            return tokenizer(text, add_special_tokens)

        def infer(ids):
            events.append("infer")
            self.assertEqual(ids, [100]*100)
            return [0.0] * 24

        def clock():
            events.append("clock")
            return next(times)

        result = measure(row, encode, infer, 1.0, lambda: events.append("sync"), clock)
        self.assertEqual(events, ["sync", "clock", "encode", "infer", "sync", "clock"])
        self.assertEqual(result["elapsed_seconds"], 2.0)
        self.assertEqual(result["probabilities"], [0.5, 0.5])
        with self.assertRaises(ValueError):
            measure(row, encode, lambda _: [float("nan")]*24, 1.0, lambda: None)

    def test_the_thermal_gate_waits_for_nominal_and_rejects_invalid_or_expired_observations(self):
        states, clock, sleeps = iter([2, 0]), iter([0.0, 0.0, 1.0]), []
        self.assertEqual(wait_nominal(lambda: next(states), clock=lambda: next(clock), sleep=sleeps.append), 1.0)
        self.assertEqual(sleeps, [1])
        clock = iter([0.0, 600.0])
        with self.assertRaises(TimeoutError):
            wait_nominal(lambda: 1, clock=lambda: next(clock), sleep=sleeps.append)
        with self.assertRaises(ValueError):
            wait_nominal(lambda: 4)

    def test_every_adapter_must_be_enabled_unmerged_scaled_and_in_eval(self):
        summary = {"consumed_pairs": 400, "file_pairs": 400, "missing": [], "unexpected": [],
                   "scaling": [2.0], "training": False}
        arguments = {"pairs": 400, "scale": 2.0, "enabled": 400, "total": 400, "merged": 0}
        self.assertTrue(all(validate_lora(summary, **arguments).values()))
        for key, value in [("enabled", 399), ("total", 401), ("merged", 1), ("scale", 4.0)]:
            with self.assertRaises(ValueError):
                validate_lora(summary, **(arguments | {key: value}))
        with self.assertRaises(ValueError):
            validate_lora(summary | {"training": True}, **arguments)
