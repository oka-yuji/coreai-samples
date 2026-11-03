import json
from pathlib import Path
import random
import tempfile
import unittest

from Tools.Conversion.adapters.jev27b.calibrate import (
    HIGH, LOW, REFIT, calibrated_bundle, evaluate, fit, float32, holdout, recorded_sums, select, shortest,
)
from Tools.Conversion.adapters.jev27b.decision import DECISION, SUMS, listing, sha256, validate
from Tools.Conversion.adapters.jev27b.typed_head import SLOTS, softmax
from Tools.Conversion.recipe import load_recipe
from Tools.Conversion.tests.test_jev27b_decision import bundle

ROOT = Path(__file__).resolve().parents[3]
BINDINGS = {name: "/unused" for name in ("base", "corpus", "hf_cache", "jglue", "python", "runner", "setup_record",
                                         "snapshot", "upstream", "verify")}


def rows(temperature, count=300, seed=3, kinds=("noul", "choice", "score")):
    rng = random.Random(seed)
    found = []
    for index in range(count):
        kind = kinds[index % len(kinds)]
        size = rng.choice((2, 4, 16)) if kind == "choice" else SLOTS[kind][1] - SLOTS[kind][0]
        logits = [rng.gauss(0, 2) for _ in range(24)]
        start = SLOTS[kind][0]
        target = softmax([value / temperature for value in logits[start:start + size]])
        found.append({"id": f"row{index}", "kind": kind, "state": f"state {index}", "target": target, "logits": logits})
    return found


class FitTests(unittest.TestCase):
    def test_a_known_temperature_is_recovered_from_its_logits(self):
        for temperature in (0.7, 1.0161280671450397, 2.5):
            result = fit(rows(temperature, kinds=("choice",)))
            self.assertLess(abs(result["temperature"] - temperature), 1e-3, temperature)
            self.assertIsNone(result["bound"])
            self.assertLess(result["bracket"][1] - result["bracket"][0], 1e-7)
            self.assertLessEqual(result["cross_entropy"], result["cross_entropy_t1"])

    def test_reaching_an_end_of_the_beta_range_is_recorded_without_widening_it(self):
        lower = fit(rows(8.0, count=60, kinds=("noul",)))
        self.assertEqual((lower["bound"], lower["bracket"][0]), ("lower", LOW))
        self.assertAlmostEqual(lower["temperature"], 1 / LOW, places=5)
        upper = fit(rows(0.1, count=60, kinds=("score",)))
        self.assertEqual((upper["bound"], upper["bracket"][1]), ("upper", HIGH))
        self.assertAlmostEqual(upper["temperature"], 1 / HIGH, places=5)

    def test_temperatures_are_rounded_to_float32_and_written_as_the_shortest_decimal(self):
        self.assertEqual(shortest(1.0143134751376188), 1.0143135)
        self.assertNotEqual(float32(1.0143134751376188), 1.0143134751376188)
        rng = random.Random(1)
        for value in [0.25, 4.0, 1.0, *(rng.uniform(0.25, 4.0) for _ in range(200))]:
            decimal = shortest(value)
            self.assertEqual(float32(decimal), float32(value))
            self.assertEqual(float32(json.loads(json.dumps(decimal))), float32(value))
            self.assertLessEqual(len(repr(decimal)), 11)


class SelectionTests(unittest.TestCase):
    def test_rows_that_share_a_state_with_the_fit_are_held_out(self):
        fitted = [{"state": "b"}, {"state": "z"}]
        kept, excluded = holdout([{"state": "a"}, {"state": "b"}, {"state": "c"}, {"state": "b"}], fitted)
        self.assertEqual(([row["state"] for row in kept], [row["state"] for row in excluded]), (["a", "c"], ["b", "b"]))

    def test_a_candidate_is_adopted_only_when_neither_nll_nor_brier_is_worse(self):
        def item(nll, brier):
            return {"nll": nll, "brier_multiclass": brier}

        statistics = {"noul": {"card": item(0.5, 0.2), "candidate": item(0.4, 0.1)},
                      "choice": {"card": item(0.5, 0.2), "candidate": item(0.5, 0.2)},
                      "score": {"card": item(0.5, 0.2), "candidate": item(0.4, 0.2000001)}}
        decisions = select(statistics)
        self.assertEqual({kind: value["adopted"] for kind, value in decisions.items()},
                         {"noul": True, "choice": True, "score": False})
        self.assertEqual(decisions["score"]["worse"], ["brier_multiclass"])

    def test_the_evaluation_prefers_the_temperature_that_made_the_targets(self):
        data = rows(1.6, count=120)
        result = evaluate(data, {kind: {"card": float32(1.0), "candidate": float32(1.6)} for kind in SLOTS})
        for kind in SLOTS:
            card, candidate = result[kind]["card"], result[kind]["candidate"]
            self.assertEqual(card["rows"], 40)
            self.assertLess(candidate["nll"], card["nll"])
            for name in ("soft", "hard"):
                spread = candidate["ece_interval"][name]
                self.assertLessEqual(spread["low"], spread["high"])
                self.assertEqual((spread["valid"], spread["invalid"]), (2000, 0))


class BundleTests(unittest.TestCase):
    PER_KIND = {"noul": 1.25, "choice": 1.0161280671450397, "score": 0.875}

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.source = Path(self.folder.name) / "source"
        self.source.mkdir()
        bundle(self.source)
        self.output = Path(self.folder.name) / "output"
        self.output.mkdir()
        self.before = {path: path.read_bytes() for path in self.source.rglob("*") if path.is_file()}

    def tearDown(self):
        self.folder.cleanup()

    def test_the_copy_records_the_refit_and_lists_it_in_files_and_sums(self):
        destination = self.output / self.source.name
        info = calibrated_bundle(self.source, destination, {"schema": "refit"}, self.PER_KIND, True)
        decision = json.loads((destination / DECISION).read_text())
        digest = sha256(destination / REFIT)
        self.assertEqual(decision["temperatures"], {"per_kind": self.PER_KIND, "source": "refit",
                                                    "calibration_sha256": digest,
                                                    "calibration_id": f"jev27b-refit/{digest[:16]}"})
        self.assertIn(REFIT, [entry["path"] for entry in decision["files"]])
        self.assertEqual(decision["files"], listing(destination))
        self.assertEqual(recorded_sums(destination),
                         {entry["path"]: entry["sha256"] for entry in listing(destination, (SUMS,))})
        self.assertEqual(recorded_sums(destination)[DECISION], info["decision_sha256"])
        self.assertEqual(validate(decision, destination), [])
        self.assertEqual(info["asset_sha256"], recorded_sums(self.source)["bundle.aimodel/main.mlirb"])
        self.assertEqual({path: path.read_bytes() for path in self.source.rglob("*") if path.is_file()}, self.before)

    def test_keeping_every_card_temperature_keeps_the_source_card(self):
        destination = self.output / self.source.name
        calibrated_bundle(self.source, destination, {"schema": "refit"}, self.PER_KIND, False)
        self.assertEqual(json.loads((destination / DECISION).read_text())["temperatures"]["source"], "card")

    def test_a_changed_source_or_an_existing_destination_stops_the_update(self):
        destination = self.output / self.source.name
        destination.mkdir()
        with self.assertRaises(FileExistsError):
            calibrated_bundle(self.source, destination, {"schema": "refit"}, self.PER_KIND, True)
        destination.rmdir()
        (self.source / "NOTICE.md").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256SUMS"):
            calibrated_bundle(self.source, destination, {"schema": "refit"}, self.PER_KIND, True)


class RecipeTests(unittest.TestCase):
    def test_calibrate_follows_the_scoring_and_writes_a_copy_named_like_the_variant(self):
        recipe = load_recipe(ROOT / "Tools/Conversion/recipes/jev27b/recipe.json", BINDINGS, ROOT)
        steps = {step["id"]: step for step in recipe["steps"]}
        ids = list(steps)
        step = steps["calibrate"]
        self.assertEqual(ids[ids.index("score_q") + 1:], ["calibrate", "verify_merge"])
        self.assertEqual(step["depends"], ["typed_decide_cal", "typed_decide_h", "typed_decide_f", "export_gdnfp32",
                                           "verify_quantization"])
        self.assertIn("$export_gdnfp32/$bundle_gdnfp32", step["command"])
        self.assertIn(recipe["variables"]["bundle_gdnfp32"], step["outputs"])
        self.assertEqual(step["checks"], [{"file": "calibration.json", "key": ["recorded"], "equals": True},
                                          {"file": "comparison-refit.json", "key": ["passed"], "equals": True}])
        self.assertTrue({"$adapters/calibrate.py", "$adapters/score.py", "$adapters/decision.py", "$fixtures/Cal.json",
                         "$fixtures/H.json"} <= set(step["inputs"]))
        self.assertFalse(step["heavy"])


if __name__ == "__main__":
    unittest.main()
