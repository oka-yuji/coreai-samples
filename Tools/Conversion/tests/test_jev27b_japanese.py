import ast
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from Tools.Conversion.adapters.jev27b.compare import tokens_sha256
from Tools.Conversion.adapters.jev27b.japanese import (
    ADOPTION_SCHEMA, GROUPS, JDEV_ROWS, NLI_LABELS, NOUL, ONE, PERMUTATIONS, VARIANTS, adopt, answer, fixed_texts,
    frozen_adoption, jdev_document, label_problems, load_wrappers, metrics, order_sensitivity, pearson, plan, ranks,
    reference_problems, spearman, swift_problems, tokenized, wrapped,
)
from Tools.Conversion.adapters.jev27b.typed_head import FIXED_OPTIONS, decide, render
from Tools.Conversion.recipe import load_recipe

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "Tools/Conversion/recipes/jev27b"
WRAPPERS = RECIPE / "wrappers.json"
BINDINGS = {name: "/unused" for name in ("base", "corpus", "hf_cache", "jglue", "python", "runner", "setup_record",
                                         "snapshot", "upstream", "verify")}
PINS = {"jnli_options": ["e", "c", "n"], "jnli_labels": ["entailment", "contradiction", "neutral"]}


def units():
    return [{"id": "jglue:JNLI:1", "task": "JNLI", "row_sha256": "1" * 64,
             "row": {"kind": "choice", "state": "premise one", "question": "hypothesis one", "options": ["e", "c", "n"],
                     "label": 0}},
            {"id": "jglue:JNLI:2", "task": "JNLI", "row_sha256": "2" * 64,
             "row": {"kind": "choice", "state": "premise two", "question": "hypothesis two", "options": ["e", "c", "n"],
                     "label": 2}},
            {"id": "jglue:JCommonsenseQA:3", "task": "JCommonsenseQA", "row_sha256": "3" * 64,
             "row": {"kind": "choice", "state": "", "question": "which one", "options": ["a", "b", "c", "d", "e"],
                     "label": 1}},
            {"id": "jglue:JSTS:4", "task": "JSTS", "row_sha256": "4" * 64,
             "row": {"kind": "score", "state": "sentence one", "question": "sentence two",
                     "options": list(FIXED_OPTIONS["score"]), "label": 3.2}}]


def logits(kind, values):
    result = [0.0] * 24
    start = {"noul": 0, "score": 2, "choice": 8}[kind]
    result[start:start + len(values)] = values
    return result


def tokens(text):
    return [ord(character) for character in text]


class WrapperTests(unittest.TestCase):
    def test_both_variants_wrap_the_text_without_changing_it(self):
        wrappers = load_wrappers(WRAPPERS)
        for variant in VARIANTS:
            for unit in units():
                for group in [group for group in GROUPS if group.split(":")[0] == unit["task"]]:
                    row = wrapped(unit, variant, group, wrappers)
                    item = wrappers["variants"][variant][group]
                    self.assertEqual(row["question"], item["prefix"] + unit["row"]["question"] + item["suffix"])
                    self.assertIn(unit["row"]["question"], row["question"])
                    self.assertEqual(row["state"], unit["row"]["state"])
                    render(row["kind"], row["state"], row["question"], row["options"])
                    if group == "JNLI":
                        self.assertEqual(row["options"], item["options"])
                    elif group == NOUL:
                        self.assertEqual(row["options"], list(FIXED_OPTIONS["noul"]))
                    else:
                        self.assertEqual(row["options"], unit["row"]["options"])

    def test_the_wrappers_name_every_group_for_both_variants(self):
        data = json.loads(WRAPPERS.read_text())
        self.assertEqual(set(data["variants"]), set(VARIANTS))
        broken = json.loads(WRAPPERS.read_text())
        del broken["variants"]["en"]["JSTS"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wrappers.json"
            path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                load_wrappers(path)

    def test_the_jnli_options_follow_the_order_of_the_jnli_labels(self):
        wrappers = load_wrappers(WRAPPERS)
        pins = json.loads((RECIPE / "fixtures/jglue.json").read_text())
        self.assertEqual(tuple(pins["jnli_labels"]), NLI_LABELS)
        self.assertEqual(wrappers["variants"]["en"]["JNLI"]["options"], list(NLI_LABELS))
        self.assertEqual(wrappers["variants"]["ja"]["JNLI"]["options"], pins["jnli_options"])

    def test_noul_is_true_only_for_entailment(self):
        wrappers = load_wrappers(WRAPPERS)
        first, second = units()[:2]
        self.assertEqual(wrapped(first, "en", NOUL, wrappers)["label"], 1)
        self.assertEqual(wrapped(second, "en", NOUL, wrappers)["label"], 0)
        for label in (1, 2):
            unit = {**first, "row": {**first["row"], "label": label}}
            self.assertEqual(wrapped(unit, "ja", NOUL, wrappers)["label"], 0)
        self.assertEqual(wrapped(first, "ja", NOUL, wrappers)["id"], "jglue:JNLI:1:noul|ja|identity")


class PermutationTests(unittest.TestCase):
    def test_the_four_orders_move_the_label_with_its_option(self):
        self.assertEqual(sorted(map(tuple, (sorted(order) for order in PERMUTATIONS.values()))), [(0, 1, 2, 3, 4)] * 4)
        self.assertEqual(len(set(PERMUTATIONS.values())), 4)
        wrappers = load_wrappers(WRAPPERS)
        unit = units()[2]
        for name, order in PERMUTATIONS.items():
            row = wrapped(unit, "ja", "JCommonsenseQA", wrappers, name)
            self.assertEqual(row["options"][row["label"]], unit["row"]["options"][unit["row"]["label"]])
            self.assertEqual(row["options"], [unit["row"]["options"][index] for index in order])
            values = [0.0] * 5
            values[row["label"]] = 1.0
            self.assertEqual(answer(row, values), unit["row"]["label"])
        reverse = wrapped(unit, "ja", "JCommonsenseQA", wrappers, "reverse")
        self.assertEqual(answer(reverse, [0.2] * 5), 4)
        with self.assertRaises(ValueError):
            wrapped(units()[0], "ja", "JNLI", wrappers, "reverse")

    def test_order_sensitivity_counts_rows_whose_four_answers_agree(self):
        wrappers = load_wrappers(WRAPPERS)
        unit = units()[2]
        other = {**unit, "id": "jglue:JCommonsenseQA:5", "row": {**unit["row"], "question": "another"}}
        rows = plan([unit, other], wrappers, ("JCommonsenseQA",), "en")
        values = []
        for row in rows:
            choice = [0.0] * 5
            target = row["label"] if row["source_id"] == unit["id"] else row["order"].index(0)
            if row["source_id"] == other["id"] and row["permutation"] == "rotate2":
                target = row["order"].index(3)
            choice[target] = 2.0
            values.append(logits("choice", choice))
        result = order_sensitivity(rows, values, ONE, "en")
        self.assertEqual((result["rows"], result["complete_rows"], result["consistent_rows"]), (2, 2, 1))
        self.assertEqual(result["orders"]["identity"]["hits"], 1)
        self.assertEqual(result["orders"]["rotate2"]["hits"], 1)


class StatisticTests(unittest.TestCase):
    def test_ranks_average_ties(self):
        self.assertEqual(ranks([3.0, 1.0, 3.0, 2.0]), [3.5, 1.0, 3.5, 2.0])
        self.assertEqual(ranks([5.0, 5.0, 5.0]), [2.0, 2.0, 2.0])

    def test_pearson_and_spearman_match_known_values_and_are_undefined_for_constants(self):
        self.assertAlmostEqual(pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0, places=15)
        self.assertAlmostEqual(pearson([1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]), 0.6, places=15)
        self.assertAlmostEqual(spearman([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0]), 1.0, places=15)
        self.assertAlmostEqual(spearman([1.0, 2.0, 2.0, 3.0], [1.0, 3.0, 2.0, 4.0]),
                               pearson([1.0, 2.5, 2.5, 4.0], [1.0, 3.0, 2.0, 4.0]), places=15)
        self.assertIsNone(pearson([0.1] * 3, [1.0, 2.0, 3.0]))
        self.assertIsNone(spearman([1.0, 2.0, 3.0], [4.0] * 3))
        self.assertIsNone(pearson([1.0], [2.0]))

    def test_metrics_follow_the_fixed_definitions(self):
        wrappers = load_wrappers(WRAPPERS)
        first, second, _, jsts = units()
        noul = [wrapped(first, "ja", NOUL, wrappers), wrapped(second, "ja", NOUL, wrappers)]
        choice = [wrapped(first, "ja", "JNLI", wrappers), wrapped(second, "ja", "JNLI", wrappers)]
        score = wrapped(jsts, "ja", "JSTS", wrappers)
        items = [(noul[0], logits("noul", [0.0, 0.0])), (noul[1], logits("noul", [1.0, 0.0])),
                 (choice[0], logits("choice", [1.0, 1.0, 0.0])), (choice[1], logits("choice", [0.0, 0.0, 3.0])),
                 (score, logits("score", [0.0] * 6))]
        result = metrics(items, ONE)
        self.assertEqual((result[NOUL]["hits"], result[NOUL]["accuracy"]), (1, 0.5))
        p_true = decide(logits("noul", [1.0, 0.0]), "noul", 2, 1.0)[0][1]
        self.assertAlmostEqual(result[NOUL]["brier_hard"], ((0.5 - 1) ** 2 + p_true ** 2) / 2, places=15)
        self.assertEqual(result[NOUL]["auroc"], 1.0)
        self.assertEqual(result["JNLI"]["confusion"]["matrix"], [[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        self.assertEqual(result["JNLI"]["hits"], 2)
        self.assertAlmostEqual(result["JSTS"]["mae"], abs(2.5 - 3.2), places=14)
        self.assertIsNone(result["JSTS"]["pearson"])
        one_class = metrics(items[:1], ONE)[NOUL]
        self.assertIsNone(one_class["auroc"])


class JdevTests(unittest.TestCase):
    def tasks(self):
        tasks = {"JNLI": {}, "JCommonsenseQA": {}, "JSTS": {}}
        for index in range(JDEV_ROWS + 30):
            tasks["JNLI"][str(index)] = {"sentence1": f"premise {index % 40}", "sentence2": f"hypothesis {index}",
                                         "label": "neutral"}
            tasks["JSTS"][str(index)] = {"sentence1": f"first {index}", "sentence2": f"second {index}", "label": 1.0}
            tasks["JCommonsenseQA"][str(index)] = {"question": f"question {index}", "label": 0,
                                                   **{f"choice{k}": f"c{k}" for k in range(5)}}
        tasks["JCommonsenseQA"]["7"]["choice1"] = "c0"
        return tasks

    def test_jdev_leaves_out_the_fixed_rows_and_every_shared_sentence(self):
        identifiers = {"jglue:JNLI:0", "jglue:JSTS:1", "jglue:JCommonsenseQA:2"}
        texts = {"premise 0", "hypothesis 5", "second 9", "first 8", "question 3", "premise 1"}
        document = jdev_document(self.tasks(), PINS, identifiers, texts)
        self.assertTrue(document["passed"])
        chosen = {row["id"] for row in document["rows"]}
        self.assertEqual(len(chosen), 3 * JDEV_ROWS)
        self.assertFalse(chosen & identifiers)
        excluded = {f"jglue:JNLI:{index}" for index in range(JDEV_ROWS + 30) if index % 40 in (0, 1)} | {
            "jglue:JNLI:5", "jglue:JSTS:9", "jglue:JSTS:8", "jglue:JCommonsenseQA:3", "jglue:JCommonsenseQA:7"}
        self.assertFalse(chosen & excluded)
        self.assertEqual(document["pools"]["JCommonsenseQA"],
                         {"rows": JDEV_ROWS + 30, "in_fixed_sets": 1, "shares_text": 1, "not_renderable": 1,
                          "eligible": JDEV_ROWS + 27})
        self.assertEqual(document["pools"]["JNLI"]["shares_text"], 8)
        self.assertEqual(jdev_document(self.tasks(), PINS, identifiers, texts)["rows"], document["rows"])
        self.assertNotEqual([row["id"] for row in jdev_document(self.tasks(), PINS, set(), set())["rows"]],
                            [row["id"] for row in document["rows"]])


class FixedTextTests(unittest.TestCase):
    def test_the_fixed_texts_come_from_the_jglue_rows_of_both_f_and_j(self):
        records = {
            "JNLI": [{"sentence_pair_id": "1", "sentence1": "p1", "sentence2": "h1", "label": "neutral"},
                     {"sentence_pair_id": "2", "sentence1": "p2", "sentence2": "h2", "label": "entailment"}],
            "JCommonsenseQA": [{"q_id": 7, "question": "q7", "label": 0, **{f"choice{k}": f"c{k}" for k in range(5)}},
                               {"q_id": 8, "question": "q8", "label": 1, **{f"choice{k}": f"d{k}" for k in range(5)}}],
            "JSTS": [{"sentence_pair_id": "3", "sentence1": "s3", "sentence2": "t3", "label": 1.0}],
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pins = {"files": {}}
            for task, rows in records.items():
                data = "".join(json.dumps(row) + "\n" for row in rows).encode()
                (root / f"{task}.json").write_bytes(data)
                pins["files"][task] = {"path": f"{task}.json", "bytes": len(data),
                                       "sha256": hashlib.sha256(data).hexdigest()}
            (root / "F.json").write_text(json.dumps({"rows": [{"id": "row1", "source": "test_set_30k"},
                                                              {"id": "jglue:JNLI:1", "source": "jglue:JNLI"}]}))
            j = [{"id": "jglue:JCommonsenseQA:7", "source": "jglue:JCommonsenseQA"},
                 {"id": "jglue:JSTS:3", "source": "jglue:JSTS"}]
            (root / "J.json").write_text(json.dumps({"rows": j}))
            tasks, identifiers, texts = fixed_texts(root, pins, root)
        self.assertEqual(identifiers, {"jglue:JNLI:1", "jglue:JCommonsenseQA:7", "jglue:JSTS:3"})
        self.assertEqual(texts, {"p1", "h1", "q7", "s3", "t3"})
        self.assertEqual(set(tasks["JCommonsenseQA"]), {"7", "8"})


class AdoptionTests(unittest.TestCase):
    def results(self, ja, en):
        return {variant: {"JNLI": {"hits": values[0]}, "JCommonsenseQA": {"hits": values[1]}, "JSTS": {"mae": values[2]}}
                for variant, values in (("ja", ja), ("en", en))}

    def test_ties_adopt_ja_and_noul_follows_jnli_choice(self):
        self.assertEqual(adopt(self.results((50, 60, 1.0), (50, 60, 1.0))), dict.fromkeys(GROUPS, "ja"))
        self.assertEqual(adopt(self.results((50, 61, 1.0), (51, 60, 0.9))),
                         {"JNLI": "en", NOUL: "en", "JCommonsenseQA": "ja", "JSTS": "en"})

    def test_only_a_frozen_adoption_for_the_same_wrappers_is_accepted(self):
        good = {"schema": ADOPTION_SCHEMA, "frozen": True, "wrappers_sha256": "a" * 64, "adopted": dict.fromkeys(GROUPS, "ja")}
        changes = [{"frozen": False}, {"wrappers_sha256": "b" * 64}, {"adopted": {"JNLI": "ja"}},
                   {"adopted": {**good["adopted"], "JSTS": "fr"}}, {"schema": "other"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prompt-adoption.json"
            path.write_text(json.dumps(good))
            self.assertEqual(frozen_adoption(path, "a" * 64)["adopted"], good["adopted"])
            for change in changes:
                path.write_text(json.dumps({**good, **change}))
                with self.assertRaises(ValueError):
                    frozen_adoption(path, "a" * 64)


class ConsistencyTests(unittest.TestCase):
    BUNDLE = {"version": "v" * 64, "asset_sha256": "s" * 64, "temperatures": {"noul": 1.1, "choice": 0.9, "score": 1.2}}

    def documents(self):
        rows = tokenized(plan(units(), load_wrappers(WRAPPERS), GROUPS, "ja"), tokens, 8192)
        answers = []
        for index, row in enumerate(rows):
            values = [((7 * index + slot) % 5 - 2) / 3 for slot in range(24)]
            shipped, plain = decide(values, row["kind"], len(row["options"]), self.BUNDLE["temperatures"][row["kind"]])
            answers.append({"id": row["id"], "kind": row["kind"], "token_count": row["token_count"],
                            "tokens_sha256": tokens_sha256(row["tokens"]), "capacity": 256, "logits": values,
                            "probabilities": shipped, "probabilities_t1": plain, "top": 0, "elapsed_seconds": 0.1})
        swift = {"schema": "typed-decide/1", "bundle": {"version": self.BUNDLE["version"],
                                                        "asset_sha256": self.BUNDLE["asset_sha256"]}, "rows": answers}
        plain = [row for row in rows if row["permutation"] == "identity"]
        reference = {"passed": True, "rows": [{**row, "logits1m": [0.1] * 24} for row in plain]}
        return rows, swift, plain, reference

    def test_consistent_outputs_have_no_problems(self):
        rows, swift, plain, reference = self.documents()
        self.assertEqual(len(rows), 2 * 6 + 3)
        self.assertEqual((swift_problems(rows, swift, self.BUNDLE), reference_problems(plain, reference),
                          label_problems(rows)), ([], [], []))

    def test_missing_or_inconsistent_outputs_are_problems(self):
        def token(swift, reference):
            swift["rows"][1]["tokens_sha256"] = "0" * 64

        def missing(swift, reference):
            swift["rows"].pop()

        def nonfinite(swift, reference):
            swift["rows"][2]["logits"][3] = math.nan

        def bundle(swift, reference):
            swift["bundle"]["version"] = "w" * 64

        def shipping(swift, reference):
            swift["rows"][0]["probabilities"] = swift["rows"][0]["probabilities_t1"]

        for change in (token, missing, nonfinite, bundle, shipping):
            rows, swift, plain, reference = self.documents()
            change(swift, reference)
            self.assertTrue(swift_problems(rows, swift, self.BUNDLE), change.__name__)

        def text(swift, reference):
            reference["rows"][0]["text_sha256"] = "0" * 64

        def order(swift, reference):
            reference["rows"].reverse()

        def failed(swift, reference):
            reference["passed"] = False

        def infinite(swift, reference):
            reference["rows"][1]["logits1m"][0] = math.inf

        for change in (text, order, failed, infinite):
            rows, swift, plain, reference = self.documents()
            change(swift, reference)
            self.assertTrue(reference_problems(plain, reference), change.__name__)
        rows = self.documents()[0]
        rows[0]["label"] = 3
        rows[-1]["label"] = 5.5
        self.assertEqual(label_problems(rows), [rows[0]["id"], rows[-1]["id"]])


def imported(path):
    found, pending = set(), [path]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        for node in ast.walk(ast.parse(current.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level and node.module:
                base = current.parents[node.level - 1]
                candidate = base.joinpath(*node.module.split(".")).with_suffix(".py")
                if candidate.is_file():
                    pending.append(candidate)
    return found


class RecipeTests(unittest.TestCase):
    STEPS = ["jdev_fixture", "typed_input_jdev", "typed_decide_jdev", "prompt_adoption", "typed_input_j",
             "typed_decide_j", "reference_j", "score_j"]

    def steps(self):
        recipe = load_recipe(RECIPE / "recipe.json", BINDINGS, ROOT)
        return recipe, {step["id"]: step for step in recipe["steps"]}

    def test_the_japanese_steps_follow_the_calibration_and_precede_verify_merge(self):
        recipe, steps = self.steps()
        ids = list(steps)
        self.assertEqual(ids[ids.index("calibrate") + 1:], self.STEPS + ["verify_merge"])
        expected = {"jdev_fixture": ["fixture"], "typed_input_jdev": ["jdev_fixture", "prepare"],
                    "typed_decide_jdev": ["calibrate", "typed_input_jdev"],
                    "prompt_adoption": ["typed_decide_jdev", "typed_input_jdev", "jdev_fixture", "calibrate"],
                    "typed_input_j": ["prompt_adoption", "fixture", "prepare"],
                    "typed_decide_j": ["calibrate", "typed_input_j"], "reference_j": ["prepare", "merge", "fixture"],
                    "score_j": ["typed_input_j", "typed_decide_j", "reference_j", "prompt_adoption", "calibrate"]}
        self.assertEqual({name: steps[name]["depends"] for name in self.STEPS}, expected)
        self.assertEqual({name for name in self.STEPS if steps[name].get("heavy")},
                         {"typed_decide_jdev", "typed_decide_j", "reference_j"})
        for name in ("typed_decide_jdev", "typed_decide_j"):
            self.assertEqual(steps[name]["command"][:2], ["$verify", "typed-decide"])
            self.assertIn("$calibrate/$bundle_gdnfp32", steps[name]["command"])
        self.assertIn(recipe["variables"]["bundle_gdnfp32"], steps["calibrate"]["outputs"])
        self.assertEqual(steps["prompt_adoption"]["checks"],
                         [{"file": "prompt-adoption.json", "key": ["frozen"], "equals": True}])
        self.assertEqual(steps["score_j"]["checks"], [{"file": "japanese.json", "key": ["recorded"], "equals": True}])
        self.assertEqual(steps["reference_j"]["checks"],
                         [{"file": "reference-merged-J.json", "key": ["passed"], "equals": True}])

    def test_every_python_step_lists_the_modules_it_imports(self):
        _, steps = self.steps()
        adapters = ROOT / "Tools/Conversion/adapters/jev27b"
        for name in self.STEPS:
            step = steps[name]
            if step["command"][0] != "$python":
                continue
            module = step["command"][step["command"].index("-m") + 1].split(".")[-1]
            listed = {item.replace("$adapters", str(adapters)).replace("$repo", str(ROOT)) for item in step["inputs"]}
            self.assertLessEqual({str(path) for path in imported(adapters / f"{module}.py")}, listed, name)
            if "--wrappers" in step["command"]:
                self.assertIn("$recipe_dir/wrappers.json", step["inputs"], name)


if __name__ == "__main__":
    unittest.main()
