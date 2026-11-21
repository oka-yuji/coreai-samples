import json
from pathlib import Path
import random
import re
import unicodedata
import unittest

from Tools.LLMEvaluation.items import FILES, ROOT, digest, item, load, read_json
from Tools.LLMEvaluation.synthetic.generate import generate
from Tools.LLMEvaluation.synthetic.solve import fields, solve_extract, solve_format, solve_long

EXAMPLES = json.loads((Path(__file__).resolve().parent / "examples.json").read_text(encoding="utf-8"))
BUNDLE = load(sets=False)
CONTRACT, PROMPTS, VOCABULARY = BUNDLE["contract"], BUNDLE["prompts"], BUNDLE["vocabulary"]
CATEGORIES = {category["id"]: category for category in CONTRACT["categories"]}
SEED = CONTRACT["seed"]


def made(name, index, records=None):
    return generate(CATEGORIES[name], index, SEED, PROMPTS, VOCABULARY, records)


def spec_final_integer(output):
    text = unicodedata.normalize("NFKC", output)
    marker = CONTRACT["scorers"]["final_integer"]["marker"]
    if marker not in text:
        return None
    found = re.match(r"\s*(-?\d+)", text[text.rindex(marker) + len(marker):])
    return int(found.group(1)) if found else None


def spec_first_object(output):
    def pairs(items):
        keys = [key for key, _ in items]
        if len(keys) != len(set(keys)):
            raise KeyError("duplicate key")
        return dict(items)

    decoder = json.JSONDecoder(object_pairs_hook=pairs)
    for start in [index for index, character in enumerate(output) if character == "{"]:
        try:
            value, _ = decoder.raw_decode(output, start)
        except json.JSONDecodeError:
            continue
        return value
    return None


def spec_json_exact(output, answer):
    try:
        value = spec_first_object(output)
    except KeyError:
        return False
    return value is not None and json.dumps(value, sort_keys=True, ensure_ascii=False) == json.dumps(
        answer, sort_keys=True, ensure_ascii=False)


def spec_format(output, rules):
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    if len(lines) != rules["items"] or not all(line.startswith(rules["bullet"]) for line in lines):
        return False
    texts = [line[len(rules["bullet"]):].strip() for line in lines]
    return (all(len(text) <= rules["max_chars"] and text.endswith(tuple(rules["endings"])) for text in texts)
            and all(any(word in text for text in texts) for word in rules["required"]))


def records_of(content):
    return fields(PROMPTS["synth-long"]["instruction"], content, dotall=True)


def rendered(records, question):
    return PROMPTS["synth-long"]["instruction"].format(records="\n".join(records), question=question)


class GeneratorTests(unittest.TestCase):
    def test_same_seed_gives_identical_items_and_hashes(self):
        for name in ("synth-extract", "synth-format", "synth-long-4k", "synth-long-16k"):
            records = 300 if CATEGORIES[name]["generator"]["kind"] == "long" else None
            for index in range(25):
                first, second = made(name, index, records), made(name, index, records)
                self.assertEqual(first, second)
                entry = {"id": f"{name}-0001", "source": {"index": index, "template": first["template"],
                                                          **({"records": records} if records else {})},
                         "token_count": 1}
                self.assertEqual(digest(item(CATEGORIES[name], entry, None, BUNDLE)),
                                 digest(item(CATEGORIES[name], entry, None, BUNDLE)))

    def test_different_indexes_give_different_combinations(self):
        for name in ("synth-extract", "synth-format", "synth-long-4k"):
            combinations = {json.dumps(made(name, index, 300)["combination"], ensure_ascii=False) for index in range(40)}
            self.assertGreater(len(combinations), 30)

    def test_independent_answers_agree_with_the_generator(self):
        for index in range(200):
            extract = made("synth-extract", index)
            self.assertEqual(solve_extract(extract["content"], PROMPTS), extract["answer"])
            rules = made("synth-format", index)
            self.assertEqual(solve_format(rules["content"], PROMPTS), rules["scorer"])
            for name, records in (("synth-long-4k", 260), ("synth-long-16k", 1100)):
                long = made(name, index, records)
                self.assertEqual(solve_long(long["content"], PROMPTS), long["answer"])

    def test_extract_items_have_nesting_and_an_empty_list(self):
        for index in range(100):
            answer = made("synth-extract", index)["answer"]
            rows = next(iter(answer.values()))
            lists = [value for row in rows for value in row.values() if isinstance(value, list)]
            self.assertTrue(any(value == [] for value in lists))
            self.assertTrue(any(value for value in lists))

    def test_long_answers_need_distant_records(self):
        for index in range(50):
            long = made("synth-long-4k", index, 260)
            lines = records_of(long["content"])["records"].split("\n")
            target = long["combination"][1:]
            hits = [number for number, line in enumerate(lines) if all(word in line for word in target)]
            self.assertGreaterEqual(len(hits), 2)
            self.assertGreaterEqual(hits[-1] - hits[0], len(lines) // 2)

    def test_unrelated_records_and_order_do_not_change_the_answer(self):
        for index in range(30):
            long = made("synth-long-4k", index, 260)
            more = made("synth-long-4k", index, 400)
            self.assertEqual(solve_long(more["content"], PROMPTS), long["answer"])
            found = records_of(long["content"])
            lines = found["records"].split("\n")
            random.Random(index).shuffle(lines)
            self.assertEqual(solve_long(rendered(lines, found["question"]), PROMPTS), long["answer"])
            extra = records_of(more["content"])["records"].split("\n")[:50]
            unrelated = [line for line in extra if not all(word in line for word in long["combination"][1:])]
            self.assertEqual(solve_long(rendered(lines + unrelated, found["question"]), PROMPTS), long["answer"])

    def test_extract_order_follows_the_records(self):
        extract = made("synth-extract", 0)
        part = PROMPTS["synth-extract"]["templates"][extract["template"]]
        found = fields(PROMPTS["synth-extract"]["instruction"], extract["content"], dotall=True)
        records = found["records"].split(part["separator"])
        reordered = PROMPTS["synth-extract"]["instruction"].format(
            schema=found["schema"], records=part["separator"].join(reversed(records)))
        root, rows = next(iter(extract["answer"].items()))
        self.assertEqual(solve_extract(reordered, PROMPTS), {root: list(reversed(rows))})

    def test_items_checked_by_hand_in_the_review_match_the_generator_and_solver(self):
        committed = {entry["id"]: entry for entry in read_json(ROOT / FILES["primary"])["items"]}
        for example in EXAMPLES["review_hand_checked"]:
            with self.subTest(example["id"]):
                entry = committed[example["id"]]
                self.assertEqual(entry["source"], example["source"])
                generated = made(entry["category"], example["source"]["index"], example["source"].get("records"))
                self.assertEqual(generated["template"], example["source"]["template"])
                self.assertEqual(generated["answer"], example["hand_answer"])
                kind = CATEGORIES[entry["category"]]["generator"]["kind"]
                solver = solve_extract if kind == "extract" else solve_long
                self.assertEqual(solver(generated["content"], PROMPTS), example["hand_answer"])


class ScoringSpecificationExamples(unittest.TestCase):
    def test_final_integer_examples(self):
        answer = made("synth-long-4k", 3, 260)["answer"]
        marker = CONTRACT["scorers"]["final_integer"]["marker"]
        self.assertEqual(spec_final_integer(f"{marker} {answer}"), answer)
        self.assertEqual(spec_final_integer(EXAMPLES["final_integer"]["full_width_marker"] + str(answer)), answer)
        self.assertEqual(spec_final_integer(f"{marker} 1\n{marker} {answer}"), answer)
        self.assertNotEqual(spec_final_integer(f"{marker} {answer + 1}"), answer)
        self.assertIsNone(spec_final_integer(str(answer)))

    def test_well_formed_but_wrong_json_is_incorrect(self):
        extract = made("synth-extract", 0)
        answer = extract["answer"]
        self.assertTrue(spec_json_exact(json.dumps(answer, ensure_ascii=False, indent=2), answer))
        self.assertTrue(spec_json_exact("```json\n" + json.dumps(answer, ensure_ascii=False) + "\n```", answer))
        root, rows = next(iter(answer.items()))
        wrong = json.loads(json.dumps(answer))
        first = wrong[root][0]
        key = next(key for key, value in first.items() if isinstance(value, str))
        first[key] = first[key] + "X"
        self.assertFalse(spec_json_exact(json.dumps(wrong, ensure_ascii=False), answer))
        extra = json.loads(json.dumps(answer))
        extra["extra"] = []
        self.assertFalse(spec_json_exact(json.dumps(extra, ensure_ascii=False), answer))
        self.assertFalse(spec_json_exact(json.dumps({root: rows[::-1]}, ensure_ascii=False), answer))
        text = json.dumps(answer, ensure_ascii=False)
        self.assertFalse(spec_json_exact(text[:-1] + f', "{root}": []' + "}", answer))
        quantity = made("synth-extract", 2)
        self.assertEqual(quantity["template"], "orders")
        typed = json.dumps(quantity["answer"], ensure_ascii=False)
        number = re.search(r": (\d+)", typed)
        swapped = typed[:number.start(1)] + f'"{number.group(1)}"' + typed[number.end(1):]
        self.assertFalse(spec_json_exact(swapped, quantity["answer"]))

    def test_format_examples(self):
        examples = EXAMPLES["format"]
        self.assertTrue(spec_format(examples["correct"], examples["rules"]))
        for name, output in examples["incorrect"].items():
            with self.subTest(name):
                self.assertFalse(spec_format(output, examples["rules"]))
        generated = made("synth-format", 1)
        self.assertEqual(solve_format(generated["content"], PROMPTS), generated["scorer"])

if __name__ == "__main__":
    unittest.main()
