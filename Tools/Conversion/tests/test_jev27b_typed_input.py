from pathlib import Path
import unittest

from Tools.Conversion.adapters.jev27b.typed_head import band, render
from Tools.Conversion.adapters.jev27b.typed_input import input_rows
from Tools.Conversion.recipe import load_recipe
from Tools.Conversion.storage import PipelineError

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "Tools/Conversion/recipes/jev27b/recipe.json"
BINDINGS = {name: "/unused" for name in ("base", "corpus", "hf_cache", "jglue", "python", "runner", "setup_record",
                                         "snapshot", "upstream")}


def encode(text):
    return list(text)


ROWS = [{"id": "a", "kind": "noul", "state": "s", "question": "q", "options": ["false", "true"], "target": [0, 1]},
        {"id": "b", "kind": "choice", "state": "", "question": "pick", "options": ["x", "y", "z"], "label": 2}]


def entries(rows=ROWS):
    counts = [len(render(row["kind"], row["state"], row["question"], row["options"])) for row in rows]
    return [{"id": row["id"], "tokens": count, "band": band(count)} for row, count in zip(rows, counts)]


class TypedInputTests(unittest.TestCase):
    def test_rows_keep_only_the_request_fields_in_fixture_order(self):
        self.assertEqual(input_rows(entries(), ROWS, encode),
                         [{key: row[key] for key in ("id", "kind", "state", "question", "options")} for row in ROWS])

    def test_rows_that_differ_from_the_fixture_are_refused(self):
        changes = [lambda items: items[0].update(tokens=items[0]["tokens"] + 1),
                   lambda items: items[1].update(band="513-2048"),
                   lambda items: items[1].update(id="c"),
                   lambda items: items.pop()]
        for change in changes:
            broken = entries()
            change(broken)
            with self.assertRaises(ValueError):
                input_rows(broken, ROWS, encode)


class RecipeTests(unittest.TestCase):
    def test_the_core_ai_steps_follow_the_quantized_references_and_need_the_verify_binding(self):
        with self.assertRaisesRegex(PipelineError, "verify"):
            load_recipe(RECIPE, BINDINGS, ROOT)
        steps = load_recipe(RECIPE, {**BINDINGS, "verify": "/unused/coreai-verify"}, ROOT)["steps"]
        ids = [step["id"] for step in steps]
        self.assertEqual(ids[ids.index("verify_quantization") + 1:],
                         ["typed_input", "typed_decide_f", "compare_f", "reference_q", "reference_merged_q",
                          "typed_input_eval", "typed_decide_q", "typed_decide_cal", "typed_decide_h", "score_q",
                          "calibrate", "jdev_fixture", "typed_input_jdev", "typed_decide_jdev", "prompt_adoption",
                          "typed_input_j", "typed_decide_j", "reference_j", "score_j", "verify_merge"])
        decide = steps[ids.index("typed_decide_f")]
        self.assertEqual((decide["command"][:2], decide["heavy"]), (["$verify", "typed-decide"], True))


if __name__ == "__main__":
    unittest.main()
