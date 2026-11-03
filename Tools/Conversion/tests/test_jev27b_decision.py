import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from Tools.Conversion.adapters.jev27b.decision import (
    DECISION, SUMS, STATES, TEMPLATE, document, listing, validate, write_sums,
)
from Tools.Conversion.adapters.jev27b.typed_head import FIXED_OPTIONS, LABELS, render
from Tools.Conversion.storage import write_json

CONTRACT = {"model_id": "autotrust/JEV-27B", "revision": "1" * 40,
            "base_model": {"id": "Qwen/Qwen3.8-27B", "revision": "2" * 40}, "merged_model_id": "owner/merged",
            "head": "head.safetensors", "context_length": 262144, "limits": {"options": 16, "input_tokens": 8192},
            "upstream_revision": "3" * 40, "compression": "qwen3_5_cfg", "precision": "bfloat16"}
JUDGE = {"slots": {"verbalizers": [*FIXED_OPTIONS["noul"], *FIXED_OPTIONS["score"], *LABELS]},
         "verbalizer_ids": list(range(24))}
CALIBRATION = {"per_kind": {"noul": 1.0143134751376188, "choice": 1.0161280671450397, "score": 1.0036213515883572}}
DIGESTS = {"head": "a" * 64, "calibration": "b" * 64, "adapter": "c" * 64, "adapter_config": "d" * 64}
ASSET = "bundle.aimodel"


def bundle(root):
    files = {f"{ASSET}/main.mlirb": b"program", f"{ASSET}/main.hash": b"hash", "head.safetensors": b"head",
             "tokenizer/tokenizer.json": b"{}", "metadata.json": b"{}", "LICENSE": b"license", "NOTICE.md": b"notice"}
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    digests = {**DIGESTS, "head": hashlib.sha256(files["head.safetensors"]).hexdigest()}
    decision = document(CONTRACT, JUDGE, CALIBRATION, digests, "4" * 40, ASSET, 5120, listing(root))
    write_json(root / DECISION, decision)
    write_sums(root)
    return decision


class DecisionValidationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.decision = bundle(self.root)

    def tearDown(self):
        self.folder.cleanup()

    def test_a_complete_bundle_passes_with_its_files(self):
        self.assertEqual(validate(self.decision), [])
        self.assertEqual(validate(self.decision, self.root), [])
        self.assertEqual(self.decision["backbone"]["states"], list(STATES))
        self.assertEqual(self.decision["head"]["weight_shape"], [24, 5120])
        self.assertEqual(self.decision["temperatures"]["source"], "card")

    def test_a_missing_key_is_rejected(self):
        for key in ("head", "slots", "files"):
            broken = copy.deepcopy(self.decision)
            del broken[key]
            self.assertIn(f"decision.{key} is missing", validate(broken))
        broken = copy.deepcopy(self.decision)
        del broken["backbone"]["hidden"]
        self.assertIn("backbone.hidden is missing", validate(broken))

    def test_overlapping_or_partial_slots_are_rejected(self):
        broken = copy.deepcopy(self.decision)
        broken["slots"]["score"] = [1, 8]
        self.assertIn("slots must be disjoint and cover every head row", validate(broken))
        broken["slots"]["score"] = [3, 8]
        self.assertIn("slots must be disjoint and cover every head row", validate(broken))

    def test_a_nonpositive_or_missing_temperature_is_rejected(self):
        for value in (0.0, -1.0, float("inf")):
            broken = copy.deepcopy(self.decision)
            broken["temperatures"]["per_kind"]["score"] = value
            self.assertIn("temperatures.per_kind must give every kind a positive finite value", validate(broken))
        broken = copy.deepcopy(self.decision)
        del broken["temperatures"]["per_kind"]["noul"]
        self.assertIn("temperatures.per_kind must give every kind a positive finite value", validate(broken))

    def test_a_changed_file_or_head_digest_is_rejected(self):
        (self.root / "head.safetensors").write_bytes(b"other")
        self.assertIn("file differs: head.safetensors", validate(self.decision, self.root))
        broken = copy.deepcopy(self.decision)
        broken["head"]["sha256"] = "e" * 64
        self.assertIn("head.file must be listed with the head digest", validate(broken))

    def test_an_unlisted_file_and_a_shape_mismatch_are_rejected(self):
        (self.root / "extra.txt").write_bytes(b"x")
        self.assertIn("files differ from the bundle: ['extra.txt']", validate(self.decision, self.root))
        broken = copy.deepcopy(self.decision)
        broken["backbone"]["hidden"]["size"] = 4096
        self.assertIn("head must hold proj.weight [24, hidden] and proj.bias [24] in float32", validate(broken))
        broken = copy.deepcopy(self.decision)
        broken["limits"]["options"] = 0
        self.assertIn("limits must give positive integer options and input_tokens", validate(broken))

    def test_sums_list_every_file_but_themselves_in_shasum_format(self):
        lines = (self.root / SUMS).read_text().splitlines()
        paths = [line.split("  ", 1)[1] for line in lines]
        self.assertEqual(paths, sorted(paths))
        self.assertIn(DECISION, paths)
        self.assertNotIn(SUMS, paths)
        for line in lines:
            digest, path = line.split("  ", 1)
            self.assertEqual(digest, hashlib.sha256((self.root / path).read_bytes()).hexdigest())
        self.assertEqual([entry["path"] for entry in self.decision["files"]], [p for p in paths if p != DECISION])


class DecisionTemplateTests(unittest.TestCase):
    def test_the_template_reproduces_the_bare_rendering_for_every_kind(self):
        cases = [("choice", "state", "question?", ["first", "second", "third"]),
                 ("noul", "s", "q", ["false", "true"]), ("score", "", "q", ["0", "1", "2", "3", "4", "5"])]
        for kind, state, question, options in cases:
            lines = [TEMPLATE["choice_line"].format(label=TEMPLATE["labels"][i], option=option)
                     for i, option in enumerate(options)] if kind == "choice" else list(TEMPLATE["fixed_options"][kind])
            text = TEMPLATE["text"].format(kind=kind, state=state, question=question,
                                           options=TEMPLATE["option_separator"].join(lines))
            self.assertEqual(text, render(kind, state, question, options))
            self.assertTrue(text.endswith(TEMPLATE["last_token"]))


if __name__ == "__main__":
    unittest.main()
