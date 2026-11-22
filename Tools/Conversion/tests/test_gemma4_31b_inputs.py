import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from Tools.Conversion.adapters.gemma4_31b import inputs
from Tools.LLMEvaluation.items import digest

FRAME = {"prefix": "<bos><|turn>user\n", "suffix": "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
         "tokenization": {"first_ids": [2, 105]}}
SPECIAL = {"<bos>": 2, "<|turn>": 105, "<turn|>": 106, "<|channel>": 100, "<channel|>": 101}
CONTRACT = {"version": "G31-QC-v1", "sha256": "c" * 64}


class FakeTokenizer:
    def __init__(self, template=None, special=None):
        self.template = template or (lambda content: FRAME["prefix"] + content + FRAME["suffix"])
        self.special = SPECIAL if special is None else special

    def apply_chat_template(self, messages, tokenize, return_dict=False, add_generation_prompt=False,
                            enable_thinking=True):
        text = self.template(messages[0]["content"]) if add_generation_prompt and not enable_thinking else "other"
        return {"input_ids": self(text, add_special_tokens=False)["input_ids"]} if tokenize else text

    def __call__(self, text, add_special_tokens):
        ids = []
        while text:
            name = next((key for key in self.special if text.startswith(key)), None)
            ids.append(self.special[name] if name else 1000 + ord(text[0]))
            text = text[len(name) if name else 1:]
        return {"input_ids": ids}


def item(identifier, content, **extra):
    return {"id": identifier, "category": identifier.rsplit("-", 1)[0], "unit": "question",
            "messages": [{"role": "user", "content": content}], "max_new_tokens": 8, "scorer": {"kind": "choice_digit"},
            "answer": 1, **extra}


def items_and_entries():
    items = [item("ja-choice-0001", "abc"), item("ja-choice-0002", "de"),
             item("long-ja-4k-0001", "fghij", token_count=31)]
    entries = [{"id": value["id"], "item_sha256": digest(value), **({"token_count": 31} if "token_count" in value else {})}
               for value in items]
    return items, entries


class RecordTests(unittest.TestCase):
    def records(self, tokenizer=None, items=None, entries=None):
        found_items, found_entries = items_and_entries()
        return inputs.records(tokenizer or FakeTokenizer(), items or found_items, entries or found_entries, FRAME,
                              set(SPECIAL.values()), 2)

    def test_rebuilding_the_same_items_writes_the_same_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            outputs = [Path(folder) / name for name in ("first", "second")]
            for output in outputs:
                inputs.write(output, self.records(), CONTRACT)
            files = [{name: (output / name).read_bytes() for name in ("inputs.jsonl", "inputs.json")} for output in outputs]
        self.assertEqual(files[0], files[1])
        summary = json.loads(files[0]["inputs.json"])
        self.assertEqual(summary, {"contract": CONTRACT, "count": 3,
                                   "sha256": hashlib.sha256(files[0]["inputs.jsonl"]).hexdigest()})
        lines = files[0]["inputs.jsonl"].split(b"\n")
        self.assertEqual((len(lines), lines[-1]), (4, b""))
        self.assertEqual(json.loads(lines[0]), self.records()[0])
        self.assertEqual(list(json.loads(lines[0])), ["id", "input_ids", "input_sha256", "max_new_tokens"])

    def test_each_input_sha256_is_the_sha256_of_the_decimal_ids_joined_by_commas(self):
        self.assertEqual(inputs.tokens_sha256([2, 105, 7]), hashlib.sha256(b"2,105,7").hexdigest())
        records = self.records()
        for record in records:
            text = ",".join(str(token) for token in record["input_ids"])
            self.assertEqual(record["input_sha256"], hashlib.sha256(text.encode("utf-8")).hexdigest())
        self.assertEqual(records[0]["input_ids"][:4], [2, 105, 1000 + ord("u"), 1000 + ord("s")])
        self.assertEqual([record["id"] for record in records], ["ja-choice-0001", "ja-choice-0002", "long-ja-4k-0001"])

    def test_an_item_that_differs_from_the_set_or_the_frame_stops_without_records(self):
        items, entries = items_and_entries()
        special = item("ja-choice-0002", "d<turn|>e")
        other_count = [dict(entry) for entry in entries]
        other_count[2]["token_count"] = 30
        cases = {"item hash": ([items[0], {**items[1], "max_new_tokens": 9}, items[2]], entries, None,
                               "differs from its item_sha256"),
                 "order": ([items[1], items[0], items[2]], entries, None, "not those of the set"),
                 "missing item": (items[:2], entries, None, "not those of the set"),
                 "special token in the content": ([items[0], special, items[2]],
                                                  [entries[0], {"id": special["id"], "item_sha256": digest(special)},
                                                   entries[2]], None, "special tokens outside the frame"),
                 "token count": (items, other_count, None, "not the token_count 30"),
                 "template": (items, entries, FakeTokenizer(lambda content: FRAME["prefix"] + content + "<turn|>\n"),
                              "differs from the frame"),
                 "first ids": (items, entries, FakeTokenizer(special={key: value for key, value in SPECIAL.items()
                                                                      if key != "<|turn>"}),
                               "ja-choice-0001: the input has special tokens outside the frame")}
        for label, (given, frozen, tokenizer, message) in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, message):
                inputs.records(tokenizer or FakeTokenizer(), given, frozen, FRAME, set(SPECIAL.values()), 2)

    def test_the_tokenizer_files_must_be_the_pinned_files(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder)
            files = {}
            for name in inputs.TOKENIZER_FILES:
                data = json.dumps({"added_tokens": [{"id": 2, "special": True}, {"id": 7, "special": False}]}).encode() \
                    if name == "tokenizer.json" else name.encode()
                (snapshot / name).write_bytes(data)
                files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            self.assertEqual(inputs.special_tokens(snapshot, {"files": files}), {2})
            (snapshot / "chat_template.jinja").write_bytes(b"x" * files["chat_template.jinja"]["bytes"])
            with self.assertRaises(ValueError):
                inputs.special_tokens(snapshot, {"files": files})

    def test_json_with_repeated_keys_or_non_standard_numbers_is_rejected(self):
        for text in (b'{"id": "a", "id": "b"}', b'{"value": NaN}', b'{"value": Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                inputs.strict(text)
        self.assertEqual(inputs.strict(b'{"value": 1}'), {"value": 1})


if __name__ == "__main__":
    unittest.main()
