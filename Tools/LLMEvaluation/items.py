import gzip
import hashlib
import json
from pathlib import Path
import re

from Tools.LLMEvaluation.synthetic.generate import generate

ROOT = Path(__file__).resolve().parent
FILES = {"contract": "contract.json", "primary": "sets/primary.json", "reserve": "sets/reserve.json",
         "prompts": "prompts.json"}
VOCABULARY = "synthetic/vocabulary.json"
SETS = ("primary", "reserve")
QUESTION = re.compile(r"(a\d+)(p\d+)q\d+")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(root=ROOT, sets=True):
    root = Path(root)
    bundle = {"contract": read_json(root / FILES["contract"]), "prompts": read_json(root / FILES["prompts"]),
              "vocabulary": read_json(root / VOCABULARY), "resources": {VOCABULARY: sha256(root / VOCABULARY)}}
    if sets:
        bundle["sets"] = {name: read_json(root / FILES[name]) for name in SETS}
        bundle["hashes"] = {path: sha256(root / path) for path in FILES.values()}
        bundle["document"] = (root.parents[1] / bundle["contract"]["document"]).read_text(encoding="utf-8")
    return bundle


def pinned(datasets, source):
    data = (Path(datasets) / source["folder"] / source["path"]).read_bytes()
    if len(data) != source["bytes"] or hashlib.sha256(data).hexdigest() != source["sha256"]:
        raise ValueError(f"Source file differs from its pin: {source['folder']}/{source['path']}")
    return data


def unique(rows, key):
    found = {key(row): row for row in rows}
    if len(found) != len(rows):
        raise ValueError("Duplicate source keys")
    return found


def lines(data):
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def numbered(rows):
    return unique(rows, lambda row: int(row["task_id"].split("/")[1]))


def articles(data):
    found = {}
    for article in json.loads(data)["data"]:
        paragraphs, questions = {}, {}
        for paragraph in article["paragraphs"]:
            keys = set()
            for qa in paragraph["qas"]:
                match = QUESTION.fullmatch(qa["id"])
                if match is None:
                    raise ValueError(f"Unexpected question id: {qa['id']}")
                keys.add(match.group(1) + match.group(2))
                questions[qa["id"]] = {"paragraph": match.group(1) + match.group(2), "question": qa["question"],
                                       "answers": [answer["text"] for answer in qa["answers"]]}
            if len(keys) != 1 or next(iter(keys)) in paragraphs:
                raise ValueError("Paragraph keys are not unique")
            paragraphs[next(iter(keys))] = paragraph["context"]
        names = {key.split("p")[0] for key in paragraphs}
        if len(names) != 1 or next(iter(names)) in found:
            raise ValueError("Article keys are not unique")
        found[next(iter(names))] = {"paragraphs": paragraphs, "questions": questions}
    return found


PARSERS = {
    "jcqa": lambda data: unique(lines(data), lambda row: row["q_id"]),
    "jsquad": articles,
    "humaneval": lambda data: numbered(lines(gzip.decompress(data))),
    "mbpp": lambda data: unique(json.loads(data), lambda row: row["task_id"]),
    "humanevalpack": lambda data: numbered(lines(data)),
    "ner": lambda data: dict(enumerate(json.loads(data))),
}


class Sources:
    def __init__(self, contract, datasets):
        self.contract, self.datasets, self.cache = contract, datasets, {}

    def __getitem__(self, name):
        if name not in self.cache:
            if self.datasets is None:
                raise ValueError("Public data items need --datasets")
            self.cache[name] = PARSERS[name](pinned(self.datasets, self.contract["sources"][name]))
        return self.cache[name]


def paragraph(data, key):
    return data["jsquad"][key.split("p")[0]]["paragraphs"][key]


def content(category, source, data, bundle):
    prompts, contract = bundle["prompts"], bundle["contract"]
    kind = category["kind"]
    if kind == "choice":
        row = data["jcqa"][source["q_id"]]
        choices = {f"choice{index}": row[f"choice{index}"] for index in range(5)}
        return prompts["ja-choice"].format(question=row["question"], **choices), row["label"], {"kind": "choice_digit"}
    if kind == "reading":
        question = data["jsquad"][source["article"]]["questions"][source["question"]]
        keys = source.get("paragraphs", [question["paragraph"]])
        if keys.count(question["paragraph"]) != 1:
            raise ValueError("The question paragraph must appear once")
        text = prompts["ja-reading"].format(context="\n\n".join(paragraph(data, key) for key in keys),
                                            question=question["question"])
        scorer = {"kind": "reading_exact", **({"stratum": source["position"]} if "position" in source else {})}
        return text, question["answers"], scorer
    if kind == "humaneval":
        row = data["humaneval"][source["task"]]
        return (prompts["code-gen-he"].format(prompt=row["prompt"]), None,
                {"kind": "python_tests", "setup": row["prompt"], "test": row["test"],
                 "call": f"check({row['entry_point']})"})
    if kind == "mbpp":
        row = data["mbpp"][source["task_id"]]
        tests = "\n".join(row["test_imports"] + row["test_list"])
        return (prompts["code-gen-mbpp"].format(task=row["prompt"], tests=tests), None,
                {"kind": "python_tests", "setup": "\n".join(row["test_imports"]), "test": "\n".join(row["test_list"]),
                 "call": ""})
    if kind == "humanevalfix":
        row = data["humanevalpack"][source["task"]]
        return (prompts["code-fix"].format(code=row["declaration"] + row["buggy_solution"], tests=row["test"].strip()),
                None, {"kind": "python_tests", "setup": "", "test": row["test"], "call": ""})
    if kind == "entities":
        row = data["ner"][source["index"]]
        keys = contract["entity_keys"]
        if row["curid"] != source["curid"] or any(entity["type"] not in keys for entity in row["entities"]):
            raise ValueError(f"Unexpected NER row: {source['index']}")
        answer = {key: sorted({entity["name"] for entity in row["entities"] if entity["type"] == key}) for key in keys}
        quoted = "".join(prompts["entity_key"].format(key=key) for key in keys)
        return prompts["extract-ja"].format(keys=quoted, text=row["text"]), answer, {"kind": "entity_sets"}
    if kind == "synthetic":
        made = generate(category, source["index"], contract["seed"], prompts, bundle["vocabulary"], source.get("records"))
        if made["template"] != source["template"]:
            raise ValueError(f"Template differs: {source}")
        return made["content"], made["answer"], made["scorer"]
    if kind == "free":
        row = {row["id"]: row for row in prompts["free-ja"]}[source["instruction"]]
        return row["text"], None, {"kind": "unscored"}
    raise ValueError(f"Unknown category kind: {kind}")


def item(category, entry, data, bundle):
    text, answer, scorer = content(category, entry["source"], data, bundle)
    if not text or text != text.strip():
        raise ValueError(f"User content must not start or end with whitespace: {entry['id']}")
    value = {"id": entry["id"], "category": category["id"], "unit": category["unit"],
             "messages": [{"role": "user", "content": text}], "max_new_tokens": category["max_new_tokens"],
             "scorer": scorer, "answer": answer}
    if "token_range" in category:
        value["token_count"] = entry["token_count"]
    return value


def build(bundle, datasets, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output must be a new folder: {output}")
    data = Sources(bundle["contract"], datasets)
    categories = {category["id"]: category for category in bundle["contract"]["categories"]}
    built, differs, kinds = {}, [], []
    for name in SETS:
        built[name] = []
        for entry in bundle["sets"][name]["items"]:
            value = item(categories[entry["category"]], entry, data, bundle)
            if digest(value) != entry["item_sha256"]:
                differs.append(entry["id"])
            if value["scorer"]["kind"] != categories[entry["category"]]["scorer"]:
                kinds.append(entry["id"])
            built[name].append(value)
    if kinds:
        raise ValueError(f"{len(kinds)} item scorer kinds differ from their categories, first: {kinds[:5]}")
    if differs:
        raise ValueError(f"{len(differs)} item hashes differ from the sets, first: {differs[:5]}")
    output.mkdir(parents=True)
    manifest = {"contract_version": bundle["contract"]["version"], "frozen_files": bundle["hashes"], "items": {},
                "files": {}}
    for name, rows in built.items():
        path = output / f"{name}.jsonl"
        path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
        manifest["items"][name] = len(rows)
        manifest["files"][path.name] = sha256(path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
