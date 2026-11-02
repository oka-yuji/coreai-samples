import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

from ...storage import read_json, write_json
from .prepare import check_integrity, distributed_tokenizer, load_contract
from .typed_head import BANDS, FIXED_OPTIONS, band, render, tokenize

SEED = 20261102
SPLITS = ("test_set_30k", "test", "calibration")
KINDS = ("noul", "choice", "score")
TASKS = ("JNLI", "JCommonsenseQA", "JSTS")
SETS = ("F", "Q", "Cal", "H", "J")
PLACEHOLDER = "yuri_v1"
SEPARATOR = "\n\n"
JAPANESE_IN_F = 8
JAPANESE_IN_J = 200
CARD_EXAMPLES = {
    "card-example:refund": {
        "kind": "noul", "state": "Customer says the parcel arrived damaged and wants their money back.",
        "question": "Is the customer asking for a refund?", "options": ["false", "true"],
        "card_probabilities": [0.022, 0.978]},
    "card-example:sku": {
        "kind": "choice", "state": "SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.",
        "question": "Supplier response for this scenario.",
        "options": ["issue_warning", "renegotiate", "dual_source", "maintain"],
        "card_probabilities": [0.2541, 0.1253, 0.62, 0.0007]},
}


def digest(row):
    options = json.dumps(row["options"], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256((row["state"] + "\n" + row["question"] + "\n" + options).encode("utf-8")).hexdigest()


def read_split(corpus, split):
    rows = {}
    with (Path(corpus) / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["id"] in rows:
                    raise ValueError(f"Duplicate corpus id in {split}: {row['id']}")
                rows[row["id"]] = row
    return rows


def read_task(directory, pins, task):
    pin = pins["files"][task]
    data = (Path(directory) / pin["path"]).read_bytes()
    if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError(f"JGLUE file differs from its pin: {task}")
    key = "q_id" if task == "JCommonsenseQA" else "sentence_pair_id"
    records = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    found = {str(record[key]): record for record in records}
    if len(found) != len(records):
        raise ValueError(f"Duplicate JGLUE id in {task}")
    return found


def jglue_row(task, record, pins):
    if task == "JNLI":
        return {"kind": "choice", "state": record["sentence1"], "question": record["sentence2"],
                "options": list(pins["jnli_options"]), "label": pins["jnli_labels"].index(record["label"])}
    if task == "JCommonsenseQA":
        return {"kind": "choice", "state": "", "question": record["question"],
                "options": [record[f"choice{index}"] for index in range(5)], "label": record["label"]}
    if task == "JSTS":
        return {"kind": "score", "state": record["sentence1"], "question": record["sentence2"],
                "options": list(FIXED_OPTIONS["score"]), "label": record["label"]}
    raise ValueError(f"Unknown JGLUE task: {task}")


class Sources:
    def __init__(self, corpus, jglue=None, pins=None):
        self.corpus, self.jglue, self.pins = corpus, jglue, pins
        self.splits, self.tasks = {}, {}

    def split(self, name):
        if name not in self.splits:
            self.splits[name] = read_split(self.corpus, name)
        return self.splits[name]

    def task(self, name):
        if self.jglue is None or self.pins is None:
            raise ValueError("JGLUE rows need the pinned JGLUE directory")
        if name not in self.tasks:
            self.tasks[name] = read_task(self.jglue, self.pins, name)
        return self.tasks[name]

    def restore(self, entry):
        source = entry["source"]
        if source in SPLITS:
            row = self.split(source)[entry["id"]]
            restored = {key: row[key] for key in ("kind", "state", "question", "options", "target")}
        elif source.startswith("jglue:"):
            _, task, native = entry["id"].split(":", 2)
            restored = jglue_row(task, self.task(task)[native], self.pins)
        elif source in ("synthetic-long", "boundary"):
            parts = [self.split("test_set_30k")[part] for part in entry["parts"]]
            states = [part["state"] for part in parts]
            if entry["cut"] is not None:
                states[-1] = states[-1][:entry["cut"]]
            restored = {key: parts[0][key] for key in ("kind", "question", "options")}
            restored["state"] = SEPARATOR.join(states)
            if len(parts) == 1 and entry["cut"] is None:
                restored["target"] = parts[0]["target"]
        elif source == "card-example":
            restored = {key: CARD_EXAMPLES[entry["id"]][key] for key in ("kind", "state", "question", "options")}
        else:
            raise ValueError(f"Unknown fixture source: {source}")
        return {"id": entry["id"], **restored}


def load_rows(fixture, corpus_dir, jglue_dir=None):
    data = read_json(fixture)
    pins = read_json(Path(fixture).parent / "jglue.json")
    sources = Sources(corpus_dir, jglue_dir, pins)
    rows = []
    for entry in data["rows"]:
        row = sources.restore(entry)
        if digest(row) != entry["sha256"] or row["kind"] != entry["kind"] or len(row["options"]) != entry["n_options"]:
            raise ValueError(f"Restored row differs from the fixture: {entry['id']}")
        rows.append(row)
    return rows


def eligible(rows, kind):
    return sorted((row for row in rows.values() if row["kind"] == kind and row["source"] != PLACEHOLDER),
                  key=lambda row: row["id"])


def stratified(rows, per_kind, rng, excluded):
    chosen = []
    for kind in KINDS:
        pool = [row for row in eligible(rows, kind) if row["id"] not in excluded]
        families = sorted({row["family"] for row in pool})
        quota, extra = divmod(per_kind, len(families))
        bonus = set(rng.sample(families, extra))
        for family in families:
            chosen.extend(rng.sample([row for row in pool if row["family"] == family], quota + (family in bonus)))
    return chosen


def corpus_entry(row, source):
    return {"id": row["id"], "source": source, "family": row["family"], "parts": None, "cut": None}


def jglue_entry(task, key):
    return {"id": f"jglue:{task}:{key}", "source": f"jglue:{task}", "family": None, "parts": None, "cut": None}


def japanese(sources, task):
    return random.Random(f"{SEED}:J:{task}").sample(sorted(sources.task(task)), JAPANESE_IN_F + JAPANESE_IN_J)


def joined_entry(identifier, source, parts, cut=None):
    return {"id": identifier, "source": source, "family": parts[0]["family"], "parts": [part["id"] for part in parts],
            "cut": cut}


class Builder:
    def __init__(self, sources, tokenizer):
        self.sources, self.tokenizer = sources, tokenizer

    def tokens(self, row):
        return len(tokenize(self.tokenizer, render(row["kind"], row["state"], row["question"], row["options"])))

    def finish(self, entry):
        row = self.sources.restore(entry)
        tokens = self.tokens(row)
        return {**entry, "kind": row["kind"], "n_options": len(row["options"]), "sha256": digest(row), "tokens": tokens,
                "band": band(tokens), "has_target": "target" in row or "label" in row}

    def grow(self, identifier, base, pool, low):
        parts = [base]
        while self.finish(joined_entry(identifier, "synthetic-long", parts))["tokens"] < low:
            parts.append(pool.pop())
        return parts

    def exact(self, identifier, base, pool, target):
        parts = [base]
        while self.finish(joined_entry(identifier, "boundary", parts))["tokens"] < target:
            parts.append(pool.pop())
        while True:
            def count(cut):
                return self.finish(joined_entry(identifier, "boundary", parts, cut))["tokens"]

            low, high = 0, len(parts[-1]["state"])
            while low < high:
                middle = (low + high + 1) // 2
                low, high = (middle, high) if count(middle) <= target else (low, middle - 1)
            for cut in range(max(0, low - 64), min(len(parts[-1]["state"]), low + 64) + 1):
                if count(cut) == target:
                    return joined_entry(identifier, "boundary", parts, cut)
            parts[-1] = pool.pop()


def build_f(builder, sources):
    data = sources.split("test_set_30k")
    rng = random.Random(f"{SEED}:F")
    entries = [corpus_entry(row, "test_set_30k") for row in stratified(data, 80, rng, set())]
    used = {entry["id"] for entry in entries}
    entries += [jglue_entry(task, key) for task in TASKS for key in japanese(sources, task)[:JAPANESE_IN_F]]
    pool = [row for kind in KINDS for row in eligible(data, kind) if row["id"] not in used]
    rng.shuffle(pool)
    kinds = (KINDS * 4)[:10]
    for (_, previous), (label, limit) in zip(BANDS, BANDS[1:]):
        for index, kind in enumerate(kinds):
            base = next(row for row in pool if row["kind"] == kind)
            pool.remove(base)
            identifier = f"synthetic-long:{label}:{index}"
            parts = builder.grow(identifier, base, pool, rng.randint(previous + 1, limit - 512))
            entry = joined_entry(identifier, "synthetic-long", parts)
            if builder.finish(entry)["band"] != label:
                raise ValueError(f"Synthetic row left its band: {identifier}")
            entries.append(entry)
    for options in (2, 16):
        rows = [row for row in pool if row["kind"] == "choice" and len(row["options"]) == options]
        for index, row in enumerate(rng.sample(rows, 2)):
            pool.remove(row)
            entries.append(joined_entry(f"boundary:choice{options}:{index}", "boundary", [row]))
    for target, kind in ((BANDS[-1][1], "noul"), (BANDS[-1][1] + 1, "choice")):
        base = next(row for row in pool if row["kind"] == kind)
        pool.remove(base)
        entries.append(builder.exact(f"boundary:tokens{target}", base, pool, target))
    entries += [{"id": key, "source": "card-example", "family": None, "parts": None, "cut": None} for key in CARD_EXAMPLES]
    return entries


def build_sets(sources, tokenizer):
    builder = Builder(sources, tokenizer)
    f = [builder.finish(entry) for entry in build_f(builder, sources)]
    excluded = {entry["id"] for entry in f if entry["source"] == "test_set_30k"} | {
        part for entry in f for part in entry["parts"] or []}
    q = stratified(sources.split("test_set_30k"), 300, random.Random(f"{SEED}:Q"), excluded)
    sets = {"F": f, "Q": [builder.finish(corpus_entry(row, "test_set_30k")) for row in q]}
    for name, split, count in (("Cal", "calibration", 1000), ("H", "test", 500)):
        rng = random.Random(f"{SEED}:{name}")
        rows = sources.split(split)
        sets[name] = [builder.finish(corpus_entry(row, split))
                      for kind in KINDS for row in rng.sample(eligible(rows, kind), count)]
    sets["J"] = [builder.finish(jglue_entry(task, key)) for task in TASKS for key in japanese(sources, task)[JAPANESE_IN_F:]]
    return sets


def summary(rows):
    return {"rows": len(rows), "kind": dict(sorted(Counter(row["kind"] for row in rows).items())),
            "source": dict(sorted(Counter(row["source"] for row in rows).items())),
            "band": dict(sorted(Counter(row["band"] or "rejected" for row in rows).items()))}


def serialize(document):
    head = json.dumps({key: value for key, value in document.items() if key != "rows"}, indent=2, sort_keys=True)
    rows = ",\n".join("    " + json.dumps(row, sort_keys=True) for row in document["rows"])
    return head[:-2] + ',\n  "rows": [\n' + rows + "\n  ]\n}\n"


def documents(sets, contract, pins, tokenizer_sha256):
    provenance = {"dataset": contract["dataset"], "jglue": {"repository": pins["repository"], "revision": pins["revision"]},
                  "tokenizer_sha256": tokenizer_sha256}
    return {name: {"name": name, "version": 1, "seed": SEED, "sources": provenance, "summary": summary(rows), "rows": rows}
            for name, rows in sets.items()}


def check(expected, generated, corpus, jglue):
    report = {"sets": {}}
    passed = True
    for name in SETS:
        path = Path(expected) / f"{name}.json"
        identical = path.is_file() and path.read_bytes() == (Path(generated) / f"{name}.json").read_bytes()
        rows = load_rows(path, corpus, jglue) if path.is_file() else []
        report["sets"][name] = {"identical": identical, "restored": len(rows)}
        passed &= identical and bool(rows)
    data = {name: read_json(path)["rows"] for name in SETS if (path := Path(expected) / f"{name}.json").is_file()}
    if set(data) == set(SETS):
        f_corpus = {row["id"] for row in data["F"] if row["source"] == "test_set_30k"} | {
            part for row in data["F"] for part in row["parts"] or []}
        f_japanese = [row for row in data["F"] if row["source"].startswith("jglue:")]
        report["disjoint"] = {
            "F_Q_ids": not f_corpus & {row["id"] for row in data["Q"]},
            "F_Q_sha256": not {row["sha256"] for row in data["F"]} & {row["sha256"] for row in data["Q"]},
            "F_J_ids": not {row["id"] for row in f_japanese} & {row["id"] for row in data["J"]},
            "F_J_sha256": not {row["sha256"] for row in f_japanese} & {row["sha256"] for row in data["J"]}}
        passed &= all(report["disjoint"].values())
    report["passed"] = bool(passed)
    return report


def placeholder_rows(sets, sources):
    count = 0
    for rows in sets.values():
        for entry in rows:
            for part in [entry["id"]] if entry["source"] in SPLITS else entry["parts"] or []:
                split = entry["source"] if entry["source"] in SPLITS else "test_set_30k"
                count += sources.split(split)[part]["source"] == PLACEHOLDER
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate or check the fixed JEV-27B evaluation sets.")
    for name in ("model", "snapshot", "corpus", "jglue", "fixtures", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.model)
    pins = read_json(args.fixtures / "jglue.json")
    integrity = {"corpus": check_integrity(args.corpus, contract["dataset"]["id"], contract["dataset"]["revision"]),
                 "jglue": check_integrity(args.jglue, pins["repository"], pins["revision"])}
    if any((args.output / f"{name}.json").exists() for name in SETS):
        raise FileExistsError("Fixtures must be generated into a new directory")
    sources = Sources(args.corpus, args.jglue, pins)
    tokenizer = distributed_tokenizer(args.snapshot)
    sets = build_sets(sources, tokenizer)
    placeholders = placeholder_rows(sets, sources)
    if placeholders:
        raise ValueError("Placeholder rows must not enter the fixed sets")
    tokenizer_sha256 = hashlib.sha256((args.snapshot / "tokenizer.json").read_bytes()).hexdigest()
    for name, document in documents(sets, contract, pins, tokenizer_sha256).items():
        (args.output / f"{name}.json").write_text(serialize(document), encoding="utf-8")
    if args.check:
        report = check(args.fixtures, args.output, args.corpus, args.jglue)
        report.update(integrity=integrity, placeholder_rows=placeholders)
        write_json(args.output / "check.json", report)


if __name__ == "__main__":
    sys.exit(main())
