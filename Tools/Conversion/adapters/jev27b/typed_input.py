import argparse
import hashlib
from pathlib import Path
import sys

from ...storage import read_json, write_json
from .fixtures import SETS, load_rows, serialize
from .prepare import check_integrity, distributed_tokenizer, load_contract
from .typed_head import band, render, tokenize

FIELDS = ("id", "kind", "state", "question", "options")


def input_rows(entries, rows, encode):
    result = []
    for entry, row in zip(entries, rows, strict=True):
        count = len(encode(render(row["kind"], row["state"], row["question"], row["options"])))
        if entry["id"] != row["id"] or count != entry["tokens"] or band(count) != entry["band"]:
            raise ValueError(f"Token count differs from the fixture: {entry['id']}")
        result.append({field: row[field] for field in FIELDS})
    return result


def main():
    parser = argparse.ArgumentParser(description="Write coreai-verify typed-decide inputs for fixed JEV-27B sets.")
    for name in ("model", "snapshot", "corpus", "jglue", "fixtures", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sets", nargs="+", choices=SETS, required=True)
    args = parser.parse_args()
    if (args.output / "check.json").exists() or len(set(args.sets)) != len(args.sets):
        raise ValueError("Write each set once into a new directory")
    contract = load_contract(args.model)
    pins = read_json(args.fixtures / "jglue.json")
    check_integrity(args.corpus, contract["dataset"]["id"], contract["dataset"]["revision"])
    check_integrity(args.jglue, pins["repository"], pins["revision"])
    tokenizer = distributed_tokenizer(args.snapshot)
    limit = contract["limits"]["input_tokens"]
    report = {"passed": False, "sets": {}}
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        fixture = args.fixtures / f"{name}.json"
        entries = read_json(fixture)["rows"]
        rows = input_rows(entries, load_rows(fixture, args.corpus, args.jglue), lambda text: tokenize(tokenizer, text))
        data = serialize({"set": name, "rows": rows}).encode("utf-8")
        (args.output / f"input-{name}.json").write_bytes(data)
        report["sets"][name] = {"rows": len(rows), "rejected": sum(entry["tokens"] > limit for entry in entries),
                                "sha256": hashlib.sha256(data).hexdigest()}
    report["passed"] = True
    write_json(args.output / "check.json", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
