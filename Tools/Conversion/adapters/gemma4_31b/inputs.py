import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from Tools.LLMEvaluation.items import digest
from ...storage import write_json
from .prepare import TOKENIZER_FILES, load_contract, runtime_tokenizer
from .reference import render

OPTIONS = {"add_generation_prompt": True, "enable_thinking": False}


def strict(data):
    def pairs(entries):
        if len({key for key, _ in entries}) != len(entries):
            raise ValueError("Duplicate JSON key")
        return dict(entries)

    def constant(name):
        raise ValueError(f"Not a JSON value: {name}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def tokens_sha256(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode("utf-8")).hexdigest()


def line(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def special_tokens(snapshot, model):
    files = {}
    for name in TOKENIZER_FILES:
        files[name] = (Path(snapshot) / name).read_bytes()
        pinned = model["files"][name]
        if len(files[name]) != pinned["bytes"] or hashlib.sha256(files[name]).hexdigest() != pinned["sha256"]:
            raise ValueError(f"{name} differs from the contract")
    return {token["id"] for token in strict(files["tokenizer.json"])["added_tokens"] if token["special"]}


def records(tokenizer, items, entries, frame, special, bos):
    if not all(isinstance(value, dict) for value in items) or [value.get("id") for value in items] != \
            [entry["id"] for entry in entries]:
        raise ValueError("The items are not those of the set in its order")
    expected = sorted(token for token in tokenizer(frame["prefix"] + "x" + frame["suffix"],
                                                   add_special_tokens=False)["input_ids"] if token in special)
    found = []
    for value, entry in zip(items, entries):
        name = entry["id"]
        if digest(value) != entry["item_sha256"]:
            raise ValueError(f"{name}: the item differs from its item_sha256")
        text = tokenizer.apply_chat_template(value["messages"], tokenize=False, **OPTIONS)
        if text != frame["prefix"] + value["messages"][0]["content"] + frame["suffix"]:
            raise ValueError(f"{name}: the rendered input differs from the frame")
        ids, _ = render(tokenizer, {"id": name, "kind": "chat", "messages": value["messages"]}, bos)
        if ids[:2] != frame["tokenization"]["first_ids"] or sorted(token for token in ids if token in special) != expected:
            raise ValueError(f"{name}: the input has special tokens outside the frame")
        if "token_count" in entry and len(ids) != entry["token_count"]:
            raise ValueError(f"{name}: {len(ids)} tokens, not the token_count {entry['token_count']}")
        found.append({"id": name, "input_ids": ids, "input_sha256": tokens_sha256(ids),
                      "max_new_tokens": value["max_new_tokens"]})
    return found


def write(folder, found, contract):
    data = b"".join(map(line, found))
    folder = Path(folder)
    folder.mkdir(parents=True)
    temporary = folder / "inputs.jsonl.tmp"
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, folder / "inputs.jsonl")
    summary = {"contract": contract, "count": len(found), "sha256": hashlib.sha256(data).hexdigest()}
    write_json(folder / "inputs.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Render and tokenize the items of a quality evaluation set.")
    for name in ("model", "snapshot", "contract", "set", "items", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The input records must be written to a new folder")
    model = load_contract(args.model)
    special = special_tokens(args.snapshot, model)
    data = args.contract.read_bytes()
    contract = strict(data)
    items = [strict(row) for row in args.items.read_bytes().split(b"\n") if row]
    found = records(runtime_tokenizer(args.snapshot), items, strict(args.set.read_bytes())["items"],
                    contract["generation"]["template"], special, model["config"]["bos_token_id"])
    summary = write(args.output, found, {"version": contract["version"], "sha256": hashlib.sha256(data).hexdigest()})
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
