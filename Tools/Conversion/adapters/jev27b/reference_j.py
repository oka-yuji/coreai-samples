import argparse
from pathlib import Path
import sys
import time

from ...storage import read_json, write_json
from .decision import sha256
from .japanese import GROUPS, load_wrappers, plan, restore, tokenized
from .prepare import check_integrity, distributed_tokenizer, load_contract
from .reference import decision_head
from .reference_sets import merged_stage
from .typed_head import tokenize

FIELDS = ("id", "source_id", "task", "group", "variant", "permutation", "order", "kind", "options", "label",
          "original_label", "band", "row_sha256", "text_sha256", "token_count", "tokens", "rejected")


def reference_rows(args, contract, tokenizer):
    pins = read_json(args.fixtures / "jglue.json")
    check_integrity(args.jglue, pins["repository"], pins["revision"])
    units = restore(read_json(args.fixtures / "J.json")["rows"], pins, args.jglue)
    rows = tokenized(plan(units, load_wrappers(args.wrappers), GROUPS), lambda text: tokenize(tokenizer, text),
                     contract["limits"]["input_tokens"])
    return [{"set": "J", **{field: row[field] for field in FIELDS}, "last_token": tokenizer.decode(row["tokens"][-1:])}
            for row in rows]


def main():
    parser = argparse.ArgumentParser(description="Run the wrapped rows of J through the merged JEV-27B checkpoint "
                                                 "(stage 1').")
    for name in ("model", "merge", "snapshot", "jglue", "fixtures", "wrappers", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    output = args.output / "reference-merged-J.json"
    if output.exists():
        raise FileExistsError(f"{output.name} must be written to a new directory")
    contract = load_contract(args.model)
    started = time.perf_counter()
    head = decision_head(args.snapshot, contract)
    rows = reference_rows(args, contract, distributed_tokenizer(args.merge / "checkpoint"))
    prepared = time.perf_counter()
    document = merged_stage(args, contract, head, rows, started, prepared)
    write_json(output, {**document, "sets": ["J"], "wrappers_sha256": sha256(args.wrappers), "rows": rows})
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
