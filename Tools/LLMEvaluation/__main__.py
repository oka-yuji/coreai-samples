import argparse
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from Tools.LLMEvaluation import checks, items


def validate(arguments):
    bundle = items.load()
    found = checks.problems(bundle)
    for problem in found:
        print(f"FAIL: {problem}")
    if found:
        return 1
    counts = {name: len(bundle["sets"][name]["items"]) for name in items.SETS}
    print(f"PASS: {bundle['contract']['version']} contract, sets and document agree ({json.dumps(counts)})")
    return 0


def select(arguments):
    from Tools.LLMEvaluation import selection

    bundle = items.load(sets=False)
    template = arguments.template or arguments.tokenizer.with_name("chat_template.jinja")
    counter = selection.Counter(bundle["contract"], arguments.tokenizer, template)
    rows, counts = selection.choose(bundle, arguments.datasets, counter)
    documents = selection.documents(bundle, rows, counter)
    differs = []
    for name, document in documents.items():
        path = items.ROOT / items.FILES[name]
        data = selection.serialize(document)
        if arguments.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != data:
                differs.append(items.FILES[name])
        else:
            path.write_text(data, encoding="utf-8")
    print(json.dumps({"tokens": selection.statistics_of(counts), "differs": differs}, indent=2, ensure_ascii=False))
    return 1 if differs else 0


def build(arguments):
    bundle = items.load()
    found = checks.problems(bundle)
    if found:
        for problem in found:
            print(f"FAIL: {problem}")
        return 1
    print(json.dumps(items.build(bundle, arguments.datasets, arguments.output), indent=2, sort_keys=True))
    return 0


def main():
    parser = argparse.ArgumentParser(prog="python3 -m Tools.LLMEvaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    chosen = commands.add_parser("select")
    chosen.add_argument("--datasets", type=Path, required=True)
    chosen.add_argument("--tokenizer", type=Path, required=True)
    chosen.add_argument("--template", type=Path)
    chosen.add_argument("--check", action="store_true")
    built = commands.add_parser("build")
    built.add_argument("--datasets", type=Path, required=True)
    built.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    return {"validate": validate, "select": select, "build": build}[arguments.command](arguments)


if __name__ == "__main__":
    sys.exit(main())
