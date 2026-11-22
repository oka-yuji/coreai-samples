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
    differs, versions = [], {}
    for name, document in documents.items():
        path = items.ROOT / items.FILES[name]
        if arguments.check:
            same, note = selection.compare(document, path.read_text(encoding="utf-8")) if path.is_file() else (False, None)
            if not same:
                differs.append(items.FILES[name])
            if note:
                versions[items.FILES[name]] = note
        else:
            path.write_text(selection.serialize(document), encoding="utf-8")
    print(json.dumps({"tokens": selection.statistics_of(counts), "differs": differs, "versions": versions}, indent=2,
                     ensure_ascii=False))
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


def score(arguments):
    from Tools.LLMEvaluation import scoring

    bundle = items.load()
    found = checks.problems(bundle)
    if found:
        for problem in found:
            print(f"FAIL: {problem}")
        return 1
    tokenizer, pinned = scoring.pinned(arguments.tokenizer, bundle["contract"])
    print(json.dumps(scoring.score_runs(bundle, arguments.items, arguments.run, arguments.output, tokenizer, pinned),
                     indent=2, sort_keys=True))
    return 0


def compare(arguments):
    from Tools.LLMEvaluation import comparison

    bundle = items.load()
    found = checks.problems(bundle)
    if found:
        for problem in found:
            print(f"FAIL: {problem}")
        return 1
    print(json.dumps(comparison.compare_runs(bundle, arguments.items, arguments.baseline, arguments.candidate,
                                             arguments.scores, arguments.output, arguments.rerun_scores,
                                             arguments.layer1, arguments.substitute_reason), indent=2))
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
    scored = commands.add_parser("score")
    scored.add_argument("--items", type=Path, required=True)
    scored.add_argument("--run", type=Path, nargs="+", required=True)
    scored.add_argument("--tokenizer", type=Path, required=True)
    scored.add_argument("--output", type=Path, required=True)
    compared = commands.add_parser("compare")
    for name in ("--baseline", "--candidate", "--scores", "--items", "--output"):
        compared.add_argument(name, type=Path, required=True)
    compared.add_argument("--rerun-scores", type=Path)
    compared.add_argument("--layer1", type=Path)
    compared.add_argument("--substitute-reason")
    arguments = parser.parse_args()
    return {"validate": validate, "select": select, "build": build, "score": score,
            "compare": compare}[arguments.command](arguments)


if __name__ == "__main__":
    sys.exit(main())
