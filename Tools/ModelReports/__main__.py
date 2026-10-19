import argparse
import json
import sys
from pathlib import Path

from .reports import ROOT, check, load, render, summarize, template


def main():
    parser = argparse.ArgumentParser(description="Validate and render model evaluation records.")
    parser.add_argument("command", choices=["check", "render", "template", "summarize"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--kind", default="llm")
    parser.add_argument("--model", default="new-model")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--require-complete")
    args = parser.parse_args()
    try:
        spec, catalog, models = load(args.root)
        if args.command == "template":
            print(json.dumps(template(spec, args.kind, args.model), ensure_ascii=False, indent=2))
        elif args.command == "summarize":
            if args.input is None:
                raise ValueError("summarize requires --input")
            print(json.dumps(summarize(args.input, spec), indent=2, allow_nan=False))
        elif args.command == "render":
            render(args.root, spec, catalog, models, write=True)
            print(f"Rendered {len(models)} bilingual model reports and both README tables.")
        else:
            check(args.root, spec, catalog, models, args.require_complete)
            print(f"MODEL REPORTS PASS: {len(models)} records, evidence, links and generated content.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"MODEL REPORTS FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
