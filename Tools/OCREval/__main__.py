import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".local" / "research" / "teleocr-eval"


def rounded(m):
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}


def main():
    parser = argparse.ArgumentParser(prog="python -m Tools.OCREval", description="TeleOCR evaluation set, reference runs and scoring.")
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make", help="Regenerate the evaluation images and ground truth.")
    make.add_argument("--out", type=Path)
    make.add_argument("--omnidocbench", action="store_true", help="Fetch and extract the pinned OmniDocBench subset instead.")
    score = sub.add_parser("score", help="Score reference outputs or OCR result JSON files.")
    score.add_argument("--selftest", action="store_true")
    score.add_argument("--set", type=Path, default=WORK / "set")
    source = score.add_mutually_exclusive_group()
    source.add_argument("--outputs", type=Path, help="Directory of <id>-<prompt>-<dtype>.txt reference outputs.")
    source.add_argument("--results", type=Path, help="Directory of <id>.json OCR pipeline results.")
    score.add_argument("--out", type=Path)
    run = sub.add_parser("run-reference", help="Run the original PyTorch model, one process per run, under the shared lock.")
    run.add_argument("--set", type=Path, default=WORK / "set")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--logs", type=Path, default=WORK / "logs")
    run.add_argument("--only", nargs="*")
    run.add_argument("--job", default="ocr-eval-reference")
    run.add_argument("--session", default="Tools.OCREval")
    run.add_argument("--wait-minutes", type=float, default=120)
    run.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--sample", help=argparse.SUPPRESS)
    run.add_argument("--prompt", choices=["text", "table", "layout"], help=argparse.SUPPRESS)
    run.add_argument("--crop", help=argparse.SUPPRESS)
    run.add_argument("--resize", type=int, help=argparse.SUPPRESS)
    run.add_argument("--name", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.command == "make":
        if args.omnidocbench:
            from .omnidocbench import make as make_public
            manifest = make_public(args.out or WORK / "omnidocbench")
        else:
            from .fixtures import make as make_set
            manifest = make_set(args.out or WORK / "set")
        for row in manifest["samples"]:
            print(row["id"], row["stratum"], row["image_sha256"][:12])
        print(f"{len(manifest['samples'])} samples")
        return 0
    if args.command == "score":
        from .score import score_set, selftest
        if args.selftest:
            selftest()
            return 0
        if args.outputs is None and args.results is None:
            parser.error("score needs --outputs, --results or --selftest")
        result = score_set(args.set, outputs=args.outputs, results=args.results)
        if not result["samples"]:
            print(f"no ground truth in {args.set}; run `python -m Tools.OCREval make` first", file=sys.stderr)
            return 2
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
        for stratum, agg in result["strata"].items():
            print(stratum, json.dumps(rounded(agg), ensure_ascii=False))
        if "f178" in result:
            print("f178", json.dumps(result["f178"]["judgments"], ensure_ascii=False), result["f178"]["overall"])
        return 0
    from .reference import run_one, run_plan
    if args.one:
        crop = [int(v) for v in args.crop.split(",")] if args.crop else None
        run_one(args.set.resolve(), args.out.resolve(), args.sample, args.prompt, crop, args.resize, args.name)
        return 0
    failed = run_plan(args.set.resolve(), args.out.resolve(), args.logs.resolve(), args.only, args.job, args.session, args.wait_minutes)
    print(json.dumps({"failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
