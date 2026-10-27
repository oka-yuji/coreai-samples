import argparse
import hashlib
import io
import json
from pathlib import Path

from .evaluation import canonical_hash, quality_summary, relative
from .fixtures import quality_axes
from .mobile_evidence import mobile_fixtures
from .mobile_ratings import rating_identity_matches


COHORTS = ("A0", "A1", "B", "C")


def bytes_record(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def checked_bytes(root, name, expected=None):
    path = Path(root)
    for part in relative(name).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Review inputs cannot contain symbolic links")
    data = path.read_bytes()
    record = bytes_record(data)
    if expected is not None and record != expected:
        raise ValueError("Review input differs from its recorded identity: " + name)
    return data, record


def read_json(root, name, expected=None):
    data, record = checked_bytes(root, name, expected)
    return json.loads(data), record


def validate_ratings(ratings, cohorts):
    if ratings.get("format") != 2 or set(ratings.get("cohorts", {})) != set(COHORTS):
        raise ValueError("All four quality cohorts are required")
    axes = quality_axes()
    for name in COHORTS:
        cohort = ratings["cohorts"][name]
        if not rating_identity_matches(cohort.get("identity"), cohorts[name]):
            raise ValueError("Ratings belong to a different output cohort")
        rows = cohort.get("ratings", [])
        if len(rows) != len(axes) or {row.get("run_id") for row in rows} != set(axes):
            raise ValueError("Every frozen output must retain a rating row")
        scores = {row["run_id"]: row["axes"] for row in rows}
        quality_summary(scores, scores, axes)


def load_review_guide():
    recipes = Path(__file__).parents[2] / "recipes"
    path = recipes / "qwen_image21_mobile/quality-review-ja.json"
    data = path.read_bytes()
    guide = json.loads(data)
    definition = json.loads((recipes / "qwen_image21/evaluation-cases.json").read_text())
    cases = {case["id"]: case for case in definition["cases"]}
    if guide.get("format") != 1 or guide.get("language") != "ja" or set(guide.get("cases", {})) != set(cases):
        raise ValueError("Review guidance must cover the frozen cases")
    if set(guide.get("cohorts", {})) != set(COHORTS) or set(guide.get("axes", {})) != {
            "instruction", "detail", "text", "preservation", "alpha"}:
        raise ValueError("Review guidance changed the cohorts or axes")
    for key, case in cases.items():
        display = guide["cases"][key]
        if (display.get("source_prompt") != case["prompt"] or not isinstance(display.get("prompt_ja"), str)
                or not display["prompt_ja"].strip() or set(display.get("checks", {})) != set(case["axes"])):
            raise ValueError("Review guidance differs from its frozen prompt or axes: " + key)
        for points in display["checks"].values():
            if not isinstance(points, list) or not points or any(not isinstance(point, str) or not point.strip() for point in points):
                raise ValueError("Review guidance requires concrete inspection points: " + key)
    return guide, bytes_record(data)


def copied_png(root, name, expected, output, destination):
    from PIL import Image

    data, record = checked_bytes(root, name, expected)
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.size != (512, 512):
            raise ValueError("Review requires the recorded 512-square PNG")
        image.load()
    target = output / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)
    return record


def build(args):
    roots = {"A0": args.baseline, "A1": args.mixed_reference, "B": args.reference, "C": args.verification}
    target = args.output.resolve()
    for source in [args.comparison, args.fixtures, *roots.values()]:
        resolved = source.resolve()
        if target == resolved or target in resolved.parents or resolved in target.parents:
            raise ValueError("The review output must be separate from its inputs")
        if source.is_symlink():
            raise ValueError("Review input roots cannot be symbolic links")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Review output must be a new directory")
    comparison, comparison_file = read_json(args.comparison, "comparison.json")
    cohorts = comparison.get("cohorts", {})
    if comparison.get("format") != 2 or set(cohorts) != set(COHORTS):
        raise ValueError("The review requires a complete four-cohort comparison")
    ratings, ratings_file = read_json(args.comparison, "quality-ratings.json", comparison["quality_file"])
    validate_ratings(ratings, cohorts)
    manifest, _ = read_json(args.fixtures, "fixtures.json")
    mobile_fixtures(args.fixtures, manifest)
    expected = {row["run_id"]: row for row in manifest["runs"]}
    rows = comparison.get("runs", [])
    if len(rows) != 60 or {row.get("run_id") for row in rows} != set(expected):
        raise ValueError("The comparison must retain every frozen output")
    guide, guide_file = load_review_guide()
    metadata_by_run = {}
    for row in rows:
        if canonical_hash(row["input"]) != canonical_hash(expected[row["run_id"]]):
            raise ValueError("The comparison input differs from the frozen fixture")
        key = row["run_id"]
        identifier = expected[key]["fixture_id"]
        metadata_name = identifier + "/input.json"
        metadata, _ = read_json(args.fixtures, metadata_name, manifest["inventory"][metadata_name])
        if metadata["prompt"] != guide["cases"][identifier]["source_prompt"]:
            raise ValueError("The displayed review guidance belongs to another prompt: " + key)
        metadata_by_run[key] = metadata
    outputs = {}
    for name in COHORTS:
        filename = "verification.json" if name == "C" else "summary.json"
        binding = "verification_file" if name == "C" else "summary_file"
        report, _ = read_json(roots[name], filename, cohorts[name][binding])
        if (name == "C" and report.get("numeric_passed") is not True) or (
                name != "C" and report.get("complete") is not True):
            raise ValueError("The reviewed cohort is not complete")
        recorded = report.get("runs", [])
        if len(recorded) != 60 or {row.get("run_id") for row in recorded} != set(expected):
            raise ValueError("The output cohort must retain every frozen input")
        outputs[name] = {}
        for row in recorded:
            key = row["run_id"]
            if row.get("status") != "completed" or canonical_hash(row["input"]) != canonical_hash(expected[key]):
                raise ValueError("The reviewed image belongs to another run or input")
            path = ("rollout/" if name == "C" else "") + key + "/output.png"
            record = report["evidence_files"][path] if name == "C" else row["files"]["output.png"]
            outputs[name][key] = (path, record)
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    images = {}
    for key, run in expected.items():
        metadata = metadata_by_run[key]
        display = guide["cases"][run["fixture_id"]]
        entry = {"run_id": key, "prompt": metadata["prompt"], "seed": run["seed"], "images": {},
                 "guidance": {"prompt_ja": display["prompt_ja"], "checks": display["checks"]}}
        reference = metadata.get("reference")
        if reference:
            destination = "images/reference-" + run["fixture_id"] + ".png"
            if destination not in images:
                images[destination] = copied_png(
                    args.fixtures, reference, manifest["inventory"][reference], args.output, destination)
            entry["reference"] = destination
        for name in COHORTS:
            path, record = outputs[name][key]
            destination = "images/" + name + "/" + key + ".png"
            images[destination] = copied_png(roots[name], path, record, args.output, destination)
            entry["images"][name] = destination
        cases.append(entry)
    payload = {"format": 1, "comparison_sha256": comparison_file["sha256"], "cases": cases, "ratings": ratings,
               "review": {key: guide[key] for key in ("language", "ui", "axes", "cohorts")}}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).parents[2] / "recipes/qwen_image21_mobile/quality-review.html"
    template_data = template.read_bytes()
    html = template_data.decode().replace("__REVIEW_PAYLOAD__", encoded)
    with (args.output / "index.html").open("x") as stream:
        stream.write(html)
    receipt = {"format": 1, "comparison_file": comparison_file, "initial_ratings_file": ratings_file,
               "images": images, "html": bytes_record(html.encode()), "display_guide_file": guide_file,
               "template_file": bytes_record(template_data), "complete": True,
               "scope": "Offline human-rating tool; does not independently certify conversion or quality"}
    with (args.output / "review-receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    for name in ("comparison", "fixtures", "baseline", "mixed-reference", "reference", "verification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
