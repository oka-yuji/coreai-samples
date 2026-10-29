import argparse
import copy
import json
from pathlib import Path

from .evaluation import canonical_hash
from .fixtures import file_record, quality_axes
from .mobile_acceptance import OWNER_REVIEW, OWNER_REVIEW_PATH
from .mobile_gallery import COHORTS, bytes_record, checked_bytes, read_json, validate_ratings
from .mobile_profile import NATIVE_COMPLEX_CANDIDATE


REVIEWED_GALLERY_RECEIPT = {"bytes": 38598, "sha256": "8595efe52d5930a784a949fef3ad6630f134d83cdbe31914ad00f0a9851fca93"}
EQUIVALENCE_FILES = ("mobile_visual_equivalence.py", "mobile_gallery.py", "evaluation.py")


def cohort_roots(values):
    roots = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or name not in COHORTS or name in roots or not path:
            raise ValueError("Each of A0, A1, B and C requires one distinct root binding")
        roots[name] = Path(path)
    if set(roots) != set(COHORTS):
        raise ValueError("All four cohort roots are required")
    return roots


def rows_by_id(rows, expected):
    if len(rows) != len(expected) or {row.get("run_id") for row in rows} != set(expected):
        raise ValueError("Every frozen run must occur exactly once")
    return {row["run_id"]: row for row in rows}


def load_cohorts(comparison_root, roots):
    if set(roots) != set(COHORTS) or any(root.is_symlink() for root in [comparison_root, *roots.values()]):
        raise ValueError("Four real cohort directories are required")
    comparison, record = read_json(comparison_root, "comparison.json")
    cohorts = comparison.get("cohorts", {})
    if comparison.get("format") != 2 or set(cohorts) != set(COHORTS):
        raise ValueError("A complete four-cohort comparison is required")
    rows = rows_by_id(comparison.get("runs", []), quality_axes())
    reports, outputs = {}, {}
    for name in COHORTS:
        filename = "verification.json" if name == "C" else "summary.json"
        binding = "verification_file" if name == "C" else "summary_file"
        report, _ = read_json(roots[name], filename, cohorts[name][binding])
        if (name == "C" and (report.get("numeric_complete") is not True or report.get("numeric_passed") is not True)) or (
                name != "C" and report.get("complete") is not True):
            raise ValueError("An incomplete cohort cannot establish visual equivalence")
        recorded = rows_by_id(report.get("runs", []), rows)
        reports[name], outputs[name] = report, {}
        for key, row in recorded.items():
            if row.get("status") != "completed" or canonical_hash(row["input"]) != canonical_hash(rows[key]["input"]):
                raise ValueError("The output belongs to a different or incomplete run")
            path = ("rollout/" if name == "C" else "") + key + "/output.png"
            expected = report["evidence_files"][path] if name == "C" else row["files"]["output.png"]
            outputs[name][key] = (path, expected)
    identity = reports["C"]["identity"]
    if identity["profile_sha256"] != canonical_hash(identity["profile"]) or cohorts["C"]["profile_sha256"] != identity["profile_sha256"]:
        raise ValueError("The comparison and numeric profile identities differ")
    return comparison, record, rows, reports["C"], outputs


def unchanged_contract(previous, candidate):
    old, new = previous["profile"], candidate["profile"]
    if old["id"] != "qmi-mobile-w4-dit-g16-blocks16-23-f32-v1" or new["id"] != "qmi-mobile-" + NATIVE_COMPLEX_CANDIDATE + "-v1":
        raise ValueError("Visual reuse is limited to the original FMA and native-complex candidate")
    old_code, new_code = old["source"]["converter_sha256"], new["source"]["converter_sha256"]
    allowed = {"metal_rotary.py", "mobile_convert.py", "mobile_models.py", "mobile_profile.py"}
    if set(old_code) != set(new_code) or {key for key in old_code if old_code[key] != new_code[key]} != allowed:
        raise ValueError("Converter changes exceed the reviewed rotary selection")

    def normalized(profile):
        result = copy.deepcopy(profile)
        result.pop("id")
        result["source"].pop("converter_sha256")
        return result

    if normalized(previous["profile"]) != normalized(candidate["profile"]):
        raise ValueError("Visual reuse requires unchanged source, workload and precision contracts")
    for key in ("fixtures_file", "policy_sha256", "versions"):
        if previous[key] != candidate[key]:
            raise ValueError("Visual reuse requires unchanged numerical inputs and dependencies: " + key)
    for key in ("source_files", "effective_compression", "sigmas", "prefix_kv_cache"):
        if previous["source_identity"][key] != candidate["source_identity"][key]:
            raise ValueError("Visual reuse requires unchanged source and effective weights: " + key)
    return [{"path": "/id", "previous": old["id"], "candidate": new["id"]},
            *[{"path": "/source/converter_sha256/" + key, "previous": old_code[key], "candidate": new_code[key]} for key in sorted(allowed)]]


def gallery_content(root, comparison, cohorts, outputs):
    if root.is_symlink():
        raise ValueError("The preserved gallery must be a real directory")
    receipt, receipt_file = read_json(root, "review-receipt.json")
    if receipt.get("format") != 1 or receipt.get("complete") is not True or receipt["comparison_file"] != comparison:
        raise ValueError("The preserved gallery belongs to another comparison")
    html, _ = checked_bytes(root, "index.html", receipt["html"])
    template = (Path(__file__).parents[2] / "recipes/qwen_image21_mobile/quality-review.html").read_bytes()
    if bytes_record(template) != receipt["template_file"]:
        raise ValueError("The gallery must retain the frozen review template")
    _, separator, rest = html.partition(b'<script id="payload" type="application/json">')
    encoded, closing, _ = rest.partition(b"</script>")
    if not separator or not closing or template.replace(b"__REVIEW_PAYLOAD__", encoded) != html:
        raise ValueError("The rendered gallery differs outside its recorded payload")
    payload = json.loads(encoded)
    if set(payload) != {"format", "comparison_sha256", "cases", "ratings", "review"} or payload["comparison_sha256"] != comparison["sha256"]:
        raise ValueError("The displayed gallery belongs to another comparison")
    validate_ratings(payload["ratings"], cohorts)
    expected = {}
    cases = rows_by_id(payload["cases"], outputs["C"])
    for key, row in cases.items():
        if set(row["images"]) != set(COHORTS):
            raise ValueError("The displayed case must show all four cohorts")
        for name in COHORTS:
            destination = "images/" + name + "/" + key + ".png"
            if row["images"][name] != destination:
                raise ValueError("The gallery changes the cohort and run mapping")
            expected[destination] = outputs[name][key][1]
        if row.get("reference"):
            expected[row["reference"]] = receipt["images"][row["reference"]]
    if receipt["images"] != expected:
        raise ValueError("The gallery image inventory differs from the numerical outputs")
    for name, record in expected.items():
        checked_bytes(root, name, record)
    content = {key: payload[key] for key in ("format", "cases", "review")}
    content["ratings"] = {name: payload["ratings"]["cohorts"][name]["ratings"] for name in COHORTS}
    return content, {"receipt_file": receipt_file, "display_guide_file": receipt["display_guide_file"],
                     "template_file": receipt["template_file"], "images": receipt["images"]}


def prove_equivalence(previous_comparison, previous_roots, candidate_comparison, candidate_roots):
    review, review_file = read_json(OWNER_REVIEW_PATH.parent, OWNER_REVIEW_PATH.name)
    amendment = review.get("acceptance_amendment", {})
    if (review.get("format") != 1 or review.get("id") != OWNER_REVIEW
            or review.get("decision") != "accepted_for_current_local_research_candidate_visual_quality"
            or amendment.get("criterion_id") != "QMI-OWNER-VISUAL-01" or amendment.get("status") != "met"):
        raise ValueError("The original owner visual acceptance is required")
    previous, previous_file, previous_rows, previous_numeric, previous_outputs = load_cohorts(previous_comparison, previous_roots)
    if (previous_file["sha256"] != review["comparison_sha256"]
            or previous["cohorts"]["C"]["verification_file"]["sha256"] != review["numeric_verification_sha256"]
            or previous["cohorts"]["C"]["profile_sha256"] != review["candidate_profile_sha256"]):
        raise ValueError("The previous gallery is not the one accepted by the owner")
    candidate, candidate_file, candidate_rows, candidate_numeric, candidate_outputs = load_cohorts(candidate_comparison, candidate_roots)
    contract_changes = unchanged_contract(previous_numeric["identity"], candidate_numeric["identity"])
    rows = []
    for key, previous_row in previous_rows.items():
        input_hash = canonical_hash(previous_row["input"])
        if input_hash != canonical_hash(candidate_rows[key]["input"]):
            raise ValueError("The displayed outputs have different input identities: " + key)
        images = {}
        for name in COHORTS:
            old_path, old_record = previous_outputs[name][key]
            new_path, new_record = candidate_outputs[name][key]
            old_data, _ = checked_bytes(previous_roots[name], old_path, old_record)
            new_data, _ = checked_bytes(candidate_roots[name], new_path, new_record)
            if old_data != new_data or old_record != new_record:
                raise ValueError("The candidate changes a reviewed image: " + name + "/" + key)
            images[name] = old_record
        rows.append({"run_id": key, "input_sha256": input_hash, "images": images})
    return {"format": 1, "original_owner_review": {"id": OWNER_REVIEW, "file": review_file},
            "previous": {"comparison_file": previous_file, "cohorts": previous["cohorts"], **previous["cohorts"]["C"]},
            "candidate": {"comparison_file": candidate_file, "cohorts": candidate["cohorts"], **candidate["cohorts"]["C"]},
            "allowed_profile_changes": contract_changes,
            "equivalence_code": {name: file_record(Path(__file__).with_name(name)) for name in EQUIVALENCE_FILES},
            "same_reviewed_images": True, "run_count": len(rows), "image_count": len(rows) * len(COHORTS), "runs": rows,
            "scope": "Byte-identical PNGs and unchanged inputs, source, precision and effective compression. Evidence for reusing the original visual feedback only; not numerical auditing, benchmark scoring, new human approval or device acceptance."}


def prove_presentation(result, previous_gallery, candidate_gallery):
    outputs = {name: {row["run_id"]: (None, row["images"][name]) for row in result["runs"]} for name in COHORTS}
    previous_content, previous = gallery_content(previous_gallery, result["previous"]["comparison_file"], result["previous"]["cohorts"], outputs)
    if previous["receipt_file"] != REVIEWED_GALLERY_RECEIPT:
        raise ValueError("The previous presentation is not the preserved owner-reviewed gallery")
    candidate_content, candidate = gallery_content(candidate_gallery, result["candidate"]["comparison_file"], result["candidate"]["cohorts"], outputs)
    if (previous_content != candidate_content or previous["images"] != candidate["images"]
            or previous["display_guide_file"] != candidate["display_guide_file"] or previous["template_file"] != candidate["template_file"]):
        raise ValueError("The gallery changes prompts, references, ordering or review guidance")
    return {**result, "same_reviewed_presentation": True,
            "presentation": {"previous": previous, "candidate": candidate, "content_sha256": canonical_hash(previous_content)}}


def main():
    parser = argparse.ArgumentParser()
    for name in ("previous-comparison", "candidate-comparison", "previous-gallery", "candidate-gallery", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("previous-root", "candidate-root"):
        parser.add_argument("--" + name, action="append", default=[])
    args = parser.parse_args()
    result = prove_equivalence(args.previous_comparison, cohort_roots(args.previous_root),
                               args.candidate_comparison, cohort_roots(args.candidate_root))
    result = prove_presentation(result, args.previous_gallery, args.candidate_gallery)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
