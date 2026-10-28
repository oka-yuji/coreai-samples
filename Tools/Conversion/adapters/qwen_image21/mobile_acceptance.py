import json
from pathlib import Path

from .evaluation import canonical_hash
from .fixtures import file_record


OWNER_REVIEW = "qmi-owner-visual-v1"
OWNER_REVIEW_PATH = Path(__file__).parents[4] / "docs/research/qwen-image21-owner-quality-20261028.json"
OWNER_IDENTICAL_REVIEW = "qmi-owner-identical-gallery-v1"
OWNER_IDENTICAL_PATH = Path(__file__).parents[4] / "docs/research/qwen-image21-native-complex-visual-equivalence-20261029.json"
OWNER_IDENTICAL_RECORD = None


def identical_acceptance(comparison, profile_hash, numeric_file, acceptance):
    from .fixtures import quality_axes
    from .mobile_gallery import COHORTS, read_json
    from .mobile_visual_equivalence import EQUIVALENCE_FILES, REVIEWED_GALLERY_RECEIPT

    review, review_file = read_json(OWNER_REVIEW_PATH.parent, OWNER_REVIEW_PATH.name)
    if OWNER_IDENTICAL_RECORD is None:
        raise ValueError("Reviewed visual-equivalence evidence is not pinned")
    proof, proof_file = read_json(OWNER_IDENTICAL_PATH.parent, OWNER_IDENTICAL_PATH.name, OWNER_IDENTICAL_RECORD)
    amendment = review.get("acceptance_amendment", {})
    previous, candidate = proof.get("previous", {}), proof.get("candidate", {})
    if (review.get("format") != 1 or review.get("id") != OWNER_REVIEW
            or review.get("decision") != "accepted_for_current_local_research_candidate_visual_quality"
            or amendment.get("criterion_id") != "QMI-OWNER-VISUAL-01" or amendment.get("status") != "met"
            or proof.get("format") != 1 or proof.get("same_reviewed_images") is not True
            or proof.get("same_reviewed_presentation") is not True
            or proof.get("original_owner_review") != {"id": OWNER_REVIEW, "file": review_file}
            or previous.get("comparison_file", {}).get("sha256") != review.get("comparison_sha256")
            or previous.get("verification_file", {}).get("sha256") != review.get("numeric_verification_sha256")
            or previous.get("profile_sha256") != review.get("candidate_profile_sha256")
            or candidate.get("profile_sha256") != profile_hash or candidate.get("verification_file") != numeric_file
            or candidate.get("comparison_file") != file_record(comparison)
            or proof.get("presentation", {}).get("previous", {}).get("receipt_file") != REVIEWED_GALLERY_RECEIPT
            or proof.get("equivalence_code") != {name: file_record(Path(__file__).with_name(name)) for name in EQUIVALENCE_FILES}):
        raise ValueError("Visual equivalence does not bind this exact candidate to the original owner review")
    rows = proof.get("runs", [])
    if (proof.get("run_count") != 60 or proof.get("image_count") != 240 or len(rows) != 60
            or {row.get("run_id") for row in rows} != set(quality_axes())
            or any(set(row.get("images", {})) != set(COHORTS) for row in rows)):
        raise ValueError("Visual equivalence must retain every reviewed image")
    return {**acceptance, "method": OWNER_IDENTICAL_REVIEW, "accepted": True,
            "criterion_id": amendment["criterion_id"], "review_file": review_file, "evidence": "evidence/owner-review.json",
            "equivalence_file": proof_file, "equivalence_evidence": "evidence/visual-equivalence.json",
            "scope": "Original qualitative visual acceptance reused for the exact byte-identical gallery and presentation; no new human review, benchmark score or device acceptance"}


def quality_acceptance(comparison, profile, verification, owner_review=None):
    result = json.loads(comparison.read_text())
    numeric = json.loads(verification.read_text())
    numeric_file = file_record(verification)
    profile_hash = canonical_hash(profile)
    if (numeric.get("numeric_complete") is not True or numeric.get("numeric_passed") is not True
            or result.get("cohorts", {}).get("C") != {
                "verification_file": numeric_file, "profile_sha256": profile_hash}):
        raise ValueError("Quality acceptance requires the complete matching numerical verification")
    benchmark_passed = result.get("quality_passed") is True
    if benchmark_passed != (result.get("quality", {}).get("passed") is True):
        raise ValueError("The benchmark quality result is inconsistent")
    acceptance = {"method": "case-axis-benchmark", "accepted": benchmark_passed,
                  "benchmark_passed": benchmark_passed, "device_verification": "pending"}
    if owner_review is None:
        return acceptance
    if owner_review == OWNER_IDENTICAL_REVIEW:
        return identical_acceptance(comparison, profile_hash, numeric_file, acceptance)
    if owner_review != OWNER_REVIEW:
        raise ValueError("Unknown owner review; a new candidate requires its own approval")
    review = json.loads(OWNER_REVIEW_PATH.read_text())
    amendment = review.get("acceptance_amendment", {})
    if (review.get("format") != 1 or review.get("id") != OWNER_REVIEW
            or review.get("decision") != "accepted_for_current_local_research_candidate_visual_quality"
            or amendment.get("criterion_id") != "QMI-OWNER-VISUAL-01" or amendment.get("status") != "met"
            or review.get("candidate_profile_sha256") != profile_hash
            or review.get("numeric_verification_sha256") != numeric_file["sha256"]
            or review.get("comparison_sha256") != file_record(comparison)["sha256"]):
        raise ValueError("Owner review does not accept this exact profile and output gallery")
    acceptance.update(method=OWNER_REVIEW, accepted=True,
                      criterion_id=amendment["criterion_id"], review_file=file_record(OWNER_REVIEW_PATH),
                      evidence="evidence/owner-review.json",
                      scope="Qualitative visual acceptance of this bound local research candidate; not a benchmark score or device acceptance")
    return acceptance
