import json
from pathlib import Path

from .evaluation import canonical_hash
from .fixtures import file_record


OWNER_REVIEW = "qmi-owner-visual-v1"
OWNER_REVIEW_PATH = Path(__file__).parents[4] / "docs/research/qwen-image21-owner-quality-20261028.json"


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
