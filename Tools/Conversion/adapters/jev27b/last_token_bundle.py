import copy
import shutil
from pathlib import Path

from ...storage import read_json, write_json
from .decision import DECISION, LAST_OUTPUT, OUTPUT, listing, sha256, validate, write_sums

NOTE = "The main function now returns only the final token's hidden state as last_token_hidden_state [1, 1, H]."


def source_identity(source, baseline_asset, revision):
    source = Path(source)
    decision = read_json(source / DECISION)
    errors = validate(decision, source)
    if errors:
        raise ValueError("The source bundle failed validation: " + "; ".join(errors))
    if decision["backbone"]["output"] != OUTPUT or decision["backbone"]["hidden"]["dtype"] != "float16" \
            or decision["source"].get("merged_revision") != revision:
        raise ValueError("The source bundle must be the matching full-sequence float16 conversion")
    asset = source / decision["backbone"]["asset"] / "main.mlirb"
    digest = sha256(asset)
    if digest != sha256(Path(baseline_asset) / "main.mlirb"):
        raise ValueError("The source asset differs from the quantized-hash baseline")
    metadata = read_json(source / "metadata.json")
    return {"decision_sha256": sha256(source / DECISION), "asset_sha256": digest,
            "repository": metadata.get("distribution", {}).get("repository"),
            "calibration": copy.deepcopy(decision["temperatures"])}


def finish(destination, source, identity, asset_name, hidden_size):
    destination, source = Path(destination), Path(source)
    decision = read_json(source / DECISION)
    if sha256(source / DECISION) != identity["decision_sha256"] or validate(decision, source):
        raise ValueError("The source bundle changed during export")
    if decision["backbone"]["hidden"]["size"] != hidden_size:
        raise ValueError("The exported width differs from the source bundle")
    generated = read_json(destination / DECISION)
    preserved = ("head", "template", "slots", "verbalizers", "verbalizer_ids", "limits")
    if any(generated[key] != decision[key] for key in preserved):
        raise ValueError("The exported decision contract differs from the source bundle")
    copied = {}
    asset_prefix = decision["backbone"]["asset"] + "/"
    for entry in decision["files"]:
        path = entry["path"]
        if path.startswith(asset_prefix) or path in ("metadata.json", "NOTICE.md"):
            continue
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / path, target)
        copied[path] = sha256(target)
        if copied[path] != entry["sha256"]:
            raise ValueError(f"A preserved file changed during the copy: {path}")
    metadata_path = destination / "metadata.json"
    metadata = read_json(metadata_path)
    if metadata.get("assets") != {"main": asset_name}:
        raise ValueError("The exported metadata points at a different asset")
    source_metadata = read_json(source / "metadata.json")
    metadata["modifications"] = [*source_metadata.get("modifications", []), NOTE]
    metadata["distribution"] = {
        "status": "unpublished-candidate", "format": decision["format"], "notice": "NOTICE.md",
        "derived_from": {key: identity[key] for key in ("repository", "decision_sha256", "asset_sha256")},
        "calibration": {"status": "reused-without-refitting", **identity["calibration"]},
    }
    write_json(metadata_path, metadata)
    notice = (source / "NOTICE.md").read_text(encoding="utf-8")
    (destination / "NOTICE.md").write_text(notice.rstrip() + "\n\nAdditional change in this candidate:\n\n- " + NOTE + "\n",
                                          encoding="utf-8")
    decision["backbone"]["asset"] = asset_name
    decision["backbone"]["output"] = LAST_OUTPUT
    decision["files"] = listing(destination)
    write_json(destination / DECISION, decision)
    sums = write_sums(destination)
    errors = validate(read_json(destination / DECISION), destination)
    if errors:
        raise ValueError("The candidate bundle failed validation: " + "; ".join(errors))
    return {"source": identity, "preserved_file_sha256": copied,
            "head_sha256": copied[decision["head"]["file"]],
            "tokenizer_sha256": {path.removeprefix("tokenizer/"): digest for path, digest in copied.items()
                                 if path.startswith("tokenizer/")},
            "license_sha256": copied["LICENSE"], "license_source": "source-bundle",
            "files": len(sums), "decision_errors": errors}
