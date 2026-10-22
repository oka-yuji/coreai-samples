import json
from pathlib import Path

import numpy as np

from .contracts import tensor_metrics
from .evaluation import canonical_hash, validate_input_pair
from .fixtures import file_record, validate_fixture_suite
from .mobile_artifacts import inventory
from .numeric_policy import numeric_policy


POLICY_SHA256 = "6e3c72b9321b6cdc105f8ddf38aaea9c151ab912e2bb9c36dd57654b31d4c793"


def mobile_fixtures(root, manifest):
    policy = numeric_policy()
    if canonical_hash(policy) != POLICY_SHA256:
        raise ValueError("The predeclared mobile numerical policy changed")
    if file_record(Path(root) / "fixtures.json")["sha256"] != policy["fixture_manifest_raw_sha256"]:
        raise ValueError("The fixture manifest differs from the one bound to the numerical policy")
    return validate_fixture_suite(root, manifest)


def trace_files(operation):
    names = {"encoder-inputs.npz", "language-inputs.npz", "language-reference.npy", "vae-decode-input.npy", "rgba.npy"}
    for index in range(8):
        names.update({f"transformer-{index}-inputs.npz", f"joint-noise-{index}.npy", f"step-latent-input-{index}.npy",
                      f"step-noise-input-{index}.npy", f"step-raw-output-{index}.npy", f"latent-{index}.npy"})
    if operation == "edit":
        names.update({"vision-inputs.npz", "vae-encode-input.npy", "vae-encode-output.npy",
                      *["vision-" + name + ".npy" for name in ("merged", "deepstack_0", "deepstack_1", "deepstack_2")]})
    elif operation != "generate":
        raise ValueError("Unknown source capture operation")
    return names


def validate_trace(directory, operation):
    directory = Path(directory)
    trace = json.loads((directory / "trace.json").read_text())
    editing = int(operation == "edit")
    expected = {"vision": editing, "text": 1, "denoiser": 8, "vae_encoder": editing, "vae_decoder": 1,
                "scheduler": 8, "callback": 8, "encoder_inputs": 1, "language_inputs": 1, "vision_inputs": editing}
    expected_layouts = {f"transformer-{index}-layout.json" for index in range(8)}
    not_applicable = [] if editing else [{"component": name, "reason": "generation has no reference image"}
                                       for name in ("vision", "vae_encoder")]
    if (trace.get("complete") is not True or trace.get("counts") != expected
            or set(trace.get("tensors", {})) != trace_files(operation)
            or set(trace.get("layouts", {})) != expected_layouts
            or trace.get("not_applicable") != not_applicable
            or [row.get("index") for row in trace.get("steps", [])] != list(range(8))):
        raise ValueError("The source capture is incomplete or changed its comparison coverage")
    files = inventory(directory)
    expected_files = trace_files(operation) | expected_layouts | {"trace.json", "output.png", "result.json"}
    if set(files) != expected_files:
        raise ValueError("The source capture file set differs from its declared coverage")
    result = json.loads((directory / "result.json").read_text())
    if result.get("files") != {name: value for name, value in files.items() if name != "result.json"}:
        raise ValueError("Source capture files changed after the run completed")
    for name, record in {**trace["tensors"], **trace["layouts"]}.items():
        if {key: record.get(key) for key in ("bytes", "sha256")} != files[name]:
            raise ValueError("The source trace points to different tensor data")
        if name.endswith(".npy") and record.get("nonfinite_elements") != 0:
            raise ValueError("Non-finite source output cannot be a numerical reference")
    return result, trace


def validate_run_result(result, expected):
    if (result.get("status") != "completed" or result.get("run_id") != expected["run_id"]
            or canonical_hash(result.get("input")) != canonical_hash(expected)):
        raise ValueError("The source result belongs to another run, input or completion state")
    validate_input_pair(expected, result["input"])


def source_suite(root, profile, fixtures, manifest, selected, component_records, source_files, versions):
    root = Path(root)
    identity = json.loads((root / "identity.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (identity.get("format") != 2 or identity.get("load_status") != "completed"
            or canonical_hash(identity.get("profile")) != canonical_hash(profile)
            or identity.get("profile_sha256") != canonical_hash(profile)
            or identity.get("fixtures_file") != file_record(Path(fixtures) / "fixtures.json")
            or identity.get("exported_components") != component_records
            or identity.get("source_files") != source_files or identity.get("versions") != versions
            or identity.get("route") != "original-diffusers-with-declared-precision-boundaries"
            or identity.get("sigmas") != profile["workload"]["sigmas"] or identity.get("prefix_kv_cache") is not False
            or canonical_hash(summary.get("identity")) != canonical_hash(identity)):
        raise ValueError("Source reference belongs to a different profile, input suite or exported weight identity")
    code = ("mobile_reference.py", "mobile_source.py", "mobile_capture.py", "mobile_evidence.py", "fixtures.py", "numeric_policy.py")
    if identity.get("reference_code") != {name: file_record(Path(__file__).with_name(name)) for name in code}:
        raise ValueError("Source reference belongs to different reference code")
    rows = summary.get("runs", [])
    expected = {run["run_id"]: run for run in manifest["runs"]}
    if len(rows) != len(expected) or {row.get("run_id") for row in rows} != set(expected):
        raise ValueError("Source outcomes must preserve the complete frozen run set")
    records = {row["run_id"]: row for row in rows}
    for key in selected:
        if key not in expected or records[key].get("status") != "completed":
            raise ValueError("The selected source run is missing or did not complete: " + key)
        validate_run_result(records[key], expected[key])
        result, _ = validate_trace(root / key, expected[key]["operation"])
        if result != records[key]:
            raise ValueError("Source outcome differs from its saved completed result")
    repeats = summary.get("repeatability", [])
    if len(repeats) != 2 or {row.get("run_id") for row in repeats} != {"g01-20261018", "e01-20261018"}:
        raise ValueError("The required source repeat identities changed")
    for row in repeats:
        key = row["run_id"]
        if key not in selected:
            continue
        if row.get("status") != "completed":
            raise ValueError("A selected source repeat did not complete")
        repeated, _ = validate_trace(root / (key + "-repeat"), expected[key]["operation"])
        validate_run_result(repeated, expected[key])
        names = ["rgba.npy", *[f"latent-{index}.npy" for index in range(8)]]
        comparisons = {name: tensor_metrics(np.load(root / key / name, allow_pickle=False),
                                           np.load(root / (key + "-repeat") / name, allow_pickle=False)) for name in names}
        if row.get("comparisons") != comparisons or not all(value["passed"] for value in comparisons.values()):
            raise ValueError("The source repeat does not reproduce its saved comparison")
    return identity, records
