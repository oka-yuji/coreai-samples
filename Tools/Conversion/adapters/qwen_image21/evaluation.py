import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


MODEL = "Qwen/Qwen-Image-2.1-Turbo"
REVISION = "fc13ba1986fcb905854dc8de2f52eee395f49b5b"
IMPLEMENTATION = "b800f828f3951c4d206e154411bcbfa693cd322e"
SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
COMPONENTS = {"vision", "text", "denoiser", "vae_encoder", "vae_decoder"}
FLOATS = {"float32", "float16", "bfloat16"}
SCALARS = FLOATS | {"int32", "bool"}
CATEGORIES = {"weights", "activations", "cache", "workspace", "images", "application"}
CONDITIONS = {"profile_sha256", "workload_sha256", "device", "os_build", "route", "operation",
              "cache_state", "diagnostics", "power", "thermal"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid {name}")
    return value


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid {name}")
    return value


def hexadecimal(value, length, name):
    require(isinstance(value, str) and len(value) == length
            and all(character in "0123456789abcdef" for character in value), f"Invalid {name}")
    return value


def relative(value):
    require(isinstance(value, str) and value and not value.startswith("/") and "\\" not in value,
            "Expected a relative POSIX path")
    require(all(part not in ("", ".", "..") for part in value.split("/")), "Unsafe path component")
    return PurePosixPath(value)


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def raw_manifest_hash(data):
    require(isinstance(data, bytes), "Manifest identity requires its saved raw bytes")
    return hashlib.sha256(data).hexdigest()


def validate_tensor(tensor):
    require(isinstance(tensor, dict), "Expected a tensor descriptor")
    require(isinstance(tensor.get("name"), str) and tensor["name"], "Missing tensor name")
    require(tensor.get("dtype") in SCALARS, "Unsupported tensor dtype")
    require(tensor.get("layout") == "row-major", "Tensor boundary layout must be explicit")
    shape = tensor.get("shape")
    require(isinstance(shape, list) and shape, "Missing tensor shape")
    count = 1
    for dimension in shape:
        count *= integer(dimension, "tensor dimension", 1)
        require(count <= 2**63 - 1, "Tensor element count overflows")
    return count


def validate_profile(profile):
    from .profile import boundaries
    require(isinstance(profile, dict) and profile.get("format") == 2, "Expected profile format 2")
    require(isinstance(profile.get("id"), str) and profile["id"], "Missing profile id")
    source = profile.get("source", {})
    require(isinstance(source, dict), "Invalid source metadata")
    require(source.get("model") == MODEL and source.get("revision") == REVISION, "Unexpected source model")
    require(source.get("implementation_revision") == IMPLEMENTATION, "Unexpected implementation revision")
    hexadecimal(source.get("dependency_lock_sha256"), 64, "dependency lock hash")
    hashes = source.get("converter_sha256")
    require(isinstance(hashes, dict) and hashes, "Missing converter identities")
    for name, digest in hashes.items():
        relative(name)
        hexadecimal(digest, 64, "converter hash")
    workload = profile.get("workload", {})
    require(isinstance(workload, dict), "Invalid workload metadata")
    require(workload.get("resolution") == [512, 512] and workload.get("reference_resolution") == [512, 512],
            "The initial profile requires 512-square output and references")
    for name, expected in (("batch", 1), ("maximum_references", 1), ("processed_text_capacity", 512)):
        require(type(workload.get(name)) is int and workload[name] == expected, f"Unsupported {name}")
    require(type(workload.get("joint_capacity")) is int and workload["joint_capacity"] in (1536, 2304),
            "Unsupported joint capacity")
    require(workload.get("operation") in ("generate", "generate-and-edit"), "Missing workload operation")
    require(workload["joint_capacity"] == 2304 or workload["operation"] == "generate",
            "The smaller joint capacity is generation-only")
    require(type(workload.get("cfg")) in (int, float) and workload["cfg"] == 1, "Initial CFG must be one")
    require(workload.get("sigmas") == SIGMAS
            and all(type(value) in (float, int) for value in workload["sigmas"]),
            "The saved Turbo schedule must be retained")
    require(workload.get("scheduler_compute_dtype") == "float32", "Euler accumulation must use float32")
    require(workload.get("scheduler_output_dtype") in FLOATS, "Missing per-step cast-back dtype")
    require(workload.get("terminal_sigma") == 0.0 and workload.get("cast_back") == "every-step-to-model-output",
            "Euler must retain the source terminal sigma and per-step cast-back")
    require(workload.get("rounding") == "nearest-ties-to-even", "Unsupported rounding policy")
    require(workload.get("denoiser_fp16_clip") == [-65504, 65504]
            and workload.get("zero_centered_norm_dtype") == "float32", "Missing denoiser precision safeguards")
    require(workload.get("prefix_kv_cache") is False, "The initial profile is stateless")
    require(workload.get("noise_shape") == [1, 1024, 64] and workload.get("noise_dtype") == "float32",
            "Unexpected saved-noise contract")
    require(isinstance(workload.get("rng_version"), str) and workload["rng_version"], "Missing RNG version")
    components = profile.get("components")
    require(isinstance(components, dict) and set(components) == COMPONENTS, "All five components are required")
    expected = boundaries(workload["joint_capacity"])
    for kind, component in components.items():
        require(isinstance(component, dict), "Invalid component metadata")
        require(relative(component.get("asset")).suffix == ".aimodel", "Expected a Core AI asset")
        require(component.get("function") == "main" and component.get("state_names") == [], "Expected a single stateless function")
        require(component.get("compute_dtype") in FLOATS, "Missing component compute dtype")
        require(isinstance(component.get("weight_encoding"), dict) and component["weight_encoding"].get("kind"),
                "Missing weight encoding")
        for direction in ("inputs", "outputs"):
            tensors = component.get(direction)
            require(isinstance(tensors, list) and tensors, f"Missing {kind} {direction}")
            names = []
            for tensor in tensors:
                validate_tensor(tensor)
                names.append(tensor["name"])
            require(len(names) == len(set(names)), "Duplicate tensor name")
            physical = {item["name"]: item for item in tensors}
            require(set(physical) == {item["name"] for item in expected[kind][direction]},
                    "Tensor names differ from the fixed profile")
            for tensor in expected[kind][direction]:
                actual = physical[tensor["name"]]
                require(actual["shape"] == tensor["shape"], "Physical tensor shape differs from the profile")
                require(actual["dtype"] in FLOATS if tensor["dtype"] in FLOATS else actual["dtype"] == tensor["dtype"],
                        "A physical index or mask has the wrong dtype")
    return canonical_hash(profile)


def validate_input_pair(reference, candidate):
    fields = ("fixture_id", "source_revision", "prompt_sha256", "reference_sha256", "noise_sha256",
              "processed_tokens_sha256", "layout_sha256", "sigmas", "operation")
    require(isinstance(reference, dict) and isinstance(candidate, dict), "Expected input records")
    require(all(name in reference and name in candidate for name in fields), "An input identity is missing")
    require(all(reference[name] == candidate[name] for name in fields), "The compared inputs differ")
    require(reference["source_revision"] == REVISION and reference["sigmas"] == SIGMAS, "Unexpected source or schedule")
    for name in ("prompt_sha256", "noise_sha256", "processed_tokens_sha256", "layout_sha256"):
        hexadecimal(reference[name], 64, name)
    if reference["operation"] == "edit":
        hexadecimal(reference["reference_sha256"], 64, "reference image hash")
    else:
        require(reference["operation"] == "generate" and reference["reference_sha256"] is None,
                "Generation must not have a reference image")


def validate_inventory(root, inventory):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Expected a real fixture directory")
    require(isinstance(inventory, dict) and inventory, "Missing fixture inventory")
    found = {}
    for name, record in inventory.items():
        parts = relative(name).parts
        path = root
        for part in parts:
            path = path / part
            require(not path.is_symlink(), "Fixture symbolic links are not accepted")
        require(path.is_file(), f"Missing fixture: {name}")
        expected_bytes = integer(record.get("bytes"), "file size")
        expected_hash = hexadecimal(record.get("sha256"), 64, "file hash")
        require(path.stat().st_size == expected_bytes, f"Fixture size differs: {name}")
        with path.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        require(observed == expected_hash, f"Fixture content differs: {name}")
        found[name] = observed
    return found


def live_memory_peak(allocations, unknown_categories=()):
    unknown = set(unknown_categories)
    require(unknown <= CATEGORIES, "Unknown memory category")
    events = defaultdict(list)
    definitions = {}
    for record in allocations:
        identity = record.get("storage_id")
        require(isinstance(identity, str) and identity, "Missing backing-storage identity")
        size = integer(record.get("backing_bytes"), "backing-storage bytes")
        begin = integer(record.get("begin_ns"), "allocation start")
        end = integer(record.get("end_ns"), "allocation end")
        require(end > begin, "Allocation lifetimes must be nonempty half-open intervals")
        category = record.get("category")
        require(category in CATEGORIES, "Unknown allocation category")
        definition = (size, category)
        require(identity not in definitions or definitions[identity] == definition,
                "Aliases must describe the same backing storage")
        definitions[identity] = definition
        events[begin].append((identity, 1))
        events[end].append((identity, -1))
    references = Counter()
    maximum = 0
    peak_time = None
    peak_categories = {key: 0 for key in sorted(CATEGORIES)}
    timeline = []
    for timestamp in sorted(events):
        for identity, change in events[timestamp]:
            references[identity] += change
            require(references[identity] >= 0, "Unbalanced lifetime")
        categories = {key: 0 for key in sorted(CATEGORIES)}
        for identity, count in references.items():
            if count:
                size, category = definitions[identity]
                categories[category] += size
        total = sum(categories.values())
        timeline.append({"time_ns": timestamp, "live_bytes": total, "categories": categories})
        if total > maximum:
            maximum, peak_time, peak_categories = total, timestamp, categories
    return {"planning_peak_bytes": maximum if not unknown else None, "known_live_peak_bytes": maximum,
            "unmeasured_categories": sorted(unknown), "time_ns": peak_time, "categories": peak_categories,
            "timeline": timeline, "scope": "declared backing-storage lifetimes; not measured device RAM"}


def measurement(value, status, unit, scope):
    require(status in ("ok", "unavailable", "failed", "not_measured"), "Unknown observation status")
    require(isinstance(unit, str) and unit and isinstance(scope, str) and scope, "Missing measurement boundary")
    if status == "ok":
        require(finite(value, "measurement") >= 0, "Negative measurement")
    else:
        require(value is None, "Unavailable measurements must not become zero")
    return {"value": value, "status": status, "unit": unit, "scope": scope}


def aggregate_runs(runs, minimum_samples=20):
    integer(minimum_samples, "minimum samples", 1)
    groups = {}
    seen = set()
    for run in runs:
        identity = run.get("run_id")
        require(isinstance(identity, str) and identity and identity not in seen, "Missing or duplicate run id")
        seen.add(identity)
        conditions = run.get("conditions")
        require(isinstance(conditions, dict) and set(conditions) == CONDITIONS, "Incomplete measurement conditions")
        for key in ("profile_sha256", "workload_sha256"):
            hexadecimal(conditions[key], 64, key)
        require(conditions["cache_state"] in ("uncached-cold", "cached-cold", "warm", "continuous"), "Unknown cache state")
        require(conditions["diagnostics"] in ("off", "summary", "trace"), "Unknown diagnostic level")
        require(conditions["operation"] in ("generate", "edit"), "Unknown operation")
        require(all(isinstance(value, str) and value for value in conditions.values()), "Empty condition")
        status = run.get("status")
        require(status in ("completed", "failed", "cancelled", "not_run"), "Unknown run outcome")
        total = run.get("total_seconds")
        if status == "completed":
            require(finite(total, "total time") >= 0, "Negative total time")
        else:
            require(total is None, "Incomplete runs must not appear as completed latency samples")
        key = canonical_hash(conditions)
        group = groups.setdefault(key, {"conditions": conditions, "outcomes": Counter(), "values": []})
        group["outcomes"][status] += 1
        if total is not None:
            group["values"].append(float(total))
    result = []
    for key, group in sorted(groups.items()):
        values = sorted(group["values"])
        def percentile(fraction):
            return values[max(0, math.ceil(len(values) * fraction) - 1)] if values else None
        result.append({"group_id": key, "conditions": group["conditions"], "outcomes": dict(group["outcomes"]),
                       "completed_samples": len(values), "minimum_samples": minimum_samples,
                       "sample_requirement_met": len(values) >= minimum_samples,
                       "percentile_method": "nearest-rank", "p50_seconds": percentile(0.5),
                       "p95_seconds": percentile(0.95), "maximum_seconds": values[-1] if values else None})
    return result


def quality_summary(reference, candidate, required_axes):
    require(isinstance(required_axes, dict) and len(required_axes) == 60, "All sixty frozen outputs are required")
    require(set(reference) == set(candidate) == set(required_axes), "Quality rows cannot be added or omitted")
    summaries = {}
    new_failures = []
    incomplete = []
    for identity, axes in required_axes.items():
        require(set(reference[identity]) == set(candidate[identity]) == set(axes), "Quality axes changed")
        reference_passed = True
        candidate_failed = False
        for axis in axes:
            left, right = reference[identity][axis], candidate[identity][axis]
            require(all(value is None or (type(value) is int and value in (0, 1, 2)) for value in (left, right)),
                    "Quality scores must be unreviewed, failed, acceptable, or good")
            row = summaries.setdefault(axis, {"required": 0, "reference_successes": 0, "candidate_successes": 0,
                                              "unreviewed": 0})
            row["required"] += 1
            row["reference_successes"] += int(left is not None and left >= 1)
            row["candidate_successes"] += int(right is not None and right >= 1)
            row["unreviewed"] += int(left is None or right is None)
            reference_passed &= left is not None and left >= 1
            candidate_failed |= right == 0
            if left is None or right is None:
                incomplete.append(identity + ":" + axis)
        if reference_passed and candidate_failed:
            new_failures.append(identity)
    for row in summaries.values():
        row["reference_rate"] = row["reference_successes"] / row["required"]
        row["candidate_rate"] = row["candidate_successes"] / row["required"]
    return {"complete": not incomplete, "unreviewed": incomplete, "axes": summaries,
            "new_failures": new_failures,
            "reference_meets_provisional_target": not incomplete and all(row["reference_rate"] >= 0.9 for row in summaries.values()),
            "candidate_meets_provisional_target": not incomplete and len(new_failures) <= 2
            and all(row["candidate_rate"] >= 0.9 for row in summaries.values()),
            "scope": "provisional 90% per applicable axis and at most two new failures; requires human ratings"}
