import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time

from ...runner import deferred_interrupts
from ...storage import PipelineError, canonical, digest, write_json
from ..jev27b.monitor import FREE_PERCENT, INTERVAL_S, STREAK, SWAP_LIMIT_BYTES
from ..pinned import sha256
from .anchor import parse
from .compare import BLOCK_ROWS, CONTRACT
from .export import (BUNDLE, FUNCTIONS, INCOMPLETE, MARGIN_BYTES, PACKAGES as EXPORT_PACKAGES, PRECISIONS,
                     SCHEMA as EXPORT_SCHEMA, WRITES, check_asset, expected_functions, metadata, name_of, quantized_names,
                     required_bytes)
from .layers import (CONDITIONS, CONTEXT_SCHEMA, PACKAGES as REFERENCE_PACKAGES, PARTS, QUANTIZED, REFERENCE_CHUNK,
                     WEIGHT_KINDS, check_layers, check_taps, joined, rounded_scale,
                     kept_rows, tokens)
from .prepare import TOKENIZER_FILES, load_contract
from .reference import LOADED_FILES, SWAP_INCREASE_LIMIT_BYTES

SCHEMA = "gemma4-31b-native-comparison/1"
MAX_ABS_LOGIT = 2.0
NEAR_TIE = 0.01
KNOWN_ENDS, KNOWN_SHORTEST = range(97, 113), 8
TIMEOUT_S = 1800
LARGEST = 2**63 - 1
BLOCK_BYTES = 16 * 2**20
CRITERIA = {"max_abs_logit": MAX_ABS_LOGIT, "near_tie": NEAR_TIE}
KNOWN_CALL = {"ends": [KNOWN_ENDS[0], KNOWN_ENDS[-1]], "shortest": KNOWN_SHORTEST}
PREFILL_CHUNKS = (2048, 4096)
FLOAT16_BYTES = 2
LONGEST = max(length for _, _, length, _, _ in CONDITIONS)
TOP1 = ("near_tie", "outside")
FIELDS = ("name", "input", "tokens", "start", "chunk", "rows", "problems", "checks", "passed")
PATHS = (("seconds",), ("verifier",), ("output",),
         ("non_finite", "reference_non_finite", "constant_rows", "max_abs_logit", "top1_mismatches", "known"))
PATH_FIELDS = [set(FIELDS).union(*PATHS[:index]) for index in range(len(PATHS) + 1)]
DISK = ("weight_bytes", "estimated_write_bytes", "margin_bytes", "required_free_bytes", "free_before_bytes",
        "lowest_free_bytes")
SWAP = ("first_bytes", "peak_bytes", "limit_increase_bytes", "increase_bytes")
REFERENCE_TIMING = ("input_hash_s", "load_s", "total_s")
REFERENCE_MEMORY = ("max_rss_bytes", "lifetime_peak_footprint_bytes")
STEPS = ("input_hash_s", "load_s", "convert_s", "save_s", "inspect_s", "bundle_hash_s")
QUANTIZING_STEPS = ("quantize_s", "tensors_s")
MONITOR = {"interval_s": INTERVAL_S, "abort": None, "aborts": 0, "failure": None, "unreadable_samples": 0,
           "limits": {"swap_used_bytes": SWAP_LIMIT_BYTES, "free_percent": FREE_PERCENT, "streak": STREAK}}
READINGS = {"first": {"swap_used_bytes", "free_bytes"}, "peak": {"rss_bytes", "max_rss_bytes", "swap_used_bytes",
                                                                  "temporary_bytes"}, "lowest": {"free_percent", "free_bytes"}}
VERIFIER = {"record": "text", "token_count": "count", "start": "count", "chunk": "count", "tokens_sha256": "text",
            "shape": "shape", "dtype": "text", "byte_order": "text", "main_calls": "count", "main_hash": "text",
            "prefix_function": "text", "prefix_calls": "count", "prefix_chunk": "count", "cached": "flag",
            "non_finite_count": "count", "load_s": "seconds", "prefix_s": "seconds", "scoring_s": "seconds",
            "specialize_s": "seconds"}
OPTIONAL = {"prefix_function"}
PUBLIC = ("cached", "prefix_function", "prefix_calls", "main_calls", "non_finite_count", "load_s", "prefix_s", "scoring_s",
          "specialize_s")


def read_once(path):
    data = Path(path).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def pinned_model(contract, contract_path):
    return {"id": contract["model_id"], "revision": contract["revision"], "contract_sha256": sha256(contract_path),
            "verified_before_loading": {name: entry for name, entry in contract["files"].items()
                                        if name in LOADED_FILES or name.endswith(".safetensors")}}


def valid(kind, value):
    if kind == "count":
        return type(value) is int and 0 <= value <= LARGEST
    if kind in ("seconds", "number"):
        return valid("count", value) or type(value) is float and math.isfinite(value) and value >= 0
    if kind == "shape":
        return type(value) is list and all(valid("count", size) for size in value)
    return type(value) is {"flag": bool, "text": str}[kind]


def validated(record):
    kept, malformed = {}, []
    for key, kind in VERIFIER.items():
        if key in record and valid(kind, record[key]):
            kept[key] = record[key]
        elif key in record or key not in OPTIONAL:
            malformed.append(key)
    return kept, malformed


def known_rows(start, length, chunk):
    rows = []
    for first in range(start, length, chunk):
        end = min(first + chunk, length)
        if end - first >= KNOWN_SHORTEST and end in KNOWN_ENDS:
            rows.extend(range(first, end))
    return rows


def prefill_chunk():
    return PREFILL_CHUNKS[1] if os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") >= 64 << 30 else PREFILL_CHUNKS[0]


def call_lengths(position, count, limit):
    lengths, end = [], position
    while end < position + count:
        length = min(limit, position + count - end)
        while length >= KNOWN_SHORTEST and end + length in KNOWN_ENDS:
            length -= 1
        lengths.append(length)
        end += length
    return lengths


def expected_verifier(entry, config, main_hash, prefill, prefix_chunk):
    ids = tokens(entry["input"], config)[:entry["tokens"]]
    return {"record": "logits", "token_count": entry["tokens"], "start": entry["start"], "chunk": entry["chunk"],
            "tokens_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
            "shape": [entry["rows"], config["vocab_size"]], "dtype": "float16", "byte_order": "little",
            "main_calls": math.ceil(entry["rows"] / entry["chunk"]), "main_hash": main_hash,
            **prefix_record(entry["start"], prefix_chunk, prefill)}


def prefix_record(start, chunk, prefill):
    if start == 0:
        return {"prefix_calls": 0, "prefix_chunk": chunk}
    lengths = call_lengths(0, start, chunk)
    unsplit = lengths == [min(chunk, start - first) for first in range(0, start, chunk)]
    return {"prefix_function": "prefill" if prefill and unsplit else "main", "prefix_calls": len(lengths),
            "prefix_chunk": chunk}


def measure(swift, reference, indices, known):
    import numpy as np

    counts = dict.fromkeys(("non_finite", "reference_non_finite", "constant_rows", "near_tie", "outside"), 0)
    largest = known_largest = 0.0
    for start in range(0, len(swift), BLOCK_ROWS):
        x = np.asarray(swift[start:start + BLOCK_ROWS], dtype=np.float64)
        y = np.asarray(reference[indices[start:start + BLOCK_ROWS]], dtype=np.float64)
        excluded = known[start:start + BLOCK_ROWS]
        counts["non_finite"] += int(x.size - np.isfinite(x).sum())
        counts["reference_non_finite"] += int(y.size - np.isfinite(y).sum())
        counts["constant_rows"] += int((x.max(axis=-1) == x.min(axis=-1)).sum())
        difference = np.abs(x - y).max(axis=-1)
        largest = float(np.maximum(largest, difference[~excluded].max(initial=0.0)))
        known_largest = float(np.maximum(known_largest, difference[excluded].max(initial=0.0)))
        top = np.partition(y, -2, axis=-1)[:, -2:]
        near = top[:, 1] - top[:, 0] <= NEAR_TIE
        changed = (x.argmax(axis=-1) != y.argmax(axis=-1)) & ~excluded
        counts["near_tie"] += int((changed & near).sum())
        counts["outside"] += int((changed & ~near).sum())
    finite = not counts["non_finite"] and not counts["reference_non_finite"]
    result = {"rows": len(swift), "non_finite": counts["non_finite"], "reference_non_finite": counts["reference_non_finite"],
              "constant_rows": counts["constant_rows"], "max_abs_logit": largest if finite else None,
              "top1_mismatches": {key: counts[key] for key in TOP1},
              "known": {"rows": int(known.sum()), "max_abs_logit": known_largest if finite and known.any() else None}}
    return {**result, "checks": verdicts(result)}


def measure_condition(entry, vocabulary, data, arrays):
    import numpy as np

    source, length, start, chunk, rows = (entry[key] for key in ("input", "tokens", "start", "chunk", "rows"))
    index = {row: position for position, row in enumerate(kept_rows(source))}
    known = np.zeros(rows, dtype=bool)
    known[[row - start for row in known_rows(start, length, chunk)]] = True
    return measure(np.frombuffer(data, dtype="<f2").reshape(rows, vocabulary), arrays[source],
                   np.array([index[position] for position in range(start, length)]), known)


def measurement_problems(entry, vocabulary, data, arrays):
    numbers = measure_condition(entry, vocabulary, data, arrays)
    if canonical({key: entry.get(key) for key in numbers}) == canonical(numbers):
        return []
    return ["the numbers of the condition are not those that its logits file and the reference give"]


def logits_bytes(rows, vocabulary):
    return rows * vocabulary * FLOAT16_BYTES


def texts(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def condition_problems(entry, config, main_hash, prefill):
    if set(entry) not in PATH_FIELDS:
        return ["the condition holds other fields than the comparison writes on its paths"]
    reached, problems, verifier = PATH_FIELDS.index(set(entry)), entry["problems"], entry.get("verifier")
    measured, read, ran = reached == len(PATHS), reached > 2, reached > 1
    typed = {
        "the problems of the condition are not a list of texts": texts(problems),
        "the condition records other checks than its numbers give": isinstance(entry["checks"], dict),
        "the condition records no time of the verifier": reached < 1 or valid("seconds", entry["seconds"]),
        "the verifier record holds a field or a value that the verifier does not write":
            not ran or isinstance(verifier, dict)
            and all(key in VERIFIER and valid(VERIFIER[key], value) for key, value in verifier.items()),
        "the verifier record misses a field that the verifier writes":
            not read or isinstance(verifier, dict) and set(VERIFIER) - OPTIONAL <= set(verifier),
        "the condition records no size of its logits file":
            not read or isinstance(entry["output"], dict) and valid("count", entry["output"].get("bytes"))}
    found = [message for message, holds in typed.items() if not holds]
    if found:
        return found
    chunk = verifier.get("prefix_chunk") if read else None
    expected = expected_verifier(entry, config, main_hash, prefill, chunk) if chunk in PREFILL_CHUNKS else {}
    steps = [verifier[key] for key in ("load_s", "prefix_s", "scoring_s") if ran and key in verifier]
    formulas = {
        "the condition records other checks than its numbers give": measured or not entry["checks"],
        "the condition records another verdict than its checks and problems give":
            canonical(entry["passed"]) == canonical(condition_passed(entry)),
        "the condition records no problem on a path that stops before its numbers": measured or bool(problems),
        "the condition records other problems than the one of the path that stops at it":
            reached not in (0, 3) or len(problems) == 1,
        "the condition records more problems than the count of the verifier after its numbers":
            not measured or len(problems) <= 1,
        "the condition took less time than the verifier recorded for its steps": not ran or sum(steps) <= entry["seconds"],
        "the verifier record differs from the reads of the condition":
            not read or verifier.get("prefix_chunk") in PREFILL_CHUNKS
            and canonical({key: verifier.get(key) for key in expected}) == canonical(expected)
            and ("prefix_function" in verifier) == (entry["start"] > 0),
        "the verifier recorded a specialization of the asset that it found in the cache":
            not ran or verifier.get("cached") is not True or "specialize_s" not in verifier
            or verifier["specialize_s"] == 0,
        "the verifier recorded no specialization of the asset that it did not find in the cache":
            not ran or verifier.get("cached") is not False or "specialize_s" not in verifier
            or verifier["specialize_s"] > 0,
        "the verifier recorded a longer specialization than its loading":
            not ran or not all(key in verifier for key in ("specialize_s", "load_s"))
            or verifier["specialize_s"] <= verifier["load_s"],
        "the logits file does not hold the rows of the condition exactly when they were measured":
            not read or (canonical(entry["output"]["bytes"]) == canonical(logits_bytes(entry["rows"], config["vocab_size"])))
            == measured,
        "the verifier counted other non-finite values than the file exactly when the condition records a problem":
            not measured or (canonical(verifier["non_finite_count"]) == canonical(entry["non_finite"])) == (not problems)}
    return [message for message, holds in formulas.items() if not holds]


def verdicts(entry):
    finite = not entry["non_finite"] and not entry["reference_non_finite"]
    return {"finite": finite, "no_constant_rows": not entry["constant_rows"],
            "max_abs_logit": finite and entry["max_abs_logit"] < MAX_ABS_LOGIT, "top1": not entry["top1_mismatches"]["outside"]}


def condition_passed(entry):
    return not entry["problems"] and bool(entry["checks"]) and all(entry["checks"].values())


def comparison_passed(problems, conditions):
    return not problems and len(conditions) == len(CONDITIONS) and all(item["passed"] for item in conditions)


def layer_problems(record, contract, label):
    try:
        if record.get("layer_types") != check_layers(contract["config"], record.get("layers")):
            return [f"the {label} recorded layer types that differ from the contract"]
    except ValueError as error:
        return [f"the {label}: {error}"]
    return []


def load_reference(folder, contract, contract_path):
    import numpy as np

    record, hashed, _ = read_record(folder / "record.json")
    problems = [] if (record.get("schema"), record.get("implementation")) == (CONTEXT_SCHEMA, "original") else [
        "the reference is not a context run of the original implementation"]
    problems += layer_problems(record, contract, "reference")
    if record.get("model") != pinned_model(contract, contract_path):
        problems.append("the reference recorded another model or contract")
    arrays = {}
    for name in ("short", "context"):
        entry, ids, rows = (record.get("inputs") or {}).get(name) or {}, tokens(name, contract["config"]), kept_rows(name)
        if (entry.get("positions"), entry.get("tokens_sha256"), entry.get("rows")) != (len(ids), digest(ids), rows):
            problems.append(f"the reference did not keep the rows of the {name} input of the conditions")
        logits = entry.get("logits") or {}
        recorded = logits.get("bytes")
        found, content = regular_file(folder / f"{name}.npy", recorded if valid("count", recorded) else None, keep=True)
        if (found["bytes"], found["sha256"]) != (recorded, logits.get("sha256")):
            problems.append(f"the {name} logits differ from the size and SHA-256 in the reference record")
            continue
        stream = io.BytesIO(content)
        array = np.load(stream)
        if array.dtype != np.float32 or array.shape != (len(rows), contract["config"]["vocab_size"]):
            problems.append(f"the {name} logits are {array.dtype} {list(array.shape)}")
            continue
        if stream.tell() != len(content):
            problems.append(f"the {name} logits file holds {len(content) - stream.tell()} B after its array")
            continue
        arrays[name] = array
    return record, hashed, arrays, problems


def reference_record_problems(record, contract, arrays):
    import numpy as np

    config, problems = contract["config"], []
    if canonical(record.get("chunk")) != canonical(REFERENCE_CHUNK):
        problems.append("the reference was not computed in the fixed chunks")
    if canonical(record.get("embed_scale")) != canonical(rounded_scale(config["hidden_size"])):
        problems.append("the reference did not scale the embeddings by the rounded square root of the hidden size")
    inputs, timing, memory = mapping(record, "inputs"), mapping(record, "timing"), mapping(record, "memory")
    for name in ("short", "context"):
        entry, count, rows = mapping(inputs, name), len(tokens(name, config)), len(kept_rows(name))
        calls = [[start, min(start + REFERENCE_CHUNK, count)] for start in range(0, count, REFERENCE_CHUNK)]
        if canonical(entry.get("calls")) != canonical(calls):
            problems.append(f"the reference did not read the {name} input in the fixed chunks")
        logits = mapping(entry, "logits")
        if canonical({key: logits.get(key) for key in ("dtype", "shape")}) \
                != canonical({"dtype": "float32", "shape": [rows, config["vocab_size"]]}):
            problems.append(f"the reference recorded the {name} logits of another type or shape")
        array = arrays.get(name)
        if array is None or canonical(logits.get("non_finite")) != canonical(int(array.size - np.isfinite(array).sum())):
            problems.append(f"the reference recorded another count of the non-finite {name} logits than its file holds")
        if not valid("seconds", entry.get("seconds")):
            problems.append(f"the reference recorded no time of the {name} input")
    spent = [timing.get("input_hash_s"), timing.get("load_s"),
             *(mapping(inputs, name).get("seconds") for name in ("short", "context"))]
    formulas = {
        "the reference holds other inputs than those of the conditions":
            set(inputs) == {source for _, source, *_ in CONDITIONS},
        "the reference recorded the versions of other packages than its writer records":
            set(mapping(record, "versions")) == set(REFERENCE_PACKAGES),
        "the reference recorded other times than those of its steps":
            set(timing) == set(REFERENCE_TIMING) and all(valid("seconds", value) for value in timing.values()),
        "the total time of the reference is shorter than its steps":
            not all(valid("seconds", value) for value in (*spent, timing.get("total_s")))
            or sum(spent) <= timing["total_s"],
        "the reference recorded other memory readings than its writer records":
            set(memory) == set(REFERENCE_MEMORY) and valid("count", memory.get("max_rss_bytes"))
            and (memory.get("lifetime_peak_footprint_bytes") is None
                 or valid("count", memory["lifetime_peak_footprint_bytes"]))}
    return problems + [message for message, holds in formulas.items() if not holds]


def regular_file(path, size=None, keep=False, limit=None):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or size is not None and status.st_size != size:
            raise ValueError(f"{Path(path).name} is not a regular file of the recorded size")
        if limit is not None and status.st_size > limit:
            raise ValueError(f"{Path(path).name} is larger than {limit} B")
        hasher, kept, total = hashlib.sha256(), [], 0
        with open(descriptor, "rb", closefd=False) as handle:
            while block := handle.read(BLOCK_BYTES if limit is None else min(BLOCK_BYTES, limit + 1 - total)):
                hasher.update(block)
                total += len(block)
                if keep:
                    kept.append(block)
    finally:
        os.close(descriptor)
    if total != status.st_size:
        raise ValueError(f"{Path(path).name} changed while it was read")
    return {"bytes": total, "sha256": hasher.hexdigest()}, b"".join(kept) if keep else None


def read_record(path, limit=None):
    found, data = regular_file(path, keep=True, limit=limit)
    return parse(data), found["sha256"], found["bytes"]


def mapping(record, key):
    value = record.get(key)
    return value if isinstance(value, dict) else {}


def recorded_tensors(record, contract):
    tensors = mapping(record, "quantized").get("tensors")
    try:
        expected = quantized_names(check_layers(contract["config"], record.get("layers")))
    except ValueError:
        return False
    return isinstance(tensors, dict) and all(isinstance(value, str) for value in tensors.values()) \
        and set(tensors) == {f"{name}.{part}" for name in expected for part in PARTS}


def arguments(record, contract):
    try:
        kinds = check_layers(contract["config"], record.get("layers"))
        taps = check_taps(joined(mapping(record, "metadata").get("tap_layers")), record.get("layers"))
    except ValueError:
        return None
    functions, context = record.get("functions"), record.get("context")
    precision = next((key for key, value in PRECISIONS.items() if value == record.get("precision")), None)
    if functions not in [list(layout) for layout in FUNCTIONS.values()] or precision is None or not valid("count", context):
        return None
    return kinds, tuple(functions), context, taps, precision


def counted(*values):
    return all(valid("count", value) for value in values)


def relation_problems(record, precision):
    weights, asset, disk = record["weights"], record["asset"], record["disk"]
    by_dtype, storage, problems = weights["elements_by_dtype"], asset["storage_types"], []
    if not all(valid("count", value) for value in (weights["elements"], *by_dtype.values(), *storage.values())) \
            or sum(by_dtype.values()) != weights["elements"]:
        problems.append("the element counts of the weights do not add up")
    differences = {"float16_minus_weights": storage.get("Float16", 0) - by_dtype.get("torch.float16", 0),
                   **({"uint8_minus_weights": storage.get("UInt8", 0) - by_dtype.get("torch.uint8", 0)}
                      if precision == "8bit" else {})}
    if canonical({key: asset.get(key) for key in differences}) != canonical(differences):
        problems.append("the recorded differences of the storage counts and the weights do not follow from them")
    checkpoint, free = disk.get("weight_bytes"), disk.get("free_before_bytes")
    rule = None if not valid("count", checkpoint) or checkpoint % 2 else {
        "estimated_write_bytes": WRITES[precision] * (checkpoint // 2), "margin_bytes": MARGIN_BYTES,
        "required_free_bytes": required_bytes(checkpoint // 2, precision)}
    if rule is None or canonical({key: disk.get(key) for key in rule}) != canonical(rule) or not valid("count", free) \
            or free < rule["required_free_bytes"]:
        problems.append("the disk record does not follow the rule of the free disk")
    resources = mapping(record, "resources")
    swap, monitor, timing = mapping(resources, "swap"), mapping(resources, "monitor"), mapping(record, "timing")
    first, peak, increase = (swap.get(key) for key in ("first_bytes", "peak_bytes", "increase_bytes"))
    if not valid("count", increase) or increase > SWAP_INCREASE_LIMIT_BYTES:
        problems.append("the export recorded no swap increase within its limit")
    lowest, started = (mapping(mapping(monitor, key), "free_bytes") for key in ("lowest", "first"))
    samples, total, percent = monitor.get("samples"), timing.get("total_s"), mapping(monitor, "lowest").get("free_percent")
    readings = [mapping(monitor, "peak").get(key) for key in ("rss_bytes", "max_rss_bytes")]
    steps = [*STEPS, *(QUANTIZING_STEPS if precision == "8bit" else ()), "total_s"]
    seconds = all(valid("seconds", value) for value in timing.values())
    uint8, float16, layouts = by_dtype.get("torch.uint8", 0), by_dtype.get("torch.float16"), \
        mapping(record, "quantized").get("layers", [])
    shapes = [entry.get("shape") if isinstance(entry, dict) else None for entry in layouts] \
        if isinstance(layouts, list) else [None]
    formulas = {
        "the swap record names another limit than the fixed one":
            canonical(swap.get("limit_increase_bytes")) == canonical(SWAP_INCREASE_LIMIT_BYTES),
        "the first swap of the record is not that of the monitor":
            canonical(first) == canonical(mapping(monitor, "first").get("swap_used_bytes")),
        "the peak swap of the record is not that of the monitor":
            canonical(peak) == canonical(mapping(monitor, "peak").get("swap_used_bytes")),
        "the swap increase is not the peak less the first swap":
            counted(first, peak, increase) and increase == peak - first,
        "the lowest free disk is not that of the monitor":
            len(lowest) == 1 and canonical(disk.get("lowest_free_bytes")) == canonical([*lowest.values()][0]),
        "the lowest free disk of the monitor is above its first free disk":
            set(started) == set(lowest) and all(counted(lowest[key], started[key]) and lowest[key] <= started[key]
                                                for key in lowest),
        "the peak resident memory is below that of the monitor":
            counted(resources.get("max_rss_bytes"), mapping(monitor, "peak").get("max_rss_bytes"))
            and resources["max_rss_bytes"] >= monitor["peak"]["max_rss_bytes"],
        "the monitor did not run under the fixed limits without stopping":
            canonical({key: monitor.get(key) for key in MONITOR}) == canonical(MONITOR)
            and canonical(mapping(monitor, "peak").get("temporary_bytes")) == canonical(0),
        "the disk record does not have the keys that the export writes": set(disk) == set(DISK),
        "the swap record does not have the keys that the export writes": set(swap) == set(SWAP),
        "the export recorded the versions of other packages than it records":
            set(mapping(record, "versions")) == set(EXPORT_PACKAGES),
        "the monitor record does not have the keys that the monitor writes":
            set(monitor) == {*MONITOR, "samples", *READINGS}
            and all(set(mapping(monitor, key)) == keys for key, keys in READINGS.items()),
        "the monitor recorded no sample":
            valid("count", samples) and samples > 0,
        "the monitor recorded more samples than its interval allows in the time of the export":
            not (valid("count", samples) and valid("seconds", total)) or samples <= total / INTERVAL_S + 1,
        "the monitor recorded one sample and another peak swap than that sample":
            samples != 1 or canonical(mapping(monitor, "peak").get("swap_used_bytes"))
            == canonical(mapping(monitor, "first").get("swap_used_bytes")),
        "the monitor recorded one sample and another lowest free disk than that sample":
            samples != 1 or canonical(lowest) == canonical(started),
        "the resident memory that ps gave the monitor is not in KiB":
            not valid("count", mapping(monitor, "peak").get("rss_bytes")) or monitor["peak"]["rss_bytes"] % 1024 == 0,
        "the monitor recorded no positive resident memory": all(valid("count", value) and value > 0 for value in readings),
        "the resident memory that ps gave the monitor is above the peak resident memory that it read":
            not all(valid("count", value) for value in readings) or readings[0] <= readings[1],
        "the lowest free memory percentage of the monitor is not a percentage":
            valid("count", percent) and percent <= 100,
        "the times are not those of the steps of the export":
            sorted(timing) == sorted(steps) and seconds,
        "the total time is shorter than its steps":
            seconds and sum(value for key, value in timing.items() if key != "total_s") <= timing.get("total_s", 0),
        "the weight elements do not follow from the checkpoint size and the quantized elements":
            counted(disk.get("weight_bytes"), uint8, float16)
            and 8 * (2 * float16 + 2 * uint8 - disk["weight_bytes"]) == uint8,
        "the uint8 elements are not those of the quantized shapes":
            all(valid("shape", shape) and len(shape) == 2 for shape in shapes)
            and canonical(uint8) == canonical(sum(rows * columns for rows, columns in shapes))}
    return problems + [message for message, holds in formulas.items() if not holds]


def load_export(folder, contract, contract_path):
    path = folder / "export.json"
    entry, data = regular_file(path, keep=True)
    record, hashed = parse(data), entry["sha256"]
    recorded = (record.get("bundle") or {}) if isinstance(record, dict) else None
    if not isinstance(recorded, dict) or not isinstance(recorded.get("files") or {}, dict):
        raise ValueError("The export record and its bundle record must be objects")
    passed = record.get("schema") == EXPORT_SCHEMA and record.get("passed") is True and record.get("problems") == []
    problems = [] if passed else ["the export is not a passed export"]
    lowest = mapping(record, "disk").get("lowest_free_bytes")
    if canonical(record.get("checks")) != canonical({"asset": True, "free_disk": True}) or not valid("count", lowest) \
            or lowest < MARGIN_BYTES:
        problems.append("the export did not pass its checks of the asset and of the free disk")
    if record.get("upstream_revision") != contract["upstream_revision"]:
        problems.append("the export was made with another converter checkout than the contract pins")
    if not valid("count", record.get("context")):
        problems.append("the export recorded no valid context limit")
    elif record["context"] < LONGEST:
        problems.append("the export recorded a context limit below the longest condition")
    problems += layer_problems(record, contract, "export")
    if canonical(record.get("model")) != canonical(pinned_model(contract, contract_path)):
        problems.append("the export recorded another model or contract")
    if record.get("precision") not in PRECISIONS.values():
        problems.append("the export recorded no known precision")
    try:
        check_taps(joined(mapping(record, "metadata").get("tap_layers")), record.get("layers"))
    except ValueError:
        problems.append("the export recorded no valid tap layers")
    problems += [f"the export folder holds {kind}" for name, kind in (("failure.json", "a failure record"),
                                                                      (INCOMPLETE, "an unfinished bundle"))
                 if os.path.lexists(folder / name)]
    shape, bundle, files = arguments(record, contract), folder / BUNDLE, recorded.get("files") or {}
    name = None if shape is None else name_of(len(shape[0]), shape[1], shape[4])
    if shape is not None:
        try:
            problems += check_asset(record, contract, *shape) + relation_problems(record, shape[4])
            found, expected = record["asset"]["functions"], expected_functions(contract["config"], *shape[:2], shape[3])
            if any(set(found[name]) != set(expected[name]) for name in set(found) & set(expected)):
                problems.append("the functions of the asset hold other parts than inputs, outputs and states")
        except (AttributeError, KeyError, TypeError, ValueError):
            problems.append("the export record does not hold a complete inspection of its asset")
    folders = [bundle, bundle / "tokenizer", *([] if name is None else [bundle / f"{name}.aimodel"])]
    if not bundle.is_dir() or any(item.is_symlink() for item in folders):
        problems.append("the bundle, its asset or its tokenizer folder is not a real folder")
    present = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_symlink() or not path.is_dir()}
    hashes, contents, main = {}, {}, f"{recorded.get('name')}.aimodel/main.hash"
    if recorded.get("folder") != BUNDLE or present != set(files):
        problems.append("the bundle folder holds other files than its export record")
    else:
        for relative, entry in sorted(files.items()):
            try:
                if not isinstance(entry, dict) or not valid("count", entry.get("bytes")):
                    raise ValueError(f"the record of {relative} has no size")
                hashes[relative], contents[relative] = regular_file(bundle / relative, entry["bytes"],
                                                                    keep=relative in (main, "metadata.json"))
            except (OSError, ValueError):
                hashes[relative] = None
            if hashes[relative] is None or hashes[relative] != entry:
                problems.append(f"a bundle file differs from the export record: {relative}")
    if any((hashes.get(f"tokenizer/{item}") or {}).get("sha256") != contract["tokenizer"]["files"].get(item)
           for item in TOKENIZER_FILES):
        problems.append("the tokenizer files of the bundle differ from the contract")
    if hashes.get("metadata.json") and name is not None:
        expected = metadata(contract, name, shape[2], "8bit" if shape[4] == "8bit" else None, shape[3])
        try:
            written = canonical(parse(contents["metadata.json"]))
        except (PipelineError, ValueError):
            written = None
        if written != canonical(expected):
            problems.append("metadata.json of the bundle differs from the contract and the record")
    if hashes and None not in hashes.values() and canonical(recorded.get("bytes")) \
            != canonical(sum(entry["bytes"] for entry in hashes.values())):
        problems.append("the export recorded another bundle size than the sum of its files")
    if contents.get(main) is None or contents[main].hex() != recorded.get("main_hash"):
        problems.append("main.hash of the bundle differs from the export record")
    elif (hashes.get(f"{recorded.get('name')}.aimodel/main.mlirb") or {}).get("sha256") != recorded.get("main_hash"):
        problems.append("main.hash of the bundle is not the SHA-256 of its main.mlirb")
    if record.get("functions") not in [list(functions) for functions in FUNCTIONS.values()]:
        problems.append("the export recorded other functions than main or main and prefill")
    if record.get("precision") == "8bit" and record.get("functions") != list(FUNCTIONS["main,prefill"]):
        problems.append("an 8-bit export recorded other functions than main and prefill")
    if record.get("precision") == "8bit":
        if not recorded_tensors(record, contract):
            problems.append("the export did not record the tensors of exactly the quantized Linear weights of its layers")
        quantized, tensors = quantized_entry(record), folder / QUANTIZED
        try:
            found = None if quantized is None else regular_file(tensors, quantized["file"]["bytes"])[0]
        except (OSError, ValueError):
            found = None
        if found is None or found != quantized["file"]:
            problems.append("the quantized tensors of the export differ from its record")
    facts = {"bundle": bundle, "main_hash": recorded.get("main_hash"), "prefill": "prefill" in (record.get("functions") or [])}
    return record, hashed, facts, problems


def quantized_entry(record):
    quantized = record.get("quantized")
    entry, tensors = (quantized.get("file"), quantized.get("tensors")) if isinstance(quantized, dict) else (None, None)
    if not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"} or not valid("count", entry["bytes"]) \
            or not valid("text", entry["sha256"]) or not isinstance(tensors, dict) \
            or not all(isinstance(key, str) and isinstance(value, str) for key, value in tensors.items()):
        return None
    return {"file": dict(entry), "tensors": len(tensors)}


def correspondence(exported, export_hash, reference):
    precision = exported.get("precision")
    if precision not in tuple(WEIGHT_KINDS) or reference.get("weights") != WEIGHT_KINDS[precision]:
        return [f"the reference does not use the weights of the export of precision {precision!r}"]
    if precision != "8bit":
        return []
    quantized = quantized_entry(exported)
    expected = {"export_sha256": export_hash, **(quantized or {})}
    if quantized is None or canonical(reference.get("quantized")) != canonical(expected):
        return ["the reference was not restored from the quantized tensors of this export"]
    return []


def held_interrupts():
    return deferred_interrupts() if threading.current_thread() is threading.main_thread() else contextlib.nullcontext()


def run_verifier(command, timeout, log):
    with log.open("xb") as errors:
        process = None
        try:
            with held_interrupts():
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, start_new_session=True)
            stdout, _ = process.communicate(timeout=timeout)
        except BaseException:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
            raise
    return process.returncode, stdout


def printed(stdout):
    lines = [line for line in stdout.decode(errors="replace").splitlines() if line.strip()]
    try:
        value = json.loads(lines[0]) if len(lines) == 1 else None
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def condition(spec, config, arrays, facts, verifier, output):
    name, source, length, start, chunk = spec
    fixed = {"name": name, "input": source, "tokens": length, "start": start, "chunk": chunk, "rows": length - start}
    entry = {**fixed, "problems": [], "checks": {}}
    try:
        measured(entry, config, arrays, facts, verifier, output)
        json.dumps(entry, allow_nan=False)
    except Exception as error:
        entry = {**fixed, "problems": [f"the condition could not be compared ({type(error).__name__})"], "checks": {}}
    entry["passed"] = condition_passed(entry)
    return entry


def measured(entry, config, arrays, facts, verifier, output):
    name, source, length, start, chunk, rows = (entry[key] for key in ("name", "input", "tokens", "start", "chunk", "rows"))
    ids, vocabulary, problems = tokens(source, config)[:length], config["vocab_size"], entry["problems"]
    (output / f"{name}.tokens.json").write_text(json.dumps(ids))
    command = [str(verifier), "logits", "--model", str(facts["bundle"]), "--tokens", str(output / f"{name}.tokens.json"),
               "--start", str(start), "--chunk", str(chunk), "--output", str(output / f"{name}.f16")]
    began = time.perf_counter()
    try:
        code, stdout = run_verifier(command, TIMEOUT_S, output / f"{name}.log")
    except subprocess.TimeoutExpired:
        problems.append(f"the verifier did not finish within {TIMEOUT_S} s")
    else:
        if code:
            problems.append(f"the verifier exited with {code}")
        record = printed(stdout)
        if record is None:
            problems.append("the verifier did not print one JSON object")
        else:
            record, malformed = validated(record)
            entry["verifier"] = record
            problems += [f"the verifier record is malformed ({key})" for key in malformed]
            if not malformed:
                expected = expected_verifier(entry, config, facts["main_hash"], facts["prefill"], prefill_chunk())
                problems += [f"the verifier recorded {key} {record.get(key)!r}, not {value!r}"
                             for key, value in expected.items() if record.get(key) != value]
                if start == 0 and "prefix_function" in record:
                    problems.append("the verifier recorded a prefix function for a condition that starts at position 0")
    entry["seconds"] = time.perf_counter() - began
    if problems:
        return
    data, content_hash = read_once(output / f"{name}.f16")
    entry["output"] = {"bytes": len(data), "sha256": content_hash}
    if len(data) != logits_bytes(rows, vocabulary):
        problems.append(f"the logits file holds {len(data)} B, not {logits_bytes(rows, vocabulary)} B")
        return
    entry.update(measure_condition(entry, vocabulary, data, arrays))
    if entry["verifier"]["non_finite_count"] != entry["non_finite"]:
        problems.append(f"the verifier counted {entry['verifier']['non_finite_count']} non-finite values, not the "
                        f"{entry['non_finite']} of its file")


def compare(export_folder, reference_folder, verifier, output, contract_path=CONTRACT):
    if os.path.lexists(output):
        raise FileExistsError(f"The comparison must be written to a new folder: {output}")
    contract = load_contract(contract_path)
    output.mkdir(parents=True)
    problems, reference, reference_hash, arrays, exported, export_hash, facts = [], {}, None, {}, {}, None, None
    try:
        reference, reference_hash, arrays, found = load_reference(reference_folder, contract, contract_path)
        problems += found
    except Exception as error:
        problems.append(f"the reference record is unreadable ({type(error).__name__})")
    try:
        exported, export_hash, facts, found = load_export(export_folder, contract, contract_path)
        problems += found
    except Exception as error:
        problems.append(f"the export record is unreadable ({type(error).__name__})")
    if None not in (reference_hash, export_hash):
        if exported.get("layers") != reference.get("layers"):
            problems.append("the export and the reference are of other layer counts")
        if not problems:
            problems += correspondence(exported, export_hash, reference)
    verifier_record = None
    if not problems:
        try:
            verifier = Path(verifier).resolve()
            verifier_record = regular_file(verifier)[0]
        except Exception as error:
            problems.append(f"the verifier is unreadable ({type(error).__name__})")
    conditions = [] if problems else [condition(spec, contract["config"], arrays, facts, verifier, output)
                                      for spec in CONDITIONS]
    result = {"schema": SCHEMA, "problems": problems,
              "passed": comparison_passed(problems, conditions), "criteria": CRITERIA, "known_call": KNOWN_CALL,
              "reference": {"sha256": reference_hash, "layers": reference.get("layers")},
              "export": {"sha256": export_hash, "layers": exported.get("layers"), "functions": exported.get("functions"),
                         "main_hash": facts["main_hash"] if facts else None},
              "verifier": verifier_record, "conditions": conditions}
    try:
        write_json(output / "comparison.json", result)
    except Exception as error:
        result = {"schema": SCHEMA, "passed": False, "conditions": [],
                  "problems": [f"the comparison could not be recorded ({type(error).__name__})"]}
        write_json(output / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare the all-position logits that coreai-verify writes for the "
                                                 "twelve conditions with the FP32 reference of the same first layers.")
    for name in ("export", "reference", "verifier", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.export, args.reference, args.verifier, args.output)
    print(json.dumps({"passed": result["passed"], "problems": result["problems"],
                      "conditions": {item["name"]: {"passed": item["passed"], "max_abs_logit": item.get("max_abs_logit")}
                                     for item in result["conditions"]}}, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
