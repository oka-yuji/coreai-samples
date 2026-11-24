import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from ...runner import deferred_interrupts
from ...storage import canonical, digest, write_json
from .compare import BLOCK_ROWS, CONTRACT
from .layers import SCHEMA as RUN_SCHEMA, terminated_as_interrupt, tokens
from .native import (TIMEOUT_S, VERIFIER, correspondence, layer_problems, load_export, mapping, pinned_model,
                     prefill_chunk, prefix_record, printed, read_record, regular_file, run_verifier, valid)
from .prepare import load_contract

SCHEMA = "gemma4-31b-speculation-outputs/1"
SMALLEST, MEDIAN, MIXUP = 0.99, 0.999, 0.5
CONDITIONS = (("short-c1", "short", 0, 1), ("short-c8", "short", 0, 8), ("long-main", "long", 0, 1400),
              ("long-prefill", "long", 1100, 1))
RECORD = {**VERIFIER, "scoring_call_s": "seconds"}
FILE = {"function": "text", "name": "text", "rows": "count", "width": "count", "sha256": "text"}
OUTPUTS = ("hidden_states", "tap_hidden_states")
FLOAT32_BYTES = 4
HEADER_LIMIT = 1 << 16
RECORD_LIMIT = 1 << 20


def cosines(x, y):
    import numpy as np

    dots, norms = (x * y).sum(axis=-1), np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)


def decided(squares, own, values, first):
    import numpy as np

    others = {key: value for key, value in squares.items() if key != own}
    nearest = min(others, key=others.get)
    smallest, median = float(values.min()), float(np.median(values))
    below = [[first + int(row), float(values[row])] for row in np.flatnonzero(values < SMALLEST)]
    defined = others[nearest] > 0
    return {"squares": {str(key): value for key, value in squares.items()}, "nearest": str(nearest),
            "ratio": squares[own] / others[nearest] if defined else None,
            "cosine": {"smallest": smallest, "smallest_position": first + int(values.argmin()), "median": median},
            "below": below,
            "checks": {"mixup": defined and all(squares[own] <= MIXUP * value for value in others.values()),
                       "cosine_smallest": smallest >= SMALLEST, "cosine_median": median >= MEDIAN}}


def taps(asset, reference, asset_taps, reference_taps, hidden, first):
    import numpy as np

    squares = np.zeros((len(asset_taps), len(reference_taps)))
    values = np.zeros((len(asset_taps), len(asset)))
    for start in range(0, len(asset), BLOCK_ROWS):
        x = np.asarray(asset[start:start + BLOCK_ROWS], dtype=np.float64)
        y = np.asarray(reference[start:start + BLOCK_ROWS], dtype=np.float64)
        for index, tap in enumerate(asset_taps):
            section = x[:, index * hidden:(index + 1) * hidden]
            for other, layer in enumerate(reference_taps):
                squares[index, other] += float(np.square(section - y[:, other * hidden:(other + 1) * hidden]).sum())
            own = reference_taps.index(tap)
            values[index, start:start + len(x)] = cosines(section, y[:, own * hidden:(own + 1) * hidden])
    return [{"layer": tap, **decided(dict(zip(reference_taps, map(float, squares[index]))), tap, values[index], first)}
            for index, tap in enumerate(asset_taps)]


def hidden(asset, norm, last, first):
    import numpy as np

    squares = {"final_norm": 0.0, "last_layer": 0.0}
    values = np.zeros(len(asset))
    for start in range(0, len(asset), BLOCK_ROWS):
        x, y, z = (np.asarray(array[start:start + BLOCK_ROWS], dtype=np.float64) for array in (asset, norm, last))
        squares["final_norm"] += float(np.square(x - y).sum())
        squares["last_layer"] += float(np.square(x - z).sum())
        values[start:start + len(x)] = cosines(x, y)
    return decided(squares, "final_norm", values, first)


def validated(record):
    kept, malformed = {}, []
    for key, kind in RECORD.items():
        if key in record and valid(kind, record[key]):
            kept[key] = record[key]
        elif key in record or key != "prefix_function":
            malformed.append(key)
    files = record.get("outputs")
    if isinstance(files, list) and all(isinstance(item, dict) and set(item) == set(FILE)
                                       and all(valid(kind, item[key]) for key, kind in FILE.items()) for item in files):
        kept["outputs"] = files
    else:
        malformed.append("outputs")
    return kept, malformed


def read_file(path, size):
    try:
        return regular_file(path, size, keep=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"{Path(path).name} is not a regular file{'' if size is None else f' of {size} B'} "
                         f"({type(error).__name__})") from None


def npy(data):
    import numpy as np

    try:
        return np.lib.format.read_array(io.BytesIO(data), allow_pickle=False)
    except Exception as error:
        raise ValueError(f"not an array in the .npy format ({type(error).__name__})") from None


def load_reference(folder, contract, contract_path):
    import numpy as np

    if (folder / "record.json").stat().st_size > RECORD_LIMIT:
        raise ValueError("The reference record is larger than a record of a run can be")
    record, hashed, _ = read_record(folder / "record.json")
    if not isinstance(record, dict):
        raise ValueError("The reference record is not an object")
    problems = [] if (record.get("schema"), record.get("implementation")) == (RUN_SCHEMA, "original") else [
        "the reference is not a run of the original implementation"]
    layered = layer_problems(record, contract, "reference")
    problems += layered
    if not layered and canonical(record.get("tap_layers")) != canonical(list(range(1, record["layers"] + 1))):
        problems.append("the reference does not hold the output of every layer")
    if canonical(record.get("model")) != canonical(pinned_model(contract, contract_path)):
        problems.append("the reference recorded another model or contract")
    arrays = {}
    if problems:
        return record, hashed, arrays, problems
    size, count = contract["config"]["hidden_size"], record["layers"]
    for name in ("short", "long"):
        ids, entry = tokens(name, contract["config"]), mapping(mapping(record, "inputs"), name)
        if canonical([entry.get("positions"), entry.get("tokens_sha256")]) != canonical([len(ids), digest(ids)]):
            problems.append(f"the reference did not read the {name} input")
        for output, width in zip(OUTPUTS, (size, size * count)):
            recorded = mapping(entry, output)
            expected = {"dtype": "float32", "shape": [len(ids), width], "non_finite": 0}
            if not valid("count", recorded.get("bytes")) \
                    or canonical({key: recorded.get(key) for key in expected}) != canonical(expected):
                problems.append(f"the reference record does not give the {name} {output} as finite float32 "
                                f"{expected['shape']}")
                continue
            data_bytes = len(ids) * width * FLOAT32_BYTES
            if not data_bytes <= recorded["bytes"] <= data_bytes + HEADER_LIMIT:
                problems.append(f"the reference record gives a size of the {name} {output} that its shape does not "
                                f"allow")
                continue
            try:
                found, content = read_file(folder / f"{name}.{output}.npy", recorded["bytes"])
                array = npy(content)
            except ValueError as error:
                problems.append(f"the {name} {output} of the reference: {error}")
                continue
            if found["sha256"] != recorded.get("sha256"):
                problems.append(f"the {name} {output} differ from the SHA-256 in the reference record")
            elif array.dtype != np.float32 or array.shape != (len(ids), width):
                problems.append(f"the {name} {output} of the reference are {array.dtype} {list(array.shape)}")
            elif not np.isfinite(array).all():
                problems.append(f"the {name} {output} of the reference hold non-finite values")
            else:
                arrays[name, output] = array
    return record, hashed, arrays, problems


def inputs(export_folder, reference_folder, verifier, contract, contract_path):
    problems, found = [], {}
    try:
        reference, reference_hash, arrays, listed = load_reference(reference_folder, contract, contract_path)
        problems += listed
        found.update(reference=reference, reference_hash=reference_hash, arrays=arrays)
    except Exception as error:
        problems.append(f"the reference record is unreadable ({type(error).__name__})")
    try:
        exported, export_hash, facts, listed = load_export(export_folder, contract, contract_path)
        problems += listed
        found.update(exported=exported, export_hash=export_hash, facts=facts)
    except Exception as error:
        problems.append(f"the export record is unreadable ({type(error).__name__})")
    if not problems:
        if not found["facts"]["prefill"]:
            problems.append("the export has no prefill function")
        if found["exported"].get("layers") != found["reference"].get("layers"):
            problems.append("the export and the reference are of other layer counts")
        problems += correspondence(found["exported"], found["export_hash"], found["reference"])
    try:
        found["path"] = Path(verifier).resolve()
        found["verifier"] = regular_file(found["path"])[0]
    except Exception as error:
        problems.append(f"the verifier is unreadable ({type(error).__name__})")
    if problems:
        raise ValueError(f"The outputs cannot be compared: {'; '.join(problems)}")
    return found


def condition(spec, found, contract, verifier, output):
    name, source, start, chunk = spec
    length = len(tokens(source, contract["config"]))
    fixed = {"name": name, "input": source, "tokens": length, "start": start, "chunk": chunk, "rows": length - start}
    entry = {**fixed, "problems": [], "outputs": {}, "checks": {}}
    try:
        measured(entry, found, contract, verifier, output)
        json.dumps(entry, allow_nan=False)
    except Exception as error:
        entry = {**fixed, "problems": [f"the condition could not be compared ({type(error).__name__})"], "outputs": {},
                 "checks": {}}
    entry["passed"] = not entry["problems"] and bool(entry["checks"]) and all(entry["checks"].values())
    return entry


def measured(entry, found, contract, verifier, output):
    import numpy as np

    name, source, start, chunk, rows = (entry[key] for key in ("name", "input", "start", "chunk", "rows"))
    config, facts, problems = contract["config"], found["facts"], entry["problems"]
    ids, size, asset_taps = tokens(source, config), config["hidden_size"], found["exported"]["metadata"]["tap_layers"]
    expected_files = {("main", "hidden_states"): (rows, size),
                      ("main", "tap_hidden_states"): (rows, size * len(asset_taps))}
    if start:
        expected_files["prefill", "tap_hidden_states"] = (start, size * len(asset_taps))
    paths = {key: output / f"{name}.{'prefill.' if key[0] == 'prefill' else ''}{key[1]}.f16" for key in expected_files}
    (output / f"{name}.tokens.json").write_text(json.dumps(ids))
    command = [str(verifier), "logits", "--model", str(facts["bundle"]), "--tokens",
               str(output / f"{name}.tokens.json"), "--start", str(start), "--chunk", str(chunk), "--output",
               str(output / f"{name}.f16")]
    for (function, kind), path in paths.items():
        command += [f"--{function}-output", f"{kind}={path}"]
    began = time.perf_counter()
    try:
        code, stdout = run_verifier(command, TIMEOUT_S, output / f"{name}.log")
    except subprocess.TimeoutExpired:
        problems.append(f"the verifier did not finish within {TIMEOUT_S} s")
        return
    entry["seconds"] = time.perf_counter() - began
    if code:
        problems.append(f"the verifier exited with {code}")
    record = printed(stdout)
    if record is None:
        problems.append("the verifier did not print one JSON object")
        return
    record, malformed = validated(record)
    entry["verifier"] = {key: value for key, value in record.items() if key != "outputs"}
    problems += [f"the verifier record is malformed ({key})" for key in malformed]
    if malformed:
        return
    expected = {"record": "logits", "token_count": len(ids), "start": start, "chunk": chunk,
                "tokens_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
                "shape": [rows, config["vocab_size"]], "dtype": "float16", "byte_order": "little",
                "main_calls": math.ceil(rows / chunk), "main_hash": facts["main_hash"],
                **prefix_record(start, prefill_chunk(), facts["prefill"])}
    problems += [f"the verifier recorded {key} {record.get(key)!r}, not {value!r}"
                 for key, value in expected.items() if record.get(key) != value]
    if start == 0 and "prefix_function" in record:
        problems.append("the verifier recorded a prefix function for a condition that starts at position 0")
    listed = {(item["function"], item["name"]): item for item in record["outputs"]}
    if len(listed) != len(record["outputs"]) or set(listed) != set(expected_files):
        problems.append(f"the verifier recorded the outputs {sorted(listed)}, not {sorted(expected_files)}")
    else:
        problems += [f"the verifier recorded {key[0]} {key[1]} of {[item['rows'], item['width']]}, not {list(shape)}"
                     for key, shape in expected_files.items()
                     if [(item := listed[key])["rows"], item["width"]] != list(shape)]
    if problems:
        return
    reference = found["arrays"]
    norm, layers_taps = reference[source, "hidden_states"], reference[source, "tap_hidden_states"]
    reference_taps = list(range(1, found["reference"]["layers"] + 1))
    for key, (count, width) in expected_files.items():
        label = ".".join(key)
        try:
            found_file, data = read_file(paths[key], count * width * 2)
        except ValueError as error:
            problems.append(f"the {label} file: {error}")
            continue
        content_hash = found_file["sha256"]
        if content_hash != listed[key]["sha256"]:
            problems.append(f"the {label} file has the SHA-256 {content_hash}, not that of the verifier record")
            continue
        asset = np.frombuffer(data, dtype="<f2").reshape(count, width)
        first = start if key[0] == "main" else 0
        span = slice(first, first + count)
        finite = np.isfinite(asset)
        result = {"rows": count, "width": width, "sha256": content_hash, "non_finite": int(asset.size - finite.sum()),
                  "non_finite_rows": [first + int(row) for row in np.flatnonzero(~finite.all(axis=1))]}
        entry["outputs"][label] = result
        entry["checks"][f"{label}.finite"] = not result["non_finite"]
        if result["non_finite"]:
            continue
        if key[1] == "hidden_states":
            last = layers_taps[span, (len(reference_taps) - 1) * size:]
            result.update(hidden(asset, norm[span], last, first))
            entry["checks"].update({f"{label}.{check}": passed for check, passed in result["checks"].items()})
        else:
            result["sections"] = taps(asset, layers_taps[span], asset_taps, reference_taps, size, first)
            for section in result["sections"]:
                entry["checks"].update({f"{label}.{section['layer']}.{check}": passed
                                        for check, passed in section["checks"].items()})


def compare(export_folder, reference_folder, verifier, output, contract_path=CONTRACT):
    if os.path.lexists(output):
        raise FileExistsError(f"The comparison must be written to a new folder: {output}")
    contract = load_contract(contract_path)
    found = inputs(export_folder, reference_folder, verifier, contract, contract_path)
    made = False
    try:
        with deferred_interrupts():
            output.mkdir(parents=True)
            made = True
        conditions = [condition(spec, found, contract, found["path"], output) for spec in CONDITIONS]
        reference, exported = found["reference"], found["exported"]
        result = {"schema": SCHEMA, "passed": all(entry["passed"] for entry in conditions),
                  "criteria": {"cosine_smallest": SMALLEST, "cosine_median": MEDIAN, "mixup_ratio": MIXUP},
                  "reference": {"sha256": found["reference_hash"], "layers": reference["layers"],
                                "tap_layers": reference["tap_layers"], "weights": reference["weights"]},
                  "export": {"sha256": found["export_hash"], "precision": exported["precision"],
                             "functions": exported["functions"], "tap_layers": exported["metadata"]["tap_layers"],
                             "main_hash": found["facts"]["main_hash"]},
                  "verifier": found["verifier"], "conditions": conditions}
        with deferred_interrupts():
            write_json(output / "comparison.json", result)
        return result
    except BaseException as error:
        if not made:
            raise
        with deferred_interrupts():
            (output / "comparison.json").unlink(missing_ok=True)
            write_json(output / "failure.json", {"schema": SCHEMA, "passed": False, "error": type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description="Compare the outputs for speculation that coreai-verify writes from "
                                                 "an export of the first layers of Gemma 4 31B with the FP32 outputs "
                                                 "of every layer of the original implementation.")
    for name in ("export", "reference", "verifier", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    with terminated_as_interrupt():
        result = compare(args.export, args.reference, args.verifier, args.output)
    print(json.dumps({"passed": result["passed"],
                      "conditions": {entry["name"]: {"passed": entry["passed"], "problems": entry.get("problems", []),
                                                     "failing": sorted(key for key, value in entry["checks"].items()
                                                                       if not value)}
                                     for entry in result["conditions"]}}, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
