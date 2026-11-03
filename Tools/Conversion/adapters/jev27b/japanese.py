import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import random
import statistics
import sys

from ...storage import read_json, write_json
from .compare import CLI_TOLERANCE, completeness, finite
from .decision import DECISION, SUMS, sha256
from .fixtures import KINDS, SEED, TASKS, digest, jglue_row, read_task, serialize
from .prepare import check_integrity, distributed_tokenizer, load_contract
from .score import auroc, expectation, mean
from .typed_head import FIXED_OPTIONS, band, decide, option_lines, render, tokenize
from .verify_merge import largest, top

SCHEMA = "typed-decision-japanese/1"
JDEV_SCHEMA = "typed-decision-jdev/1"
ROWS_SCHEMA = "typed-decision-japanese-rows/1"
ADOPTION_SCHEMA = "typed-decision-prompt-adoption/1"
WRAPPERS_SCHEMA = "typed-decision-wrappers/1"
VARIANTS = ("ja", "en")
NOUL = "JNLI:noul"
GROUPS = ("JNLI", NOUL, "JCommonsenseQA", "JSTS")
NLI_LABELS = ("entailment", "contradiction", "neutral")
TEXT_FIELDS = {"JNLI": ("sentence1", "sentence2"), "JCommonsenseQA": ("question",), "JSTS": ("sentence1", "sentence2")}
JDEV_SEED = f"{SEED}:Jdev"
JDEV_ROWS = 100
PERMUTATIONS = {"identity": (0, 1, 2, 3, 4), "reverse": (4, 3, 2, 1, 0), "rotate1": (1, 2, 3, 4, 0),
                "rotate2": (2, 3, 4, 0, 1)}
ONE = dict.fromkeys(KINDS, 1.0)
NAMES = {"JNLI": "JNLI choice", NOUL: "JNLI noul", "JCommonsenseQA": "JCommonsenseQA", "JSTS": "JSTS"}
STAGE_NAMES = {"1m": "1'", "3": "3"}
LABELS = {"accuracy": "accuracy", "auroc": "AUROC", "brier_hard": "hard Brier", "mae": "MAE of E[p]", "pearson": "Pearson",
          "spearman": "Spearman"}
FIELDS = {"JNLI": ("accuracy",), NOUL: ("accuracy", "auroc", "brier_hard"), "JCommonsenseQA": ("accuracy",),
          "JSTS": ("mae", "pearson", "spearman")}
REQUEST = ("id", "kind", "state", "question", "options")
MANIFEST = ("id", "source_id", "task", "group", "variant", "permutation", "order", "kind", "label", "original_label",
            "row_sha256", "text_sha256", "token_count", "rejected")
JDEV_RULE = ("rows of the pinned JGLUE validation files that are not in J or F and whose sentence1 or sentence2 (JNLI, "
             "JSTS) or question (JCommonsenseQA) equals no sentence1, sentence2 or question of any JGLUE row of J or F; "
             "rows whose options do not render are left out; one random.Random(seed) samples the sorted eligible IDs "
             "of JNLI, JCommonsenseQA and JSTS in this order")
ADOPTION_RULE = ("per task on Jdev at temperature 1: the variant with more correct JNLI choice rows, more correct "
                 "JCommonsenseQA rows, or the smaller MAE of the expected JSTS score; a tie adopts ja. JNLI noul follows "
                 "the JNLI choice adoption as a common policy, which is not a claim that it is best for noul.")
DEFINITIONS = {
    "scope": "Descriptive results on 200 rows per task without a pass line and without intervals; they are not a claim "
             "of Japanese quality or of calibration",
    "accuracy": "JNLI choice and JCommonsenseQA: the first largest probability is the label, a permuted row maps its "
                "answer back to the original option; JNLI noul: P(true) > 0.5 is true and P(true) = 0.5 is false",
    "auroc": "P(true) of the JNLI noul rows ranked with entailment as the positive class, ties count one half, "
             "undefined with one class",
    "brier_hard": "mean of (P(true) - y)^2 with y = 1 for entailment and 0 for contradiction and neutral",
    "jsts": "E[p] = sum k p(k) against the continuous label: MAE, Pearson and Spearman (average ranks for ties), "
            "undefined when either side is constant",
    "difference": "stage 3 minus stage 1' on the same rows, wrapper and temperature: quantization and execution path "
                  "together",
    "temperatures": "1 and the adopted temperatures of the calibrated bundle (decision.json per_kind); probabilities "
                    "are recomputed from the 24 logits of each stage",
    "permutations": "stage 3, adopted JCommonsenseQA variant: the options in the four fixed orders, the label position "
                    "follows its option, and the answers are mapped back to the original options",
}


def load_wrappers(path):
    data = read_json(path)
    variants = data.get("variants")
    if data.get("schema") != WRAPPERS_SCHEMA or data.get("template") != "bare-v1" or not isinstance(variants, dict) \
            or set(variants) != set(VARIANTS) or any(set(variants[variant]) != set(GROUPS) for variant in VARIANTS):
        raise ValueError("Wrappers must define JNLI, JNLI noul, JCommonsenseQA and JSTS for ja and en")
    for variant in VARIANTS:
        for group, item in variants[variant].items():
            keys = {"prefix", "suffix", "options"} if group == "JNLI" else {"prefix", "suffix"}
            if set(item) != keys or not all(isinstance(item[key], str) for key in ("prefix", "suffix")):
                raise ValueError(f"Invalid wrapper: {variant} {group}")
            if group == "JNLI" and len(option_lines("choice", item["options"])) != len(NLI_LABELS):
                raise ValueError("JNLI wrappers need three options in the order of the JNLI labels")
    return data


def restore(entries, pins, jglue):
    tasks, units = {}, []
    for entry in entries:
        _, task, native = entry["id"].split(":", 2)
        if task not in tasks:
            tasks[task] = read_task(jglue, pins, task)
        row = jglue_row(task, tasks[task][native], pins)
        if entry["source"] != f"jglue:{task}" or digest(row) != entry["sha256"] or row["kind"] != entry["kind"] \
                or len(row["options"]) != entry["n_options"]:
            raise ValueError(f"Restored row differs from its record: {entry['id']}")
        units.append({"id": entry["id"], "task": task, "row": row, "row_sha256": entry["sha256"]})
    return units


def wrapped(unit, variant, group, wrappers, permutation="identity"):
    row, item = unit["row"], wrappers["variants"][variant][group]
    order = PERMUTATIONS[permutation] if group == "JCommonsenseQA" else None
    if permutation != "identity" and order is None:
        raise ValueError("Only JCommonsenseQA rows are permuted")
    request = {"state": row["state"], "question": item["prefix"] + row["question"] + item["suffix"]}
    if group == NOUL:
        request.update(kind="noul", options=list(FIXED_OPTIONS["noul"]), label=int(row["label"] == 0))
    elif group == "JNLI":
        request.update(kind="choice", options=list(item["options"]), label=row["label"])
    elif order is not None:
        request.update(kind="choice", options=[row["options"][index] for index in order],
                       label=order.index(row["label"]))
    else:
        request.update(kind=row["kind"], options=list(row["options"]), label=row["label"])
    derived = unit["id"] + (":noul" if group == NOUL else "")
    return {"id": f"{derived}|{variant}|{permutation}", "source_id": unit["id"], "task": unit["task"], "group": group,
            "variant": variant, "permutation": permutation, "order": list(order) if order else None, **request,
            "original_label": row["label"], "row_sha256": unit["row_sha256"]}


def plan(units, wrappers, groups, permuted=None):
    rows = [wrapped(unit, variant, group, wrappers) for variant in VARIANTS for group in groups
            for unit in units if unit["task"] == group.split(":")[0]]
    if permuted is not None:
        rows += [wrapped(unit, permuted, "JCommonsenseQA", wrappers, name) for name in PERMUTATIONS if name != "identity"
                 for unit in units if unit["task"] == "JCommonsenseQA"]
    return rows


def tokenized(rows, encode, limit):
    for row in rows:
        text = render(row["kind"], row["state"], row["question"], row["options"])
        ids = encode(text)
        row.update(text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), token_count=len(ids), tokens=ids,
                   band=band(len(ids)), rejected=len(ids) > limit)
    return rows


def manifest_rows(rows):
    return [{**{field: row[field] for field in MANIFEST}, "n_options": len(row["options"])} for row in rows]


def fixed_texts(fixtures, pins, jglue):
    tasks = {task: read_task(jglue, pins, task) for task in TASKS}
    texts, identifiers = set(), set()
    for name in ("F", "J"):
        for entry in read_json(Path(fixtures) / f"{name}.json")["rows"]:
            if entry["source"].startswith("jglue:"):
                _, task, native = entry["id"].split(":", 2)
                identifiers.add(entry["id"])
                texts.update(tasks[task][native][field] for field in TEXT_FIELDS[task])
    return tasks, identifiers, texts


def renders(row):
    try:
        option_lines(row["kind"], row["options"])
    except ValueError:
        return False
    return True


def shares(task, record, texts):
    return any(record[field] in texts for field in TEXT_FIELDS[task])


def jdev_document(tasks, pins, identifiers, texts):
    rng = random.Random(JDEV_SEED)
    rows, pools, chosen = [], {}, []
    for task in TASKS:
        counts, eligible = Counter(), []
        for native in sorted(tasks[task]):
            record = tasks[task][native]
            if f"jglue:{task}:{native}" in identifiers:
                counts["in_fixed_sets"] += 1
            elif shares(task, record, texts):
                counts["shares_text"] += 1
            elif not renders(jglue_row(task, record, pins)):
                counts["not_renderable"] += 1
            else:
                eligible.append(native)
        for native in rng.sample(eligible, JDEV_ROWS):
            row = jglue_row(task, tasks[task][native], pins)
            chosen.append((task, tasks[task][native]))
            rows.append({"id": f"jglue:{task}:{native}", "source": f"jglue:{task}", "kind": row["kind"],
                         "n_options": len(row["options"]), "sha256": digest(row)})
        pools[task] = {"rows": len(tasks[task]), "in_fixed_sets": counts["in_fixed_sets"],
                       "shares_text": counts["shares_text"], "not_renderable": counts["not_renderable"],
                       "eligible": len(eligible)}
    identities = {row["id"] for row in rows}
    checks = {"rows": len(identities) == len(rows) == JDEV_ROWS * len(TASKS),
              "outside_fixed_sets": not identities & identifiers,
              "no_shared_text": not any(shares(task, record, texts) for task, record in chosen)}
    return {"schema": JDEV_SCHEMA, "passed": all(checks.values()), "checks": checks, "seed": JDEV_SEED,
            "per_task": JDEV_ROWS, "rule": JDEV_RULE, "pools": pools, "excluded_texts": len(texts),
            "summary": {"rows": len(rows), "source": dict(Counter(row["source"] for row in rows)),
                        "kind": dict(sorted(Counter(row["kind"] for row in rows).items()))}, "rows": rows}


def write_jdev(args):
    output = args.output / "jdev.json"
    if output.exists():
        raise FileExistsError("jdev.json must be written to a new directory")
    contract = load_contract(args.model)
    pins = read_json(args.fixtures / "jglue.json")
    integrity = check_integrity(args.jglue, pins["repository"], pins["revision"])
    tasks, identifiers, texts = fixed_texts(args.fixtures, pins, args.jglue)
    document = jdev_document(tasks, pins, identifiers, texts)
    document["sources"] = {"model": contract["model_id"],
                           "jglue": {"repository": pins["repository"], "revision": pins["revision"]},
                           "fixtures": {name: sha256(args.fixtures / f"{name}.json") for name in ("F", "J", "jglue")},
                           "integrity": integrity}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(output, document)
    return 0 if document["passed"] else 1


def frozen_adoption(path, wrappers_sha256):
    adoption = read_json(path)
    adopted = adoption.get("adopted")
    if adoption.get("schema") != ADOPTION_SCHEMA or adoption.get("frozen") is not True \
            or adoption.get("wrappers_sha256") != wrappers_sha256 or not isinstance(adopted, dict) \
            or set(adopted) != set(GROUPS) or not set(adopted.values()) <= set(VARIANTS):
        raise ValueError("The prompt adoption is not frozen for these wrappers")
    return adoption


def set_rows(args, name, tokenizer):
    pins = read_json(args.fixtures / "jglue.json")
    check_integrity(args.jglue, pins["repository"], pins["revision"])
    wrappers = load_wrappers(args.wrappers)
    limit = load_contract(args.model)["limits"]["input_tokens"]
    if name == "Jdev":
        source = read_json(args.jdev)
        if source.get("schema") != JDEV_SCHEMA or source.get("passed") is not True:
            raise ValueError("Jdev did not pass its checks")
        rows = plan(restore(source["rows"], pins, args.jglue), wrappers, TASKS)
        sources = {"jdev_sha256": sha256(args.jdev)}
    else:
        adoption = frozen_adoption(args.adoption, sha256(args.wrappers))
        rows = plan(restore(read_json(args.fixtures / "J.json")["rows"], pins, args.jglue), wrappers, GROUPS,
                    adoption["adopted"]["JCommonsenseQA"])
        sources = {"fixture_sha256": sha256(args.fixtures / "J.json"), "adoption_sha256": sha256(args.adoption)}
    return tokenized(rows, lambda text: tokenize(tokenizer, text), limit), sources


def write_inputs(args):
    if (args.output / "check.json").exists():
        raise FileExistsError("Inputs must be written to a new directory")
    rows, sources = set_rows(args, args.set, distributed_tokenizer(args.snapshot))
    args.output.mkdir(parents=True, exist_ok=True)
    data = serialize({"set": args.set, "rows": [{key: row[key] for key in REQUEST} for row in rows]}).encode("utf-8")
    (args.output / f"input-{args.set}.json").write_bytes(data)
    write_json(args.output / f"rows-{args.set}.json",
               {"schema": ROWS_SCHEMA, "set": args.set, "wrappers_sha256": sha256(args.wrappers),
                "tokenizer_sha256": sha256(args.snapshot / "tokenizer.json"), **sources, "rows": manifest_rows(rows)})
    counts = Counter("/".join((row["variant"], row["group"], row["permutation"])) for row in rows)
    write_json(args.output / "check.json",
               {"passed": len({row["id"] for row in rows}) == len(rows), "set": args.set, "rows": len(rows),
                "by_variant_group_permutation": dict(sorted(counts.items())),
                "rejected": sum(row["rejected"] for row in rows), "longest": max(row["token_count"] for row in rows),
                "tokens": sum(row["token_count"] for row in rows), "sha256": hashlib.sha256(data).hexdigest(),
                **sources})
    return 0


def bundle_identity(path):
    decision = read_json(path / DECISION)
    sums = {name: value for value, name in (line.split("  ", 1) for line in (path / SUMS).read_text().splitlines())}
    return {"name": path.name, "version": sha256(path / DECISION),
            "asset_sha256": sums[f"{decision['backbone']['asset']}/main.mlirb"],
            "temperatures": decision["temperatures"]["per_kind"], "source": decision["temperatures"]["source"],
            "calibration_id": decision["temperatures"]["calibration_id"]}


def manifest_problems(rows, manifest):
    found = manifest.get("rows", [])
    if [row.get("id") for row in found] != [row["id"] for row in rows]:
        return ["rows differ from the manifest in identity or order"]
    return [row["id"] for row, entry in zip(manifest_rows(rows), found, strict=True) if row != entry]


def swift_problems(rows, swift, bundle):
    problems = completeness(rows, swift)
    if problems:
        return problems
    problems = [f"{identifier}: not finite or not a distribution" for identifier in finite(swift["rows"])]
    record = swift.get("bundle", {})
    if record.get("version") != bundle["version"] or record.get("asset_sha256") != bundle["asset_sha256"]:
        problems.append("the stage 3 run used another bundle or asset")
    if not problems:
        problems = [f"{row['id']}: shipping probabilities differ from the bundle temperatures"
                    for row, answer in zip(rows, swift["rows"], strict=True)
                    if largest(answer["probabilities"], probabilities(answer["logits"], row,
                                                                      bundle["temperatures"])) > CLI_TOLERANCE]
    return problems


def reference_problems(rows, reference):
    problems = [] if reference.get("passed") is True else ["the stage 1' reference did not pass its checks"]
    found = reference.get("rows", [])
    if [row.get("id") for row in found] != [row["id"] for row in rows]:
        return problems + ["stage 1' rows differ in identity or order"]
    for row, entry in zip(rows, found, strict=True):
        logits = entry.get("logits1m")
        if any(entry.get(key) != row[key] for key in ("kind", "options", "row_sha256", "text_sha256", "token_count",
                                                       "tokens", "rejected")):
            problems.append(f"{row['id']}: stage 1' input differs")
        elif not isinstance(logits, list) or len(logits) != 24 \
                or not all(type(value) in (int, float) and math.isfinite(value) for value in logits):
            problems.append(f"{row['id']}: stage 1' logits are missing or not finite")
    return problems


def label_problems(rows):
    bad = []
    for row in rows:
        label = row["label"]
        if row["kind"] == "score":
            valid = type(label) in (int, float) and math.isfinite(label) and 0 <= label <= 5
        else:
            valid = type(label) is int and 0 <= label < len(row["options"])
        if not valid or row["rejected"]:
            bad.append(row["id"])
    return bad


def probabilities(logits, row, temperatures):
    return decide(logits, row["kind"], len(row["options"]), temperatures[row["kind"]])[0]


def pearson(first, second):
    if len(first) < 2 or len(set(first)) == 1 or len(set(second)) == 1:
        return None
    a, b = mean(first), mean(second)
    covariance = math.fsum((x - a) * (y - b) for x, y in zip(first, second, strict=True))
    return covariance / math.sqrt(math.fsum((x - a) ** 2 for x in first) * math.fsum((y - b) ** 2 for y in second))


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for index in order[start:end + 1]:
            result[index] = (start + end) / 2 + 1
        start = end + 1
    return result


def spearman(first, second):
    return pearson(ranks(first), ranks(second))


def answer(row, values):
    position = top(values)
    return row["order"][position] if row["order"] else position


def metrics(items, temperatures):
    groups = {}
    for row, logits in items:
        groups.setdefault(row["group"], []).append((row, probabilities(logits, row, temperatures)))
    result = {}
    for group, pairs in groups.items():
        count = len(pairs)
        if group == NOUL:
            scores, labels = [p[1] for _, p in pairs], [row["label"] for row, _ in pairs]
            hits = sum((value > 0.5) == (label == 1) for value, label in zip(scores, labels, strict=True))
            result[group] = {"rows": count, "hits": hits, "accuracy": hits / count,
                             "auroc": auroc(scores, [label == 1 for label in labels]), "positives": sum(labels),
                             "brier_hard": mean([(value - label) ** 2 for value, label in zip(scores, labels,
                                                                                               strict=True)])}
        elif group == "JSTS":
            expected, labels = [expectation(p) for _, p in pairs], [row["label"] for row, _ in pairs]
            result[group] = {"rows": count, "mae": mean([abs(value - label) for value, label in zip(expected, labels,
                                                                                                    strict=True)]),
                             "pearson": pearson(expected, labels), "spearman": spearman(expected, labels),
                             "expected_mean": mean(expected), "label_mean": mean(labels)}
        else:
            answers = [(row, answer(row, p)) for row, p in pairs]
            hits = sum(value == row["original_label"] for row, value in answers)
            result[group] = {"rows": count, "hits": hits, "accuracy": hits / count}
            if group == "JNLI":
                matrix = [[0] * len(NLI_LABELS) for _ in NLI_LABELS]
                for row, value in answers:
                    matrix[row["original_label"]][value] += 1
                result[group]["confusion"] = {"rows": "label", "columns": "prediction", "order": list(NLI_LABELS),
                                              "matrix": matrix}
    return result


def by_variant(rows, logits, temperatures):
    return {variant: metrics([(row, values) for row, values in zip(rows, logits, strict=True)
                              if row["variant"] == variant and row["permutation"] == "identity"], temperatures)
            for variant in VARIANTS}


def adopt(results):
    adopted = {}
    for task in TASKS:
        ja, en = results["ja"][task], results["en"][task]
        adopted[task] = "en" if (en["mae"] < ja["mae"] if task == "JSTS" else en["hits"] > ja["hits"]) else "ja"
    return {group: adopted[group.split(":")[0]] for group in GROUPS}


def order_sensitivity(rows, logits, temperatures, variant):
    answers, result = {}, {}
    for name in PERMUTATIONS:
        hits = count = 0
        for row, values in zip(rows, logits, strict=True):
            if row["group"] == "JCommonsenseQA" and row["variant"] == variant and row["permutation"] == name:
                found = answer(row, probabilities(values, row, temperatures))
                answers.setdefault(row["source_id"], {})[name] = found
                hits, count = hits + (found == row["original_label"]), count + 1
        result[name] = {"order": list(PERMUTATIONS[name]), "rows": count, "hits": hits,
                        "accuracy": hits / count if count else None}
    complete = [found for found in answers.values() if set(found) == set(PERMUTATIONS)]
    consistent = sum(len(set(found.values())) == 1 for found in complete)
    return {"variant": variant, "orders": result, "rows": len(answers), "complete_rows": len(complete),
            "consistent_rows": consistent, "consistent_fraction": consistent / len(complete) if complete else None}


def differences(stages):
    return {label: {variant: {group: {field: None if item["1m"][variant][group][field] is None
                                      or item["3"][variant][group][field] is None
                                      else item["3"][variant][group][field] - item["1m"][variant][group][field]
                                      for field in FIELDS[group]} for group in GROUPS} for variant in VARIANTS}
            for label, item in stages.items()}


def agreement(rows, first, second, temperatures):
    result = {}
    for variant in VARIANTS:
        result[variant] = {}
        for group in GROUPS:
            pairs = [(probabilities(a, row, temperatures), probabilities(b, row, temperatures))
                     for row, a, b in zip(rows, first, second, strict=True)
                     if row["variant"] == variant and row["group"] == group]
            errors = [largest(a, b) for a, b in pairs]
            result[variant][group] = {"rows": len(pairs), "same_top": sum(top(a) == top(b) for a, b in pairs),
                                      "max_probability_difference": max(errors),
                                      "mean_probability_difference": mean(errors)}
    return result


def elapsed(answers):
    seconds = [row["elapsed_seconds"] for row in answers if "error" not in row]
    return {"rows": len(seconds), "total_s": math.fsum(seconds), "median_s": statistics.median(seconds)}


def clean(problems):
    return not any(problems.values())


def adoption_document(args):
    rows, sources = set_rows(args, "Jdev", distributed_tokenizer(args.snapshot))
    swift, bundle = read_json(args.swift), bundle_identity(args.bundle)
    problems = {"manifest": manifest_problems(rows, read_json(args.rows)), "labels": label_problems(rows)}
    problems["3"] = swift_problems(rows, swift, bundle) if clean(problems) else []
    document = {"schema": ADOPTION_SCHEMA, "frozen": False, "problems": problems}
    if not clean(problems):
        return document
    logits = [item["logits"] for item in swift["rows"]]
    results = {"1": by_variant(rows, logits, ONE), "adopted": by_variant(rows, logits, bundle["temperatures"])}
    document.update(frozen=True, adopted=adopt(results["1"]), rule=ADOPTION_RULE, selection_temperature=1.0,
                    results=results, wrappers=load_wrappers(args.wrappers), wrappers_sha256=sha256(args.wrappers),
                    rows_sha256=sha256(args.rows), input_sha256=swift.get("input_sha256"), bundle=bundle,
                    rows={"total": len(rows), "by_variant": dict(Counter(row["variant"] for row in rows))},
                    timing=elapsed(swift["rows"]), **sources)
    return document


def score_document(args):
    adoption = frozen_adoption(args.adoption, sha256(args.wrappers))
    rows, sources = set_rows(args, "J", distributed_tokenizer(args.snapshot))
    swift, reference, bundle = read_json(args.swift), read_json(args.reference), bundle_identity(args.bundle)
    plain = [row for row in rows if row["permutation"] == "identity"]
    problems = {"manifest": manifest_problems(rows, read_json(args.rows)), "labels": label_problems(rows),
                "adoption": [] if adoption["bundle"]["version"] == bundle["version"] else
                ["the prompt adoption used another bundle"]}
    problems["3"] = swift_problems(rows, swift, bundle) if clean(problems) else []
    problems["1m"] = reference_problems(plain, reference) if clean(problems) else []
    result = {"schema": SCHEMA, "recorded": False, "problems": problems}
    if not clean(problems):
        return result
    third = {item["id"]: item["logits"] for item in swift["rows"]}
    logits = {"1m": [item["logits1m"] for item in reference["rows"]], "3": [third[row["id"]] for row in plain]}
    temperatures = {"1": ONE, "adopted": bundle["temperatures"]}
    stages = {label: {stage: by_variant(plain, logits[stage], chosen) for stage in logits}
              for label, chosen in temperatures.items()}
    variant = adoption["adopted"]["JCommonsenseQA"]
    sensitivity = {label: order_sensitivity(rows, [item["logits"] for item in swift["rows"]], chosen, variant)
                   for label, chosen in temperatures.items()}
    problems["permutations"] = [] if all(item["complete_rows"] == item["rows"] == 200
                                         for item in sensitivity.values()) else ["permuted rows are incomplete"]
    if not clean(problems):
        return result
    result.update(recorded=True, definitions=DEFINITIONS, temperatures=temperatures,
                  adoption={"adopted": adoption["adopted"], "rule": adoption["rule"],
                            "selection_temperature": adoption["selection_temperature"], "jdev": adoption["results"],
                            "sha256": sha256(args.adoption), "jdev_sha256": adoption["jdev_sha256"]},
                  wrappers=load_wrappers(args.wrappers), wrappers_sha256=sha256(args.wrappers), stages=stages,
                  differences=differences(stages), permutations=sensitivity,
                  agreement={label: agreement(plain, logits["1m"], logits["3"], chosen)
                             for label, chosen in temperatures.items()},
                  rows={"total": len(rows), "stage_1m": len(plain),
                        "by_group": dict(Counter("/".join((row["variant"], row["group"], row["permutation"]))
                                                 for row in rows))},
                  inputs={"rows_sha256": sha256(args.rows), "input_sha256": swift.get("input_sha256"),
                          "tokens": sum(row["token_count"] for row in rows),
                          "longest": max(row["token_count"] for row in rows), **sources},
                  timing={"stage3": elapsed(swift["rows"]), "stage_1m": reference.get("timing")},
                  bundle=bundle, reference={key: reference.get(key) for key in ("device", "dtype", "versions", "memory",
                                                                                 "merged_revision")})
    return result


def number(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def signed(value):
    return "-" if value is None else f"{value:+.4f}"


def adoption_lines(results, adopted):
    lines = ["| Task | Metric (temperature 1) | ja | en | Adopted |", "|---|---|---:|---:|---|"]
    for task in TASKS:
        ja, en = results["ja"][task], results["en"][task]
        if task == "JSTS":
            lines.append(f"| {task} | MAE of E[p] | {number(ja['mae'])} | {number(en['mae'])} | {adopted[task]} |")
        else:
            lines.append(f"| {NAMES[task]} | accuracy | {number(ja['accuracy'])} "
                         f"({ja['hits']}/{ja['rows']}) | {number(en['accuracy'])} ({en['hits']}/{en['rows']}) | "
                         f"{adopted[task]} |")
    lines.append(f"| JNLI noul | follows JNLI choice | - | - | {adopted[NOUL]} |")
    return lines


def adoption_report(document):
    lines = ["# JEV-27B prompt adoption on Jdev", "", f"- Frozen: {str(document['frozen']).lower()}."]
    if not document["frozen"]:
        return "\n".join(lines + ["", "## Input problems", "",
                                  *(f"- {name}: {items}" for name, items in document["problems"].items() if items)]) + "\n"
    counts = ", ".join(f"{count} {variant}" for variant, count in document["rows"]["by_variant"].items())
    lines += [f"- Rule: {document['rule']}", f"- Rows: {document['rows']['total']} ({counts}).",
              f"- Wrappers SHA-256 `{document['wrappers_sha256']}`, Jdev SHA-256 `{document['jdev_sha256']}`.", "",
              *adoption_lines(document["results"]["1"], document["adopted"])]
    return "\n".join(lines) + "\n"


def result_lines(item, delta, adopted):
    lines = ["| Task | Metric | Stage 1' ja | Stage 1' en | Stage 3 ja | Stage 3 en | 3 - 1' ja | 3 - 1' en | Adopted |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for group in GROUPS:
        for field in FIELDS[group]:
            values = [item[stage][variant][group][field] for stage in ("1m", "3") for variant in VARIANTS]
            lines.append(f"| {NAMES[group]} | {LABELS[field]} | " + " | ".join(number(value) for value in values)
                         + f" | {signed(delta['ja'][group][field])} | {signed(delta['en'][group][field])} | "
                         f"{adopted[group]} |")
    return lines


def report(result):
    lines = ["# JEV-27B Japanese (JGLUE) on J", "", f"- Recorded: {str(result['recorded']).lower()}."]
    if not result["recorded"]:
        return "\n".join(lines + ["", "## Input problems", "",
                                  *(f"- {name}: {items}" for name, items in result["problems"].items() if items)]) + "\n"
    adoption, timing = result["adoption"], result["timing"]
    lines += [f"- {DEFINITIONS['scope']}.",
              f"- Rows: {result['rows']['total']} at stage 3, {result['rows']['stage_1m']} at stage 1'. Stage 3 ran the "
              f"calibrated bundle `{result['bundle']['name']}` (decision.json SHA-256 `{result['bundle']['version']}`).",
              f"- Adopted temperatures: noul {result['temperatures']['adopted']['noul']}, choice "
              f"{result['temperatures']['adopted']['choice']}, score {result['temperatures']['adopted']['score']}.", "",
              "## Prompt adoption on Jdev (frozen before J)", "", f"- {adoption['rule']}", "",
              *adoption_lines(adoption["jdev"]["1"], adoption["adopted"])]
    for label, title in (("1", "temperature 1"), ("adopted", "the adopted temperatures")):
        lines += ["", f"## J at {title}", "",
                  *result_lines(result["stages"][label], result["differences"][label], adoption["adopted"])]
    sensitivity = result["permutations"]["1"]
    lines += ["", f"## JCommonsenseQA option order (stage 3, {sensitivity['variant']})", "",
              "| Order | Position of each original option | Accuracy |", "|---|---|---:|",
              *(f"| {name} | {' '.join(str(index) for index in item['order'])} | {number(item['accuracy'])} "
                f"({item['hits']}/{item['rows']}) |" for name, item in sensitivity["orders"].items()), "",
              f"- The four answers, mapped back to the original options, agree on {sensitivity['consistent_rows']} of "
              f"{sensitivity['complete_rows']} rows ({number(sensitivity['consistent_fraction'])}). The answers do not "
              "depend on a positive temperature; the adopted temperatures give the same counts "
              f"({result['permutations']['adopted']['consistent_rows']} rows).", "",
              "## JNLI choice confusion at temperature 1 (rows: label, columns: prediction; entailment, contradiction, "
              "neutral)", "", "| Stage | Variant | Matrix |", "|---|---|---|"]
    for stage in ("1m", "3"):
        for variant in VARIANTS:
            matrix = result["stages"]["1"][stage][variant]["JNLI"]["confusion"]["matrix"]
            lines.append(f"| {STAGE_NAMES[stage]} | {variant} | "
                         + " / ".join(" ".join(str(value) for value in row) for row in matrix) + " |")
    lines += ["", "## Stage 3 against stage 1' per row at temperature 1", "",
              "| Variant | Task | Rows | Same top | Largest probability difference | Mean |", "|---|---|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        for group in GROUPS:
            item = result["agreement"]["1"][variant][group]
            lines.append(f"| {variant} | {NAMES[group]} | {item['rows']} | {item['same_top']} | "
                         f"{number(item['max_probability_difference'], 6)} | {number(item['mean_probability_difference'], 6)} |")
    lines += ["", "## Time", "",
              f"- Stage 3: {timing['stage3']['rows']} rows, {timing['stage3']['total_s']:.1f} s in total, median "
              f"{timing['stage3']['median_s']:.3f} s per row. Stage 1' (MPS bfloat16): "
              + ", ".join(f"{key} {value}" for key, value in sorted((timing["stage_1m"] or {}).items())) + "."]
    return "\n".join(lines) + "\n"


def write_document(path, document, text):
    if path.exists():
        raise FileExistsError(f"{path.name} must be written to a new directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "report.md").write_text(text, encoding="utf-8")
    write_json(path, document)


def arguments():
    parser = argparse.ArgumentParser(description="Jdev, wrapped inputs, prompt adoption and scoring of the JGLUE rows "
                                                 "of JEV-27B.")
    commands = parser.add_subparsers(dest="command", required=True)
    paths = {"jdev": ("model", "jglue", "fixtures", "output"),
             "input": ("model", "snapshot", "jglue", "fixtures", "wrappers", "output"),
             "adopt": ("model", "snapshot", "jglue", "fixtures", "wrappers", "jdev", "rows", "swift", "bundle", "output"),
             "score": ("model", "snapshot", "jglue", "fixtures", "wrappers", "adoption", "rows", "swift", "reference",
                       "bundle", "output")}
    for name, options in paths.items():
        command = commands.add_parser(name)
        for option in options:
            command.add_argument(f"--{option}", type=Path, required=True)
    commands.choices["input"].add_argument("--set", choices=("Jdev", "J"), required=True)
    for option in ("jdev", "adoption"):
        commands.choices["input"].add_argument(f"--{option}", type=Path)
    args = parser.parse_args()
    if args.command == "input" and ((args.set == "Jdev") != (args.jdev is not None)
                                    or (args.set == "J") != (args.adoption is not None)):
        parser.error("--set Jdev takes --jdev and --set J takes --adoption")
    return args


def main():
    args = arguments()
    if args.command == "jdev":
        return write_jdev(args)
    if args.command == "input":
        return write_inputs(args)
    if args.command == "adopt":
        document = adoption_document(args)
        write_document(args.output / "prompt-adoption.json", document, adoption_report(document))
        return 0 if document["frozen"] else 1
    result = score_document(args)
    write_document(args.output / "japanese.json", result, report(result))
    return 0 if result["recorded"] else 1


if __name__ == "__main__":
    sys.exit(main())
