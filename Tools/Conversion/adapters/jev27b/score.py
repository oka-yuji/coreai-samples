import argparse
from fractions import Fraction
import math
from pathlib import Path
import random
import statistics
import sys

from ...storage import read_json, write_json
from .compare import completeness, finite as finite_swift
from .fixtures import KINDS, load_rows
from .prepare import check_integrity, load_contract
from .typed_head import slot_range, softmax
from .verify_merge import top
from .verify_quantization import complete, expected_rows, finite, same_inputs

SCHEMA = "typed-decision-quality/1"
STAGES = ("0", "0q", "1m", "2", "3")
NAMES = {"0": "0", "0q": "0q", "1m": "1'", "2": "2", "3": "3"}
REPLICATES = 2000
SEED = 20261102
QUANTILES = (0.025, 0.975)
BINS = 15
TOLERANCE = 1e-3
SHARE = Fraction(4, 5)
RELATIVE = {
    "choice_accuracy": {"kind": "choice", "field": "accuracy", "unit": "points", "sense": ">=", "limit": -2},
    "noul_brier": {"kind": "noul", "field": "brier_soft", "unit": "Brier", "sense": "<=", "limit": 0.01},
    "score_mae_normalized": {"kind": "score", "field": "mae_normalized", "unit": "MAE / 5", "sense": "<=", "limit": 0.02},
}
GAINS = {
    "choice_accuracy": {"kind": "choice", "field": "accuracy", "unit": "points", "higher_is_better": True},
    "noul_brier": {"kind": "noul", "field": "brier_soft", "unit": "Brier", "higher_is_better": False},
    "score_mae_normalized": {"kind": "score", "field": "mae_normalized", "unit": "MAE / 5", "higher_is_better": False},
    "kl": {"kind": "all", "field": "kl", "unit": "nats", "higher_is_better": False},
}
CARD = {
    "source": "reports/eval_27b_bundle.md of autotrust/JEV-27B at the card temperatures",
    "all_rows": {"rows": 29955, "kl": 0.0185, "top1": 0.9220, "ece": 0.0011, "noul_auroc": 0.9961,
                 "noul_brier_soft": 0.0013, "noul_brier_hard": 0.0741, "score_mae_expected": 0.0976,
                 "choice_top1": 0.9041, "choice_kl": 0.0364},
    "without_placeholder": {"rows": 27695, "kl": 0.0201, "noul_brier_soft": 0.0016},
    "acceptance": {"noul_auroc": ">= 0.95", "noul_brier": "<= 0.1", "choice_top1": ">= 0.9", "score_mae": "<= 0.35",
                   "kl": "<= 0.15", "ece": "<= 0.03"},
}


def mean(values):
    return math.fsum(values) / len(values)


def teacher(target, count):
    if not isinstance(target, list) or len(target) != count \
            or not all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in target):
        return None
    total = math.fsum(target)
    if abs(total - 1) > TOLERANCE:
        return None
    return [value / total for value in target], total - 1


def log_softmax(values):
    largest = max(values)
    shifted = [value - largest for value in values]
    total = math.log(math.fsum(math.exp(value) for value in shifted))
    return [value - total for value in shifted]


def scaled(logits24, kind, count, temperature):
    start, end = slot_range(kind, count)
    return [float(value) / temperature for value in logits24[start:end]]


def expectation(values):
    return math.fsum(index * value for index, value in enumerate(values))


def kl(target, logp):
    return math.fsum(value * (math.log(value) - log) for value, log in zip(target, logp, strict=True) if value > 0)


def cross_entropy(target, logp):
    return -math.fsum(value * log for value, log in zip(target, logp, strict=True) if value > 0)


def brier(probabilities, target):
    return math.fsum((first - second) ** 2 for first, second in zip(probabilities, target, strict=True))


def judge(logits24, kind, target, temperature):
    values = scaled(logits24, kind, len(target), temperature)
    probabilities, logp = softmax(values), log_softmax(values)
    predicted, teacher_top = top(probabilities), top(target)
    row = {"kind": kind, "hit": int(predicted == teacher_top), "kl": kl(target, logp), "nll": cross_entropy(target, logp),
           "brier_multiclass": brier(probabilities, target), "confidence": probabilities[predicted],
           "teacher_probability": target[predicted]}
    if kind == "noul":
        row.update(p_true=probabilities[1], positive=teacher_top == 1, tie=target[0] == target[1],
                   brier_soft=(probabilities[1] - target[1]) ** 2,
                   brier_hard=(probabilities[1] - (teacher_top == 1)) ** 2)
    elif kind == "score":
        error = abs(expectation(probabilities) - expectation(target))
        row.update(mae=error, mae_normalized=error / 5)
    else:
        row["accuracy"] = 100 * row["hit"]
    return row


def auroc(scores, labels):
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return None
    wins = math.fsum((first > second) + 0.5 * (first == second) for first in positives for second in negatives)
    return wins / (len(positives) * len(negatives))


def ece(rows, bins=BINS):
    groups = [[] for _ in range(bins)]
    for row in rows:
        groups[min(int(row["confidence"] * bins), bins - 1)].append(row)
    gaps = {"soft": [], "hard": []}
    for group in groups:
        if group:
            confidence = mean([row["confidence"] for row in group])
            for name, field in (("soft", "teacher_probability"), ("hard", "hit")):
                gaps[name].append(len(group) / len(rows) * abs(confidence - mean([row[field] for row in group])))
    return {name: math.fsum(values) for name, values in gaps.items()}


def summary(values):
    result = {"all": {"rows": len(values), "kl": mean([value["kl"] for value in values]),
                      "top1": mean([value["hit"] for value in values])}}
    for kind in KINDS:
        chosen = [value for value in values if value["kind"] == kind]
        if not chosen:
            continue
        hits = sum(value["hit"] for value in chosen)
        item = {"rows": len(chosen), "hits": hits, "top1": hits / len(chosen), "kl": mean([value["kl"] for value in chosen])}
        if kind == "choice":
            item["accuracy"] = 100 * hits / len(chosen)
        elif kind == "noul":
            ranked = [value for value in chosen if not value["tie"]]
            item.update(brier_soft=mean([value["brier_soft"] for value in chosen]),
                        brier_hard=mean([value["brier_hard"] for value in chosen]),
                        auroc=auroc([value["p_true"] for value in ranked], [value["positive"] for value in ranked]),
                        auroc_rows=len(ranked), auroc_excluded_ties=len(chosen) - len(ranked))
        else:
            item.update(mae=mean([value["mae"] for value in chosen]),
                        mae_normalized=mean([value["mae_normalized"] for value in chosen]))
        result[kind] = item
    return result


def units(rows):
    groups = {kind: {} for kind in KINDS}
    for index, row in enumerate(rows):
        groups[row["kind"]].setdefault(row["state"], []).append(index)
    return {kind: list(found.values()) for kind, found in groups.items()}


def resamples(groups, replicates=REPLICATES, seed=SEED):
    rng = random.Random(seed)
    return [{kind: [rng.randrange(len(found)) for _ in found] for kind, found in groups.items()}
            for _ in range(replicates)]


def resampled(found, draw):
    return [index for unit in draw for index in found[unit]]


def percentile(values, quantile):
    if not values:
        return None
    position = (len(values) - 1) * quantile
    low = math.floor(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (position - low) * (values[high] - values[low])


def bounds(samples):
    valid = sorted(sample for sample in samples if sample is not None)
    return {"low": percentile(valid, QUANTILES[0]), "high": percentile(valid, QUANTILES[1]), "valid": len(valid),
            "invalid": len(samples) - len(valid)}


def relative(values, groups, draws):
    result = {}
    for name, rule in RELATIVE.items():
        kind, field = rule["kind"], rule["field"]
        differences = {index: adapted[field] - merged[field]
                       for index, (adapted, merged) in enumerate(zip(values["3"], values["1m"], strict=True))
                       if adapted["kind"] == kind}
        point = mean(list(differences.values()))
        spread = bounds([mean([differences[index] for index in resampled(groups[kind], draw[kind])]) for draw in draws])
        result[name] = {**rule, "rows": len(differences),
                        "stage_1m": mean([value[field] for value in values["1m"] if value["kind"] == kind]),
                        "stage_3": mean([value[field] for value in values["3"] if value["kind"] == kind]),
                        "difference": point, "interval": spread,
                        "interval_crosses_limit": spread["low"] <= rule["limit"] <= spread["high"],
                        "passed": point >= rule["limit"] if rule["sense"] == ">=" else point <= rule["limit"]}
    return result


def metric(result, kind, field):
    item = result[kind]
    return Fraction(100 * item["hits"], item["rows"]) if field == "accuracy" else item[field]


def direction(higher_is_better, adapted, base):
    return adapted - base if higher_is_better else base - adapted


def gain_condition(values, higher_is_better):
    bf16 = direction(higher_is_better, values["1m"], values["0"])
    quantized = direction(higher_is_better, values["3"], values["0q"])
    judged = quantized - SHARE * bf16
    status = "undefined" if bf16 <= 0 else "pass" if judged >= 0 else "fail"
    reasons = {"undefined": "gain(bf16) <= 0, so the share of the adaptation kept at 4 bits is undefined",
               "pass": "gain(4bit) - 0.8 gain(bf16) >= 0", "fail": "gain(4bit) - 0.8 gain(bf16) < 0"}
    return {"gain_bf16": float(bf16), "gain_4bit": float(quantized), "judged": float(judged),
            "share": float(quantized / bf16) if bf16 > 0 else None, "status": status, "reason": reasons[status],
            "gain_pytorch_4bit": float(direction(higher_is_better, values["2"], values["0q"])),
            "stage3_against_stage2": float(direction(higher_is_better, values["3"], values["2"]))}


def gains(summaries):
    result = {}
    for name, rule in GAINS.items():
        values = {stage: metric(summaries[stage], rule["kind"], rule["field"]) for stage in STAGES}
        result[name] = {**rule, "values": {stage: float(value) for stage, value in values.items()},
                        **gain_condition(values, rule["higher_is_better"])}
    return result


def elapsed(rows):
    answered = [row for row in rows if "error" not in row]
    seconds = math.fsum(row["elapsed_seconds"] for row in answered)
    return {"rows": len(answered), "total_s": seconds, "median_s": statistics.median(row["elapsed_seconds"] for row in answered),
            "tokens": sum(row["token_count"] for row in answered),
            "tokens_per_second": sum(row["token_count"] for row in answered) / seconds}


def check_inputs(reference, peft, merged, swift, expected, restored):
    expected = [row for row in expected if row["set"] == "Q"]
    rows = [row for row in reference["rows"] if row["set"] == "Q"]
    sources = {"0": peft["rows"], "0q": rows, "1m": merged["rows"], "2": rows}
    problems = {stage: complete(source, expected, stage) for stage, source in sources.items()}
    problems["3"] = completeness(rows, swift)
    problems["references"] = [name for name, document in (("reference-Q", peft), ("reference-merged-Q", merged))
                              if document.get("passed") is not True]
    problems["targets"] = []
    targets = []
    for entry, row in zip(expected, restored, strict=True):
        found = teacher(row.get("target"), entry["n_options"]) if row["id"] == entry["id"] else None
        if found is None:
            problems["targets"].append(entry["id"])
        targets.append(found)
    if not any(problems.values()):
        problems["inputs"] = same_inputs(rows, peft["rows"]) + same_inputs(rows, merged["rows"])
        problems["nonfinite"] = sorted({identifier for stage, source in sources.items()
                                        for identifier in finite(source, stage)} | set(finite_swift(swift["rows"])))
    if any(problems.values()):
        return problems, None, None
    scored = []
    for index, (entry, row, (target, _)) in enumerate(zip(expected, restored, targets, strict=True)):
        if not entry["rejected"]:
            scored.append({"id": entry["id"], "kind": entry["kind"], "state": row["state"], "target": target,
                           "logits": {"0": peft["rows"][index]["logits0"], "0q": rows[index]["logits0q"],
                                      "1m": merged["rows"][index]["logits1m"], "2": rows[index]["logits2"],
                                      "3": swift["rows"][index]["logits"]}})
    deviations = [deviation for _, deviation in targets]
    normalization = {"rows": len(targets), "normalized": sum(deviation != 0 for deviation in deviations),
                     "largest_deviation": max(abs(deviation) for deviation in deviations), "tolerance": TOLERANCE,
                     "rejected": sum(entry["rejected"] for entry in expected)}
    return problems, scored, normalization


def score(reference, peft, merged, swift, expected, restored):
    problems, rows, normalization = check_inputs(reference, peft, merged, swift, expected, restored)
    result = {"schema": SCHEMA, "recorded": rows is not None, "passed": False, "problems": problems,
              "definitions": "agreement with the teacher distribution, not calibration against real-world correctness"}
    if rows is None:
        return result
    chosen = {"1": dict.fromkeys(KINDS, 1.0), "card": reference["temperatures"]}
    values = {label: {stage: [judge(row["logits"][stage], row["kind"], row["target"], temperatures[row["kind"]])
                              for row in rows] for stage in STAGES} for label, temperatures in chosen.items()}
    summaries = {label: {stage: summary(items) for stage, items in stages.items()} for label, stages in values.items()}
    groups = units(rows)
    conditions = {"D1-1": relative(values["1"], groups, resamples(groups)), "D1-2": gains(summaries["1"])}
    passed = {"D1-1": all(item["passed"] for item in conditions["D1-1"].values()),
              "D1-2": all(item["status"] == "pass" for item in conditions["D1-2"].values())}
    result.update(passed=all(passed.values()), conditions=conditions, conditions_passed=passed,
                  temperatures=chosen, stages=summaries, targets=normalization, card=CARD,
                  bootstrap={"replicates": REPLICATES, "seed": SEED, "generator": "random.Random(seed).randrange",
                             "unit": "rows of one kind that share a state", "quantiles": list(QUANTILES),
                             "interpolation": "linear (type 7)", "paired": "the same draws for every stage",
                             "units": {kind: len(found) for kind, found in groups.items()}},
                  rows={"scored": len(rows), "by_kind": {kind: sum(row["kind"] == kind for row in rows) for kind in KINDS}},
                  timing={"stage3": elapsed(swift["rows"]), "stages_0_1": peft.get("timing"),
                          "stage_1m": merged.get("timing")},
                  bundle=swift.get("bundle"))
    return result


def number(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def flag(value):
    return str(value).lower()


def stage_lines(summaries):
    lines = ["| Stage | KL all | KL noul | KL choice | KL score | Top-1 all | Choice accuracy | noul AUROC | "
             "noul Brier soft | noul Brier hard | score MAE | score MAE / 5 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for stage in STAGES:
        item = summaries[stage]
        lines.append(f"| {NAMES[stage]} | {number(item['all']['kl'])} | {number(item['noul']['kl'])} | "
                     f"{number(item['choice']['kl'])} | {number(item['score']['kl'])} | {number(item['all']['top1'])} | "
                     f"{number(item['choice']['accuracy'], 2)} | {number(item['noul']['auroc'])} | "
                     f"{number(item['noul']['brier_soft'])} | {number(item['noul']['brier_hard'])} | "
                     f"{number(item['score']['mae'])} | {number(item['score']['mae_normalized'])} |")
    return lines


def report(result):
    lines = ["# JEV-27B quality on Q", "",
             f"- Recorded: {flag(result['recorded'])}. Quality passed: {flag(result['passed'])}.",
             "- Every number measures agreement with the teacher distribution of the corpus (outputs of Jev 1.13), "
             "not calibration against real-world correctness.", ""]
    if not result["recorded"]:
        lines += ["## Input problems", "", *(f"- {name}: {items}" for name, items in result["problems"].items() if items)]
        return "\n".join(lines) + "\n"
    targets, bootstrap, passed = result["targets"], result["bootstrap"], result["conditions_passed"]
    lines += [f"- Rows scored {result['rows']['scored']} ({result['rows']['by_kind']}), rejected {targets['rejected']}. "
              f"Targets normalized to sum 1: {targets['normalized']} rows, largest |sum - 1| "
              f"{targets['largest_deviation']:.1e}.",
              f"- Probabilities are recomputed from the 24 logits of every stage. Temperature 1 decides D1-1 and D1-2; "
              f"the card temperatures are {result['temperatures']['card']}.",
              f"- Bootstrap: {bootstrap['replicates']} paired replicates, seed {bootstrap['seed']}, units of "
              f"{bootstrap['unit']} ({bootstrap['units']}), 95% percentile interval with linear interpolation.", "",
              f"## D1-1: stage 3 against stage 1' at temperature 1 (passed {flag(passed['D1-1'])})", "",
              "| Condition | Rows | Stage 1' | Stage 3 | Difference | 95% interval | Limit | Interval crosses the limit "
              "| Passed |", "|---|---:|---:|---:|---:|---|---|---|---|"]
    for name, item in result["conditions"]["D1-1"].items():
        spread = item["interval"]
        lines.append(f"| {name} ({item['unit']}) | {item['rows']} | {number(item['stage_1m'])} | {number(item['stage_3'])} | "
                     f"{item['difference']:+.4f} | [{spread['low']:+.4f}, {spread['high']:+.4f}] | "
                     f"{item['sense']} {item['limit']} | {flag(item['interval_crosses_limit'])} | {flag(item['passed'])} |")
    lines += ["", f"## D1-2: adaptation gain at temperature 1 (passed {flag(passed['D1-2'])})", "",
              "Accuracy gains are adapted minus base; KL, Brier and MAE gains are base minus adapted. "
              "gain(bf16) compares stage 1' with stage 0, gain(4bit) compares stage 3 with stage 0q.", "",
              "| Metric | Unit | 0 | 1' | 0q | 3 | gain(bf16) | gain(4bit) | gain(4bit) - 0.8 gain(bf16) | Share | "
              "Status |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, item in result["conditions"]["D1-2"].items():
        values = item["values"]
        lines.append(f"| {name} | {item['unit']} | {number(values['0'])} | {number(values['1m'])} | {number(values['0q'])} | "
                     f"{number(values['3'])} | {item['gain_bf16']:+.4f} | {item['gain_4bit']:+.4f} | {item['judged']:+.4f} | "
                     f"{number(item['share'], 3)} | {item['status']} ({item['reason']}) |")
    lines += ["", "| Metric | gain of stage 2 over 0q (PyTorch 4-bit) | stage 3 against stage 2 |", "|---|---:|---:|",
              *(f"| {name} | {item['gain_pytorch_4bit']:+.4f} | {item['stage3_against_stage2']:+.4f} |"
                for name, item in result["conditions"]["D1-2"].items()), "",
              "## D1-3: stages at temperature 1", "", *stage_lines(result["stages"]["1"]), "",
              "## D1-3: stages at the card temperatures", "", *stage_lines(result["stages"]["card"]), "",
              "AUROC ranks P(true) of the noul rows whose teacher top is true against the others; rows with a 0.5 target "
              f"are left out ({result['stages']['1']['3']['noul']['auroc_excluded_ties']} rows) and tied predictions count "
              "one half. The teacher top is the first largest target value.", "",
              "## Card values (reference only, not a criterion)", "",
              f"- {CARD['source']}: all {CARD['all_rows']['rows']} rows of test_set_30k, including the placeholder rows "
              f"that the fixed sets exclude, so the population and temperatures differ from Q: {CARD['all_rows']}.",
              f"- Without the placeholder rows ({CARD['without_placeholder']['rows']} rows): "
              f"{CARD['without_placeholder']}. The card's acceptance lines: {CARD['acceptance']}.", ""]
    timing = result["timing"]["stage3"]
    lines += ["## Time", "",
              f"- Stage 3: {timing['rows']} rows, {timing['total_s']:.1f} s in total, median {timing['median_s']:.3f} s per "
              f"row, {timing['tokens_per_second']:.1f} tokens/s over all rows.",
              f"- Stages 0 and 1: {result['timing']['stages_0_1']}. Stage 1': {result['timing']['stage_1m']}."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Score stages 0, 0q, 1', 2 and 3 of JEV-27B on Q against the teacher.")
    for name in ("model", "reference", "reference-q", "reference-merged-q", "swift", "fixtures", "corpus", "jglue",
                 "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "quality.json").exists():
        raise FileExistsError("quality.json must be written to a new directory")
    contract = load_contract(args.model)
    check_integrity(args.corpus, contract["dataset"]["id"], contract["dataset"]["revision"])
    result = score(read_json(args.reference), read_json(args.reference_q), read_json(args.reference_merged_q),
                   read_json(args.swift), expected_rows(args.fixtures, contract["limits"]["input_tokens"]),
                   load_rows(args.fixtures / "Q.json", args.corpus, args.jglue))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.md").write_text(report(result), encoding="utf-8")
    write_json(args.output / "quality.json", result)
    return 0 if result["recorded"] else 1


if __name__ == "__main__":
    sys.exit(main())
