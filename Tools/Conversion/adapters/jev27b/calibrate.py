import argparse
from collections import Counter
import math
from pathlib import Path
import struct
import subprocess
import sys

from ...storage import read_json, write_json
from .compare import compare, completeness, finite, report as fidelity_report, temperatures
from .decision import DECISION, SUMS, listing, sha256, validate, write_sums
from .fixtures import KINDS, load_rows
from .prepare import check_integrity, distributed_tokenizer, load_contract
from .score import (BINS, QUANTILES, REPLICATES, SEED, bounds, cross_entropy, ece, elapsed, judge, log_softmax, mean,
                    resampled, resamples, teacher, units)
from .typed_head import render, slot_range, tokenize

SCHEMA = "typed-decision-calibration/1"
REFIT_SCHEMA = "typed-decision-calibration-refit/1"
REFIT = "calibration-refit.json"
LOW, HIGH, STEP = 0.25, 4.0, 1e-7
GOLDEN = (math.sqrt(5) - 1) / 2
SHARED_STATES = 38
SETS = ("Cal", "H'", "H")


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def shortest(value):
    rounded = float32(value)
    return next(candidate for digits in range(1, 10)
                if float32(candidate := float(f"{rounded:.{digits}g}")) == rounded)


def selected(row):
    start, end = slot_range(row["kind"], len(row["target"]))
    return [float(value) for value in row["logits"][start:end]]


def objective(rows, beta):
    return mean([cross_entropy(row["target"], log_softmax([beta * value for value in selected(row)])) for row in rows])


def fit(rows, low=LOW, high=HIGH, step=STEP):
    first, last = low, high
    left, right = last - GOLDEN * (last - first), first + GOLDEN * (last - first)
    at_left, at_right = objective(rows, left), objective(rows, right)
    iterations = 0
    while last - first >= step:
        iterations += 1
        if at_left < at_right:
            last, right, at_right = right, left, at_left
            left = last - GOLDEN * (last - first)
            at_left = objective(rows, left)
        else:
            first, left, at_left = left, right, at_right
            right = first + GOLDEN * (last - first)
            at_right = objective(rows, right)
    beta = (first + last) / 2
    return {"rows": len(rows), "beta": beta, "temperature": 1 / beta, "bracket": [first, last], "iterations": iterations,
            "bound": f"T {1 / low:g} (beta {low:g})" if first == low else f"T {1 / high:g} (beta {high:g})"
            if last == high else None,
            "cross_entropy": objective(rows, beta), "cross_entropy_t1": objective(rows, 1.0)}


def holdout(rows, fitted):
    states = {row["state"] for row in fitted}
    return [row for row in rows if row["state"] not in states], [row for row in rows if row["state"] in states]


def evaluate(rows, chosen):
    groups = units(rows)
    draws = resamples(groups)
    result = {}
    for kind in KINDS:
        indices = [index for index, row in enumerate(rows) if row["kind"] == kind]
        result[kind] = {}
        for label, temperature in chosen[kind].items():
            judged = {index: judge(rows[index]["logits"], kind, rows[index]["target"], temperature) for index in indices}
            samples = [ece([judged[index] for index in resampled(groups[kind], draw[kind])]) for draw in draws]
            result[kind][label] = {"temperature": temperature, "rows": len(indices),
                                   "nll": mean([value["nll"] for value in judged.values()]),
                                   "brier_multiclass": mean([value["brier_multiclass"] for value in judged.values()]),
                                   "ece": ece(list(judged.values())),
                                   "ece_interval": {name: bounds([sample[name] for sample in samples])
                                                    for name in ("soft", "hard")}}
    return result


def select(statistics):
    decisions = {}
    for kind in KINDS:
        card, candidate = statistics[kind]["card"], statistics[kind]["candidate"]
        worse = [name for name in ("nll", "brier_multiclass") if candidate[name] > card[name]]
        decisions[kind] = {"adopted": not worse, "worse": worse,
                           "reason": "the candidate is not worse than the card temperature in NLL and multiclass Brier "
                                     "on H'" if not worse else
                                     f"the candidate is worse than the card temperature on H' in {', '.join(worse)}"}
    return decisions


def token_rows(entries, restored, tokenizer, limit):
    rows, problems = [], []
    for entry, row in zip(entries, restored, strict=True):
        tokens = tokenize(tokenizer, render(row["kind"], row["state"], row["question"], row["options"]))
        if row["id"] != entry["id"] or len(tokens) != entry["tokens"]:
            problems.append(f"{entry['id']}: differs from its fixture")
        rows.append({"id": entry["id"], "kind": entry["kind"], "options": row["options"], "token_count": len(tokens),
                     "tokens": tokens, "rejected": len(tokens) > limit})
    return rows, problems


def prepare_set(entries, restored, swift, tokenizer, limit):
    expected, problems = token_rows(entries, restored, tokenizer, limit)
    problems += completeness(expected, swift)
    if problems:
        return [], problems, None
    problems += [f"{identifier}: not finite or not a distribution" for identifier in finite(swift["rows"])]
    rows, deviations = [], []
    for base, row, answer in zip(expected, restored, swift["rows"], strict=True):
        found = teacher(row.get("target"), len(row["options"]))
        if found is None:
            problems.append(f"{base['id']}: the target is not a distribution over the options")
        elif not base["rejected"]:
            deviations.append(found[1])
            rows.append({"id": base["id"], "kind": base["kind"], "state": row["state"], "target": found[0],
                         "logits": answer["logits"]})
    targets = {"rows": len(deviations), "normalized": sum(value != 0 for value in deviations),
               "largest_deviation": max((abs(value) for value in deviations), default=0.0),
               "rejected": sum(base["rejected"] for base in expected)}
    return rows, problems, targets


def recorded_sums(root):
    return {path: digest for digest, path in
            (line.split("  ", 1) for line in (Path(root) / SUMS).read_text(encoding="utf-8").splitlines())}


def calibrated_bundle(source, destination, record, per_kind, refit):
    if destination.exists():
        raise FileExistsError(f"{destination.name} must be written to a new directory")
    subprocess.run(["/bin/cp", "-c", "-R", str(source), str(destination)], check=True)
    if recorded_sums(source) != {entry["path"]: entry["sha256"] for entry in listing(source, (SUMS,))}:
        raise ValueError("The source bundle no longer matches its SHA256SUMS")
    original = read_json(source / DECISION)
    write_json(destination / REFIT, record)
    files = listing(destination)
    if [entry for entry in files if entry["path"] != REFIT] != original["files"]:
        raise ValueError("The copied bundle differs from the source bundle")
    digest = sha256(destination / REFIT)
    decision = read_json(destination / DECISION)
    decision["files"] = files
    decision["temperatures"] = {"per_kind": per_kind, "source": "refit" if refit else "card",
                                "calibration_sha256": digest, "calibration_id": f"jev27b-refit/{digest[:16]}"}
    write_json(destination / DECISION, decision)
    write_sums(destination)
    errors = validate(read_json(destination / DECISION), destination)
    if errors:
        raise ValueError(f"The calibrated decision.json does not validate: {errors}")
    return {"name": destination.name, "decision_sha256": sha256(destination / DECISION), "files": len(files),
            "temperatures": decision["temperatures"],
            "asset_sha256": recorded_sums(destination)[f"{decision['backbone']['asset']}/main.mlirb"]}


def number(value, digits=6):
    return "-" if value is None else f"{value:.{digits}f}"


def ece_cell(item, name):
    spread = item["ece_interval"][name]
    return f"{item['ece'][name]:.5f} [{spread['low']:.5f}, {spread['high']:.5f}]"


def report(result):
    lines = ["# JEV-27B temperature refit", "", f"- Recorded: {str(result['recorded']).lower()}."]
    if not result["recorded"]:
        lines += ["", "## Input problems", "", *(f"- {name}: {items}" for name, items in result["problems"].items() if items)]
        return "\n".join(lines) + "\n"
    refit, bundle, comparison = result["refit"], result["bundle"], result["comparison_refit"]
    holdout_rows = refit["holdout"]
    lines += ["- The fit minimizes the mean cross-entropy to the teacher distribution on Cal over beta = 1 / T in "
              f"[{LOW}, {HIGH}] with a golden-section search to a bracket below {STEP}. Every temperature below is "
              "evaluated at its float32 value, as the Swift runtime uses it.",
              f"- H' is H without the {holdout_rows['excluded_rows']} rows whose state also occurs in Cal "
              f"({holdout_rows['excluded_by_kind']}); a kind adopts its candidate only when neither the NLL nor the "
              "multiclass Brier on H' is larger than at the card temperature. H selects the temperature and is not a "
              "final evaluation; all of H is recorded but is not an independent holdout.",
              "- These numbers measure agreement with the teacher distribution, not calibration against real-world "
              "correctness. ECE intervals are conditional on the chosen temperature and leave out the uncertainty of "
              "the fit and of the selection.", "",
              "| Kind | Card T | Candidate T | End of the range reached | Cal CE (T = 1 / card / candidate) | Adopted T | Adopted | "
              "Reason |", "|---|---:|---:|---|---|---:|---|---|"]
    cal = refit["statistics"]["Cal"]
    for kind, item in refit["kinds"].items():
        fitted = item["fit"]
        lines.append(f"| {kind} | {item['card']} | {item['candidate']} | {fitted['bound'] or 'no'} | "
                     f"{number(fitted['cross_entropy_t1'])} / {number(cal[kind]['card']['nll'])} / "
                     f"{number(cal[kind]['candidate']['nll'])} | {item['temperature']} | "
                     f"{str(item['adopted'] == 'candidate').lower()} | {item['reason']} |")
    for name in ("H'", "H"):
        statistics = refit["statistics"][name]
        lines += ["", f"## {name}: NLL, multiclass Brier and ECE", "",
                  "| Kind | Rows | NLL card T | NLL candidate T | Brier card T | Brier candidate T | ECE soft card T | "
                  "ECE soft adopted T | ECE hard card T | ECE hard adopted T |", "|---|---:|---:|---:|---:|---:|---|---|---|---|"]
        for kind in KINDS:
            card = statistics[kind]["card"]
            adopted = statistics[kind][refit["kinds"][kind]["adopted"]]
            lines.append(f"| {kind} | {card['rows']} | {number(card['nll'])} | "
                         f"{number(statistics[kind]['candidate']['nll'])} | {number(card['brier_multiclass'])} | "
                         f"{number(statistics[kind]['candidate']['brier_multiclass'])} | {ece_cell(card, 'soft')} | "
                         f"{ece_cell(adopted, 'soft')} | {ece_cell(card, 'hard')} | {ece_cell(adopted, 'hard')} |")
    lines += ["", "## Bundle", "",
              f"- `{bundle['name']}`: `decision.json` SHA-256 `{bundle['decision_sha256']}`, temperatures "
              f"{bundle['temperatures']}. `{REFIT}` is listed in `files` and in `SHA256SUMS`.",
              f"- Fidelity of F re-evaluated from the saved logits at the adopted temperatures: passed "
              f"{str(comparison['passed']).lower()}; at the adopted temperatures, maximum {comparison['maximum']:.6f}, mean "
              f"{comparison['mean']:.6f}, top-1 changes {comparison['top_changes']}.",
              f"- Stage 3 time: Cal {result['timing']['Cal']}, H {result['timing']['H']}."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Refit the JEV-27B temperatures on Cal from stage 3 logits, select "
                                                 "them on H and write a calibrated copy of the bundle.")
    for name in ("model", "snapshot", "corpus", "jglue", "fixtures", "swift-cal", "swift-h", "bundle", "reference-q",
                 "swift-f", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "calibration.json").exists():
        raise FileExistsError("calibration.json must be written to a new directory")
    contract = load_contract(args.model)
    check_integrity(args.corpus, contract["dataset"]["id"], contract["dataset"]["revision"])
    tokenizer = distributed_tokenizer(args.snapshot)
    swift = {"Cal": read_json(args.swift_cal), "H": read_json(args.swift_h), "F": read_json(args.swift_f)}
    source = read_json(args.bundle / DECISION)
    asset_sha256 = recorded_sums(args.bundle)[f"{source['backbone']['asset']}/main.mlirb"]
    rows, problems, targets = {}, {}, {}
    for name in ("Cal", "H"):
        fixture = args.fixtures / f"{name}.json"
        rows[name], problems[name], targets[name] = prepare_set(
            read_json(fixture)["rows"], load_rows(fixture, args.corpus, args.jglue), swift[name], tokenizer,
            contract["limits"]["input_tokens"])
    problems["bundle"] = [f"{name} was not run on the asset and decision.json of the source bundle"
                          for name, document in swift.items()
                          if document.get("bundle", {}).get("asset_sha256") != asset_sha256
                          or document.get("bundle", {}).get("version") != sha256(args.bundle / DECISION)]
    kept, excluded = holdout(rows["H"], rows["Cal"])
    problems["holdout"] = [] if len(excluded) == SHARED_STATES else [
        f"{len(excluded)} rows of H share a state with Cal; the fixed sets have {SHARED_STATES}"]
    args.output.mkdir(parents=True, exist_ok=True)
    if any(problems.values()):
        result = {"schema": SCHEMA, "recorded": False, "problems": problems}
        (args.output / "report.md").write_text(report(result), encoding="utf-8")
        write_json(args.output / "calibration.json", result)
        return 1
    card = source["temperatures"]["per_kind"]
    fits = {kind: fit([row for row in rows["Cal"] if row["kind"] == kind]) for kind in KINDS}
    chosen = {kind: {"card": float32(card[kind]), "candidate": float32(fits[kind]["temperature"])} for kind in KINDS}
    statistics = {name: evaluate(chosen_rows, chosen) for name, chosen_rows in
                  (("Cal", rows["Cal"]), ("H'", kept), ("H", rows["H"]))}
    decisions = select(statistics["H'"])
    per_kind = {kind: shortest(fits[kind]["temperature"]) if decisions[kind]["adopted"] else card[kind] for kind in KINDS}
    record = {
        "schema": REFIT_SCHEMA, "temperatures": per_kind,
        "kinds": {kind: {"card": card[kind], "card_float32": shortest(card[kind]),
                         "candidate": shortest(fits[kind]["temperature"]), "fit": fits[kind],
                         "adopted": "candidate" if decisions[kind]["adopted"] else "card", "temperature": per_kind[kind],
                         "reason": decisions[kind]["reason"]} for kind in KINDS},
        "statistics": statistics,
        "holdout": {"rule": "rows of H whose state string equals a state of Cal are excluded",
                    "excluded_rows": len(excluded), "expected_excluded_rows": SHARED_STATES,
                    "excluded_by_kind": dict(Counter(row["kind"] for row in excluded)),
                    "excluded_ids": [row["id"] for row in excluded], "rows": len(kept),
                    "by_kind": dict(Counter(row["kind"] for row in kept))},
        "settings": {"objective": "mean cross-entropy to the teacher distribution", "parameter": "beta = 1 / T",
                     "beta_range": [LOW, HIGH], "bracket_below": STEP, "search": "golden-section",
                     "evaluated_at": "float32 temperatures", "selection": "NLL and multiclass Brier on H' at most the "
                     "card values, without tolerance", "ece_bins": BINS, "ece_confidence": "max p",
                     "bootstrap": {"replicates": REPLICATES, "seed": SEED, "quantiles": list(QUANTILES),
                                   "unit": "rows of one kind that share a state"}},
        "asset_sha256": asset_sha256,
        "fixtures": {name: sha256(args.fixtures / f"{name}.json") for name in ("Cal", "H")},
        "inputs": {name: {"input_sha256": swift[name].get("input_sha256"), "bundle_version": swift[name]["bundle"]["version"]}
                   for name in ("Cal", "H")},
        "targets": targets,
    }
    refit = any(decision["adopted"] for decision in decisions.values())
    bundle = calibrated_bundle(args.bundle, args.output / args.bundle.name, record, per_kind, refit)
    reference = read_json(args.reference_q)
    write_json(args.output / "temperatures-refit.json", {"per_kind": per_kind})
    comparison = compare(reference, swift["F"], temperatures(args.output / "temperatures-refit.json", reference))
    (args.output / "comparison-refit.md").write_text(fidelity_report(comparison), encoding="utf-8")
    write_json(args.output / "comparison-refit.json", comparison)
    shipped = comparison["shipping_temperature"]
    result = {"schema": SCHEMA, "recorded": True, "adopted": {kind: decisions[kind]["adopted"] for kind in KINDS},
              "temperatures": per_kind, "refit": record, "bundle": bundle,
              "comparison_refit": {"passed": comparison["passed"], "maximum": shipped["max_probability_error"],
                                   "mean": shipped["mean_probability_error"],
                                   "top_changes": shipped["top1_changed"]["rows"]},
              "timing": {name: elapsed(swift[name]["rows"]) for name in ("Cal", "H")}, "problems": problems}
    (args.output / "report.md").write_text(report(result), encoding="utf-8")
    write_json(args.output / "calibration.json", result)
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
