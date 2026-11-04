import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import sys

from ...storage import read_json, write_json
from .typed_head import BANDS, decide
from .verify_merge import grouped, largest, stats, summary, top

SCHEMA = "typed-decision-fidelity/1"
SWIFT_SCHEMA = "typed-decide/1"
MAXIMUM, MEAN = 0.03, 0.01
SUM_TOLERANCE = 1e-3
TOP_RULE = 0.06
CLI_TOLERANCE = 1e-6
KINDS = ("noul", "choice", "score")


def tokens_sha256(tokens):
    return hashlib.sha256(",".join(str(token) for token in tokens).encode("utf-8")).hexdigest()


def temperatures(path, reference):
    values = (read_json(path)["per_kind"] if path else reference["temperatures"])
    if set(values) != set(KINDS) or any(isinstance(values[kind], bool) or not isinstance(values[kind], (int, float))
                                        or not math.isfinite(values[kind]) or values[kind] <= 0 for kind in KINDS):
        raise ValueError("Temperatures must give noul, choice and score positive finite values")
    return values


def completeness(expected, swift):
    rows = swift.get("rows") if isinstance(swift.get("rows"), list) else []
    problems = []
    if swift.get("schema") != SWIFT_SCHEMA:
        problems.append(f"schema is not {SWIFT_SCHEMA}")
    for name, items in (("reference", expected), ("swift", rows)):
        repeated = sorted(key for key, count in Counter(row.get("id") for row in items).items() if count > 1)
        if repeated:
            problems.append(f"{name} repeats ids: {repeated}")
    if not expected or [row.get("id") for row in rows] != [row["id"] for row in expected]:
        return problems + ["rows differ from the reference in identity or order"]
    for row, reference in zip(rows, expected):
        identifier, count = reference["id"], len(reference["options"])
        if row.get("kind") != reference["kind"]:
            problems.append(f"{identifier}: kind differs")
        elif reference["rejected"]:
            if row.get("error") != "tooManyTokens" or row.get("token_count") != reference["token_count"]:
                problems.append(f"{identifier}: expected a tooManyTokens rejection at {reference['token_count']} tokens")
        elif "error" in row:
            problems.append(f"{identifier}: unexpected error {row['error']}")
        elif row.get("token_count") != reference["token_count"] \
                or row.get("tokens_sha256") != tokens_sha256(reference["tokens"]):
            problems.append(f"{identifier}: token IDs differ from the reference")
        elif not all(isinstance(row.get(key), list) for key in ("logits", "probabilities", "probabilities_t1")) \
                or len(row["logits"]) != 24 or len(row["probabilities"]) != count \
                or len(row["probabilities_t1"]) != count:
            problems.append(f"{identifier}: results have the wrong length")
        elif type(row.get("capacity")) is not int or type(row.get("top")) is not int \
                or not isinstance(row.get("elapsed_seconds"), (int, float)):
            problems.append(f"{identifier}: capacity, top or elapsed_seconds is missing")
    return problems


def finite(rows):
    bad = []
    for row in rows:
        if "error" in row:
            continue
        values = [*row["logits"], *row["probabilities"], *row["probabilities_t1"]]
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                   for value in values) \
                or not all(0.0 <= value <= 1.0 for key in ("probabilities", "probabilities_t1") for value in row[key]) \
                or any(abs(math.fsum(row[key]) - 1.0) > SUM_TOLERANCE for key in ("probabilities", "probabilities_t1")):
            bad.append(row["id"])
    return bad


def changed_top(row, swift, reference):
    first, second = top(reference), top(swift)
    ranked = sorted(reference, reverse=True)
    gap = reference[first] - reference[second]
    return {"id": row["id"], "kind": row["kind"], "band": row["band"], "reference": reference, "swift": swift,
            "reference_top": first, "swift_top": second, "reference_margin": ranked[0] - ranked[1], "gap": gap,
            "within_rule": gap <= TOP_RULE}


def aggregate(errors, changes, judged=True):
    result = summary(errors, "e", "top")
    result["bands"] = {}
    for label, _ in BANDS:
        group = result["by_band"].get(label)
        if group:
            result["bands"][label] = {**group, "passed": group["max"] <= MAXIMUM and group["mean"] <= MEAN} \
                if judged else group
    result["unevaluated"] = [label for label, _ in BANDS if label not in result["bands"]]
    result["top_changes"] = changes
    result["top_changes_within_rule"] = sum(change["within_rule"] for change in changes)
    return result


def compare(reference, swift, shipping):
    expected = [row for row in reference["rows"] if row["set"] == "F"]
    problems = completeness(expected, swift)
    rows = swift["rows"] if not problems else []
    bad = finite(rows)
    pairs = [(row, base) for row, base in zip(rows, expected) if not base["rejected"]] if not bad else []
    plain, plain_changes, shipped, shipped_changes, merged, cli = [], [], [], [], [], []
    for row, base in pairs:
        kind, count = base["kind"], len(base["options"])
        plain.append({"id": base["id"], "kind": kind, "band": base["band"],
                      "e": largest(row["probabilities_t1"], base["p2"]),
                      "top": top(row["probabilities_t1"]) != top(base["p2"])})
        if plain[-1]["top"]:
            plain_changes.append(changed_top(base, row["probabilities_t1"], base["p2"]))
        swift_t, _ = decide(row["logits"], kind, count, shipping[kind])
        base_t, _ = decide(base["logits2"], kind, count, shipping[kind])
        shipped.append({"id": base["id"], "kind": kind, "band": base["band"], "e": largest(swift_t, base_t),
                        "top": top(swift_t) != top(base_t)})
        if shipped[-1]["top"]:
            shipped_changes.append(changed_top(base, swift_t, base_t))
        merged.append({"id": base["id"], "kind": kind, "band": base["band"],
                       "e": largest(row["probabilities_t1"], base["p1m"])})
        bundled, _ = decide(row["logits"], kind, count, reference["temperatures"][kind])
        cli.append(largest(row["probabilities"], bundled))
    temperature_1 = aggregate(plain, plain_changes)
    passed = not problems and not bad and bool(temperature_1["bands"]) \
        and all(band["passed"] for band in temperature_1["bands"].values())
    elapsed = [(base["band"], row["elapsed_seconds"]) for row, base in pairs]
    return {"schema": SCHEMA, "passed": passed, "criteria": {"maximum": MAXIMUM, "mean": MEAN,
                                                              "sum_tolerance": SUM_TOLERANCE, "top_rule": TOP_RULE},
            "checks": {"rows_complete": not problems, "finite": None if problems else not bad,
                       "bands_passed": bool(temperature_1["bands"])
                       and all(band["passed"] for band in temperature_1["bands"].values())},
            "problems": problems, "nonfinite": bad,
            "rows": {"total": len(expected), "forwarded": len(pairs),
                     "rejected": sum(row["rejected"] for row in expected)},
            "temperature_1": temperature_1,
            "shipping_temperature": {"temperatures": shipping, **aggregate(shipped, shipped_changes, judged=False)},
            "stage_1m_reference": {**stats([row["e"] for row in merged]),
                                   "by_band": {label: group for label, _ in BANDS
                                               if (group := grouped(merged, "band", "e").get(label))}},
            "cli_shipping_difference": max(cli, default=0.0),
            "cli_shipping_consistent": max(cli, default=0.0) <= CLI_TOLERANCE,
            "elapsed": {"total_s": math.fsum(seconds for _, seconds in elapsed),
                        "by_band": {label: stats([seconds for band, seconds in elapsed if band == label])
                                    for label, _ in BANDS if any(band == label for band, _ in elapsed)}},
            "capacity": {str(size): count for size, count in sorted(Counter(row["capacity"] for row, _ in pairs).items())},
            "bundle": swift.get("bundle"), "input_sha256": swift.get("input_sha256")}


def band_lines(title, result):
    judged = all("passed" in group for group in result["bands"].values())
    lines = [f"| {title} | Rows | Maximum | Mean |" + (" Passed |" if judged else ""),
             "|---|---:|---:|---:|" + ("---|" if judged else "")]
    for label, group in result["bands"].items():
        lines.append(f"| {label} | {group['rows']} | {group['max']:.6f} | {group['mean']:.6f} |"
                     + (f" {str(group['passed']).lower()} |" if judged else ""))
    lines += [f"| {label} | 0 | - | - |" + (" unevaluated |" if judged else "") for label in result["unevaluated"]]
    return lines


def change_lines(changes):
    lines = ["| Row | Band | Reference top | Swift top | Reference p(top) | Reference p(new) | Reference margin | "
             "Within 0.06 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for change in changes:
        reference = change["reference"]
        lines.append(f"| {change['id']} | {change['band']} | {change['reference_top']} | {change['swift_top']} | "
                     f"{reference[change['reference_top']]:.6f} | {reference[change['swift_top']]:.6f} | "
                     f"{change['reference_margin']:.6f} | {str(change['within_rule']).lower()} |")
    return lines


def report(result):
    plain, shipped = result["temperature_1"], result["shipping_temperature"]
    finite_label = "unevaluated" if result["checks"]["finite"] is None else str(result["checks"]["finite"]).lower()
    lines = ["# JEV-27B Core AI against the quantized reference (F)", "",
             f"- Passed: {str(result['passed']).lower()} (rows {str(result['checks']['rows_complete']).lower()}, "
             f"finite {finite_label}, bands "
             f"{str(result['checks']['bands_passed']).lower()}). Criteria per band at temperature 1: maximum e_j "
             f"<= {MAXIMUM}, mean <= {MEAN}.",
             f"- Rows {result['rows']['total']}, forwarded {result['rows']['forwarded']}, rejected "
             f"{result['rows']['rejected']}.", ""]
    if result["problems"] or result["nonfinite"]:
        lines += ["## Problems", "", *(f"- {problem}" for problem in result["problems"]),
                  *(f"- {identifier}: not finite or not a distribution" for identifier in result["nonfinite"]), ""]
        return "\n".join(lines) + "\n"
    lines += ["## Temperature 1 (stage 3 against stage 2)", "",
              f"- Maximum {plain['max_probability_error']:.6f}, mean {plain['mean_probability_error']:.6f}, top-1 "
              f"changes {plain['top1_changed']['rows']} ({plain['top_changes_within_rule']} within {TOP_RULE} of the "
              "reference top).", "", *band_lines("Band", plain), "",
              "| Kind | Rows | Maximum | Mean |", "|---|---:|---:|---:|",
              *(f"| {name} | {group['rows']} | {group['max']:.6f} | {group['mean']:.6f} |"
                for name, group in plain["by_kind"].items()), ""]
    if plain["top_changes"]:
        lines += ["Top-1 changes (not a criterion):", "", *change_lines(plain["top_changes"]), ""]
    lines += ["## Shipping temperature (recomputed from the 24 logits, not a criterion)", "",
              f"- Temperatures {shipped['temperatures']}. Maximum {shipped['max_probability_error']:.6f}, mean "
              f"{shipped['mean_probability_error']:.6f}, top-1 changes {shipped['top1_changed']['rows']} "
              f"({shipped['top_changes_within_rule']} within {TOP_RULE}).",
              f"- The CLI probabilities differ from the recomputed bundle-temperature values by at most "
              f"{result['cli_shipping_difference']:.3g}.", "", *band_lines("Band", shipped), ""]
    if shipped["top_changes"]:
        lines += ["Top-1 changes:", "", *change_lines(shipped["top_changes"]), ""]
    merged = result["stage_1m_reference"]
    lines += ["## Stage 3 against stage 1' (reference only)", "",
              f"- Maximum {merged['max']:.6f}, mean {merged['mean']:.6f}.", "",
              "| Band | Rows | Maximum | Mean |", "|---|---:|---:|---:|",
              *(f"| {label} | {group['rows']} | {group['max']:.6f} | {group['mean']:.6f} |"
                for label, group in merged["by_band"].items()), "",
              "## Time", "", f"- Forward rows in total {result['elapsed']['total_s']:.1f} s.", "",
              "| Band | Rows | Mean seconds | Maximum seconds |", "|---|---:|---:|---:|",
              *(f"| {label} | {group['rows']} | {group['mean']:.3f} | {group['max']:.3f} |"
                for label, group in result["elapsed"]["by_band"].items()), "",
              f"- Capacities: {result['capacity']}."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Score the coreai-verify typed-decide output against stage 2.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--set", choices=("F",), required=True)
    parser.add_argument("--swift", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperatures", type=Path)
    args = parser.parse_args()
    if (args.output / "comparison.json").exists():
        raise FileExistsError("comparison.json must be written to a new directory")
    reference = read_json(args.reference)
    result = compare(reference, read_json(args.swift), temperatures(args.temperatures, reference))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.md").write_text(report(result), encoding="utf-8")
    write_json(args.output / "comparison.json", result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
