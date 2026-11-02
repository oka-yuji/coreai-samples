import argparse
import math
from pathlib import Path
import sys

from ...storage import read_json, write_json
from .decision import compare_hashes
from .prepare import load_contract
from .typed_head import BANDS
from .verify_merge import SHARED, largest, stats, summary, top

SCHEMA = "typed-decision-quantization/1"
SETS = ("F", "Q")
SUM_TOLERANCE = 1e-3


def stage_keys(stage):
    return f"logits{stage}", f"p{stage}", f"p{stage}_T", f"elapsed{stage}_s"


def expected_rows(fixtures, limit):
    return [{"set": name, "id": entry["id"], "kind": entry["kind"], "band": entry["band"], "row_sha256": entry["sha256"],
             "token_count": entry["tokens"], "n_options": entry["n_options"], "rejected": entry["tokens"] > limit}
            for name in SETS for entry in read_json(Path(fixtures) / f"{name}.json")["rows"]]


def complete(rows, expected, stage):
    logits, plain, shipped, _ = stage_keys(stage)
    problems = []
    if [(row["set"], row["id"]) for row in rows] != [(row["set"], row["id"]) for row in expected]:
        return ["rows differ from the fixtures in identity or order"]
    for row, entry in zip(rows, expected):
        if any(row[key] != entry[key] for key in ("kind", "band", "row_sha256", "token_count", "rejected")) \
                or len(row["options"]) != entry["n_options"] or len(row["tokens"]) != row["token_count"]:
            problems.append(f"{row['id']}: differs from its fixture")
        elif row["rejected"] != (row[logits] is None):
            problems.append(f"{row['id']}: forward results do not match the rejection")
        elif not row["rejected"] and (len(row[logits]) != 24 or len(row[plain]) != entry["n_options"]
                                      or len(row[shipped]) != entry["n_options"]):
            problems.append(f"{row['id']}: results have the wrong length")
    return problems


def finite(rows, stage):
    logits, plain, shipped, _ = stage_keys(stage)
    bad = []
    for row in rows:
        if row["rejected"]:
            continue
        if any(row[key] is None for key in (logits, plain, shipped)):
            bad.append(row["id"])
            continue
        values = [value for key in (logits, plain, shipped) for value in row[key]]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values) \
                or not all(0.0 <= value <= 1.0 for key in (plain, shipped) for value in row[key]) \
                or any(abs(math.fsum(row[key]) - 1.0) > SUM_TOLERANCE for key in (plain, shipped)):
            bad.append(row["id"])
    return bad


def same_inputs(first, second):
    return [a["id"] for a, b in zip(first, second, strict=True) if any(a[key] != b[key] for key in (*SHARED, "set"))]


def compare(rows, stage, reference):
    errors = []
    for row in rows:
        if row["rejected"]:
            continue
        probe, base = row[f"p{stage}"], row[f"p{reference}"]
        ranked = sorted(base, reverse=True)
        error = largest(probe, base)
        errors.append({"id": row["id"], "kind": row["kind"], "band": row["band"], "e": error,
                       "e_T": largest(row[f"p{stage}_T"], row[f"p{reference}_T"]),
                       "top": top(probe) != top(base), "top_T": top(row[f"p{stage}_T"]) != top(row[f"p{reference}_T"]),
                       "within": ranked[0] - ranked[1] <= 2 * error})
    margin = [row["id"] for row in errors if row["within"]]
    return {"rows": len(errors), "temperature_1": summary(errors, "e", "top"),
            "shipping_temperature": summary(errors, "e_T", "top_T"),
            "margin_within_2e": {"rows": len(margin), "ids": margin}}


def differences(rows, first, second):
    return stats([largest(row[f"p{first}"], row[f"p{second}"]) for row in rows if not row["rejected"]])


def joined(probes, references):
    merged, base = probes
    stage0, stage1m = references
    found = {row["id"]: row for row in stage0["rows"]}
    found1m = {row["id"]: row for row in stage1m["rows"]}
    rows = []
    for second, zero in zip(merged["rows"], base["rows"], strict=True):
        row = {key: second[key] for key in ("set", *SHARED)}
        sources = [("2", second), ("0q", zero)]
        if row["set"] == "F":
            sources = [("0", found[row["id"]]), ("1", found[row["id"]]), ("1m", found1m[row["id"]]), *sources]
        for stage, source in sources:
            row.update({key: source[key] for key in stage_keys(stage)})
        rows.append(row)
    return rows


def lines(title, groups):
    return [f"| {title} | Rows | Maximum | Mean |", "|---|---:|---:|---:|",
            *(f"| {name} | {value['rows']} | {value['max']:.6f} | {value['mean']:.6f} |" for name, value in groups.items()
              if value["rows"])]


def section(name, comparison):
    plain, shipped = comparison["temperature_1"], comparison["shipping_temperature"]
    bands = {label: plain["by_band"][label] for label, _ in BANDS if label in plain["by_band"]}
    return [f"## {name}", "",
            f"- Rows {comparison['rows']}. Temperature 1: maximum {plain['max_probability_error']:.6f}, mean "
            f"{plain['mean_probability_error']:.6f}, top-1 changes {plain['top1_changed']['rows']}, rows with a reference "
            f"margin of at most 2e {comparison['margin_within_2e']['rows']}.",
            f"- Shipping temperature: maximum {shipped['max_probability_error']:.6f}, mean "
            f"{shipped['mean_probability_error']:.6f}, top-1 changes {shipped['top1_changed']['rows']}.", "",
            *lines("Kind", plain["by_kind"]), "", *lines("Band", bands), ""]


def report(result):
    adaptation = result["adaptation"]
    checks = result["checks"]
    return "\n".join([
        "# JEV-27B quantized references", "",
        f"- Passed: {str(result['passed']).lower()} (hashes {str(checks['hashes_match']).lower()}, rows "
        f"{str(checks['rows_complete']).lower()}, finite {str(checks['finite']).lower()}). No numeric threshold applies.",
        f"- Hashed tensors {result['hashes']['items']}, matching {result['hashes']['matching']}.",
        f"- Rows: F {result['rows']['F']} (forwarded {result['rows']['F_forwarded']}), Q {result['rows']['Q']} "
        f"(forwarded {result['rows']['Q_forwarded']}), rejected {result['rows']['rejected']}.",
        f"- Devices: stage 2 {result['devices']['2']}, stage 0q {result['devices']['0q']}.", "",
        *section("Stage 2 against stage 1' (F)", result["stage2_vs_1m"]),
        *section("Stage 0q against stage 0 (F)", result["stage0q_vs_0"]),
        "## Adaptation (not a criterion)", "",
        "| Set | Comparison | Rows | Mean | Maximum |", "|---|---|---:|---:|---:|",
        *(f"| {name} | {label} | {value['rows']} | {value['mean']:.6f} | {value['max']:.6f} |"
          for name, values in adaptation.items() for label, value in values.items() if value["rows"]), ""])


def main():
    parser = argparse.ArgumentParser(description="Check the stage 2 and 0q references and record their differences.")
    for name in ("model", "fixtures", "reference", "merged-reference", "probe-merged", "probe-base", "export", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "quantization.json").exists():
        raise FileExistsError("quantization.json must be written to a new directory")
    contract = load_contract(args.model)
    expected = expected_rows(args.fixtures, contract["limits"]["input_tokens"])
    merged, base = read_json(args.probe_merged / "reference_q-merged.json"), read_json(args.probe_base / "reference_q-base.json")
    stage0, stage1m = read_json(args.reference), read_json(args.merged_reference)
    hashes = compare_hashes(read_json(args.export / "quantized-hashes.json"),
                            read_json(args.probe_merged / "quantized-hashes.json"))
    problems = {"2": complete(merged["rows"], expected, "2"), "0q": complete(base["rows"], expected, "0q")}
    f_rows = [row for row in merged["rows"] if row["set"] == "F"]
    problems["probes_differ"] = [] if problems["2"] or problems["0q"] else same_inputs(merged["rows"], base["rows"])
    references = [[row[key] for key in SHARED] for row in stage0["rows"]]
    problems["stage0_differs"] = [] if references == [[row[key] for key in SHARED] for row in f_rows] else ["stage 0/1"]
    problems["stage1m_differs"] = [] if references == [[row[key] for key in SHARED] for row in stage1m["rows"]] \
        else ["stage 1'"]
    problems["references_failed"] = [name for name, document in (("stage 0/1", stage0), ("stage 1'", stage1m))
                                     if document.get("passed") is not True]
    rows_complete = not any(problems.values())
    bad = {"2": finite(merged["rows"], "2"), "0q": finite(base["rows"], "0q")}
    checks = {"hashes_match": hashes["hashes_match"] and merged.get("hashes_match") is True, "rows_complete": rows_complete,
              "finite": not any(bad.values())}
    result = {"schema": SCHEMA, "passed": all(checks.values()), "checks": checks,
              "hashes": {key: hashes[key] for key in ("items", "compared", "matching", "order", "layout", "settings")},
              "hash_differences": {key: hashes[key] for key in ("differing", "only_first", "only_second", "unreadable")},
              "problems": problems, "nonfinite": bad, "devices": {"2": merged["device"], "0q": base["device"]},
              "rows": {name: sum(row["set"] == name for row in expected) for name in SETS}}
    result["rows"].update({f"{name}_forwarded": sum(row["set"] == name and not row["rejected"] for row in expected)
                           for name in SETS}, rejected=sum(row["rejected"] for row in expected))
    if rows_complete and checks["finite"]:
        rows = joined((merged, base), (stage0, stage1m))
        q_rows = [row for row in rows if row["set"] == "Q"]
        f_joined = [row for row in rows if row["set"] == "F"]
        result.update(stage2_vs_1m=compare(f_joined, "2", "1m"), stage0q_vs_0=compare(f_joined, "0q", "0"),
                      adaptation={"F": {"stage 2 against 0q": differences(f_joined, "2", "0q"),
                                        "stage 1' against 0": differences(f_joined, "1m", "0")},
                                  "Q": {"stage 2 against 0q": differences(q_rows, "2", "0q")}})
        write_json(args.output / "reference_q.json", {"merged_revision": merged["revision"], "base_revision": base["revision"],
                                                       "devices": result["devices"], "dtype": merged["dtype"],
                                                       "temperatures": merged["temperatures"], "rows": rows})
        (args.output / "report.md").write_text(report(result), encoding="utf-8")
    write_json(args.output / "quantization.json", result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
