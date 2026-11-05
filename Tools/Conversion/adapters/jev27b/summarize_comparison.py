import argparse
import json
import math
from pathlib import Path
from statistics import median


ORDER = ["A", "B", "C", "D", "D", "C", "B", "A"]
LENGTHS = [100, 500, 2000, 8192]


def distribution(values, count):
    if len(values) != count or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value <= 0 for value in values
    ):
        raise ValueError("Incomplete or invalid observations")
    return {"n": count, "median": median(values), "min": min(values), "max": max(values), "p95": None}


def summarize(document):
    plan = document["plan"]
    runs = document["runs"]
    if plan["sequence"] != ORDER or [run["condition"] for run in runs] != ORDER:
        raise ValueError("The complete fixed process order is required")
    identities = [run["run_id"] for run in runs]
    if any(not isinstance(value, str) or not value for value in identities) or len(set(identities)) != len(runs):
        raise ValueError("Every observation process must have a unique identity")
    if (plan["warmup_rows_per_process"], plan["measured_rounds_per_process"],
            plan["processes_per_condition"], plan["measured_n_per_length"]) != (4, 3, 2, 6):
        raise ValueError("The fixed sample counts changed")
    inputs = plan["inputs"]
    if [row["tokens"] for row in inputs] != LENGTHS:
        raise ValueError("The fixed token lengths changed")
    samples = {condition: {str(length): [] for length in LENGTHS} for condition in "ABCD"}
    rounds = {condition: [] for condition in "ABCD"}
    memory = {condition: {key: [] for key in ["max_rss_bytes", "kernel_footprint_peak_bytes"]}
              for condition in "ABCD"}
    for run in runs:
        if run["completed"] is not True or run["guard_stop"] is not None or len(run["rows"]) != 16:
            raise ValueError("A process did not complete all scheduled rows")
        condition = run["condition"]
        for index, row in enumerate(run["rows"]):
            expected = inputs[index % 4]
            if row["token_count"] != expected["tokens"] or row["tokens_sha256"] != expected["tokens_sha256"]:
                raise ValueError("A row differs from the fixed token sequence")
            if row["thermal_state_at_start"] != "nominal":
                raise ValueError("Every row must start nominal")
            distribution([row["elapsed_seconds"]], 1)
            if index >= 4:
                samples[condition][str(row["token_count"])].append(row["elapsed_seconds"])
        for first in [4, 8, 12]:
            rounds[condition].append(sum(row["elapsed_seconds"] for row in run["rows"][first:first + 4]))
        for key in memory[condition]:
            memory[condition][key].append(run["memory"][key])
    return {condition: {
        "latency_seconds": {length: distribution(values, 6) for length, values in samples[condition].items()},
        "mixed_round_seconds": distribution(rounds[condition], 6),
        "memory": {key: distribution(values, 2) for key, values in memory[condition].items()},
    } for condition in "ABCD"}


def main():
    parser = argparse.ArgumentParser(description="Recompute the fixed JEV-27B comparison summaries.")
    parser.add_argument("record", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = json.loads(args.record.read_text())
    summary = summarize(document)
    if args.check:
        if summary != document["summary"]:
            raise ValueError("The stored summary differs from the observations")
        print("JEV-27B comparison summary: PASS")
    else:
        print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
