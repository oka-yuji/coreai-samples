import argparse
from collections import Counter
import math
from pathlib import Path
import sys

from ...storage import read_json, write_json
from ..pinned import sha256
from .fixtures import CARD_EXAMPLES
from .merge import lora_pairs, manifest_files, merge_projection, merged_revision, saved_index, saved_name, targeted
from .prepare import TOKENIZER_FILES, load_contract, safetensors_header
from .typed_head import BANDS

SCHEMA = "typed-decision-merge/1"
THRESHOLD = 0.01
SHARED = ("id", "kind", "options", "band", "row_sha256", "text_sha256", "token_count", "tokens", "rejected")
TEXT_KEYS = ("rope_parameters", "layer_types", "mamba_ssm_dtype", "mtp_num_hidden_layers")


def identical(first, second):
    import torch

    return first.dtype == second.dtype and first.shape == second.shape \
        and torch.equal(first.reshape(-1).view(torch.uint8), second.reshape(-1).view(torch.uint8))


def check_weights(snapshot, base, merged, contract):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    checkpoint = merged / "checkpoint"
    source = read_json(snapshot / "model.safetensors.index.json")
    index = read_json(checkpoint / "model.safetensors.index.json")
    shards = sorted(set(source["weight_map"].values()))
    lora = load_file(str(snapshot / "adapter/adapter_model.safetensors"))
    pairs = lora_pairs(lora)
    scale = contract["lora"]["scale"]
    projections = targeted(source["weight_map"], contract["lora"]["targets"])
    headers = {shard: safetensors_header(checkpoint / shard) for shard in shards}
    stored = {name: shard for shard, header in headers.items() for name in header}
    stored_bytes = sum(info["data_offsets"][1] - info["data_offsets"][0] for header in headers.values()
                       for info in header.values())
    consumed, dtypes = Counter(), Counter()
    mismatched, retyped, metadata = [], [], []
    with torch.no_grad():
        for shard in shards:
            with safe_open(str(snapshot / shard), framework="pt") as original, \
                    safe_open(str(checkpoint / shard), framework="pt") as saved:
                if saved.metadata() != {"format": "pt"}:
                    metadata.append(shard)
                for name in sorted(key for key, owner in source["weight_map"].items() if owner == shard):
                    if stored.get(saved_name(name)) != shard:
                        mismatched.append(name)
                        continue
                    before = original.get_tensor(name)
                    after = saved.get_tensor(saved_name(name))
                    expected = before
                    if name in pairs:
                        first, second = pairs[name]
                        expected = merge_projection(before, lora[first], lora[second], scale)
                        consumed[name] += 1
                    dtypes[f"{name.rsplit('.', 1)[-1]}:{after.dtype}"] += 1
                    if after.dtype != before.dtype:
                        retyped.append(name)
                    elif not identical(after, expected):
                        mismatched.append(name)
    config = read_json(checkpoint / "config.json")
    flat = read_json(snapshot / "config.json")
    text = config.get("text_config", {})
    different = sorted({key for key in text if key not in flat or flat[key] != text[key]} | set(TEXT_KEYS) - set(text))
    files = manifest_files(checkpoint)
    manifest = read_json(merged / "manifest.json")
    checks = {
        "tensors_equal": not mismatched and not retyped and len(source["weight_map"]) == len(stored),
        "pairs_consumed": sorted(consumed) == projections == sorted(pairs) and set(consumed.values()) == {1}
        and len(pairs) == contract["lora"]["pairs"],
        "index": index == saved_index(source["weight_map"], source["metadata"]["total_size"]),
        "index_matches_shards": stored == index["weight_map"],
        "total_size": index["metadata"]["total_size"] == stored_bytes == source["metadata"]["total_size"],
        "shard_metadata": not metadata,
        "config_bytes": (checkpoint / "config.json").read_bytes() == (base / "config.json").read_bytes(),
        "composite_config": config.get("model_type") == "qwen3_5" and bool(text),
        "text_config": not different,
        "rope_parameters": "rope_parameters" in text and text["rope_parameters"] == flat.get("rope_parameters"),
        "tokenizer": all(sha256(checkpoint / name) == sha256(snapshot / name) for name in TOKENIZER_FILES),
        "files": [entry["path"] for entry in files] == sorted({"config.json", "model.safetensors.index.json",
                                                              *TOKENIZER_FILES, *shards}),
        "manifest": manifest["files"] == files and manifest["merged_revision"] == merged_revision(files),
    }
    return {"passed": all(checks.values()), "checks": checks, "merged_revision": merged_revision(files),
            "tensors": len(source["weight_map"]), "merged": len(consumed), "copied": len(source["weight_map"]) - len(consumed),
            "pairs_consumed": len(consumed), "pairs_expected": contract["lora"]["pairs"], "targeted_projections": len(projections),
            "per_target": dict(sorted(Counter(name.split(".")[-2] for name in consumed).items())),
            "dtypes": dict(sorted(dtypes.items())), "total_size": stored_bytes, "mismatched": mismatched, "retyped": retyped,
            "text_config_differences": different}


def largest(first, second):
    return max(abs(a - b) for a, b in zip(first, second, strict=True))


def top(values):
    return max(range(len(values)), key=values.__getitem__)


def stats(values):
    return {"rows": len(values), "max": max(values), "mean": math.fsum(values) / len(values)} if values else {"rows": 0}


def grouped(rows, key, field):
    groups = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row[field])
    return {name: stats(values) for name, values in sorted(groups.items())}


def summary(errors, field, top_field):
    values = [row[field] for row in errors]
    changed = [row["id"] for row in errors if row[top_field]]
    return {"max_probability_error": max(values, default=0.0), "mean_probability_error": stats(values).get("mean", 0.0),
            "by_kind": grouped(errors, "kind", field), "by_band": grouped(errors, "band", field),
            "top1_changed": {"rows": len(changed), "ids": changed}}


def compare_outputs(reference, merged):
    first, second = reference["rows"], merged["rows"]
    matching = [row["id"] for row in first] == [row["id"] for row in second]
    differing = [a["id"] for a, b in zip(first, second) if any(a[key] != b[key] for key in SHARED)] if matching else []
    joined, errors = [], []
    for a, b in zip(first, second) if matching and not differing else ():
        row = {key: a[key] for key in SHARED}
        for stage, source in (("0", a), ("1", a), ("1m", b)):
            for key in (f"logits{stage}", f"p{stage}", f"p{stage}_T", f"elapsed{stage}_s"):
                row[key] = source[key]
        joined.append(row)
        if not row["rejected"]:
            errors.append({"id": row["id"], "kind": row["kind"], "band": row["band"],
                           "e": largest(row["p1m"], row["p1"]), "e_T": largest(row["p1m_T"], row["p1_T"]),
                           "adaptation": largest(row["p1"], row["p0"]),
                           "top": top(row["p1m"]) != top(row["p1"]), "top_T": top(row["p1m_T"]) != top(row["p1_T"]),
                           "logits1": row["logits1"], "logits1m": row["logits1m"]})
    adaptation = [row["adaptation"] for row in errors]
    outputs = {"threshold": THRESHOLD, "references_passed": reference.get("passed") is True and merged.get("passed") is True,
               "rows_match": matching and not differing, "differing_rows": differing,
               **summary(errors, "e", "top"), "shipping_temperature": summary(errors, "e_T", "top_T"),
               "adaptation": {"threshold": THRESHOLD, "rows_above": sum(value > THRESHOLD for value in adaptation),
                              **stats(adaptation)},
               "violations": [{key: row[key] for key in ("id", "kind", "band", "e", "logits1", "logits1m")}
                              for row in errors if row["e"] > THRESHOLD]}
    outputs["passed"] = outputs["references_passed"] and outputs["rows_match"] and bool(errors) \
        and outputs["max_probability_error"] <= THRESHOLD
    return outputs, joined, len(errors)


def card_lines(rows):
    lines = ["| Example | Option | Stage 1 | Stage 1' | Card | Stage 1' - card |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["id"] in CARD_EXAMPLES:
            card = CARD_EXAMPLES[row["id"]]["card_probabilities"]
            for option, first, second, expected in zip(row["options"], row["p1_T"], row["p1m_T"], card, strict=True):
                lines.append(f"| {row['id']} | {option} | {first:.4f} | {second:.4f} | {expected:.4f} | {second - expected:+.4f} |")
    return lines


def table(title, groups):
    return [f"| {title} | Rows | Maximum | Mean |", "|---|---:|---:|---:|",
            *(f"| {name} | {value['rows']} | {value['max']:.6f} | {value['mean']:.6f} |" for name, value in groups.items())]


def report(comparison, rows):
    weights, outputs = comparison["weights"], comparison["outputs"]
    shipped, adaptation = outputs["shipping_temperature"], outputs["adaptation"]
    lines = ["# JEV-27B LoRA merge check", "",
             f"- Merged revision: `{comparison['merged_revision']}`",
             f"- Passed: {str(comparison['passed']).lower()} (weights {str(weights['passed']).lower()}, "
             f"outputs {str(outputs['passed']).lower()})", "",
             "## Card examples at the shipping temperature", "",
             "The card values are shown for reference and are not a pass criterion.", "", *card_lines(rows), "",
             "## Weights", "",
             f"- Tensors {weights['tensors']}: merged {weights['merged']}, copied {weights['copied']}. "
             f"Pairs consumed {weights['pairs_consumed']} of {weights['pairs_expected']}, "
             f"targeted projections {weights['targeted_projections']}.",
             f"- Total size {weights['total_size']} bytes. Mismatched {len(weights['mismatched'])}, "
             f"dtype changes {len(weights['retyped'])}.", "",
             "| Check | Result |", "|---|---|",
             *(f"| {name} | {str(value).lower()} |" for name, value in weights["checks"].items()), "",
             "## Outputs at temperature 1 (stage 1 against stage 1')", "",
             f"- Rows {comparison['rows']}, compared {comparison['compared']}, rejected {comparison['rejected']}, "
             f"threshold {outputs['threshold']}.",
             f"- Maximum {outputs['max_probability_error']:.6f}, mean {outputs['mean_probability_error']:.6f}, "
             f"violations {len(outputs['violations'])}, top-1 changes {outputs['top1_changed']['rows']}.", "",
             *table("Kind", outputs["by_kind"]), "",
             *table("Band", {label: outputs["by_band"][label] for label, _ in BANDS if label in outputs["by_band"]}), "",
             f"- Shipping temperature: maximum {shipped['max_probability_error']:.6f}, mean "
             f"{shipped['mean_probability_error']:.6f}, top-1 changes {shipped['top1_changed']['rows']}.",
             f"- Stage 0 against stage 1: {adaptation['rows_above']} rows above {adaptation['threshold']}, "
             f"mean {adaptation.get('mean', 0.0):.6f}, maximum {adaptation.get('max', 0.0):.6f}."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Check the merged JEV-27B checkpoint by weights and by stage 1 outputs.")
    for name in ("model", "snapshot", "base", "merge", "reference", "merged-reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "comparison.json").exists():
        raise FileExistsError("comparison.json must be written to a new directory")
    contract = load_contract(args.model)
    weights = check_weights(args.snapshot, args.base, args.merge, contract)
    first, second = read_json(args.reference), read_json(args.merged_reference)
    outputs, rows, compared = compare_outputs(first, second)
    comparison = {"schema": SCHEMA, "passed": weights["passed"] and outputs["passed"],
                  "merged_revision": weights["merged_revision"], "rows": len(first["rows"]), "compared": compared,
                  "rejected": sum(row["rejected"] for row in first["rows"]), "weights": weights, "outputs": outputs}
    write_json(args.output / "reference.json", {"merged_revision": weights["merged_revision"], "device": second["device"],
                                                "dtype": second["dtype"], "temperatures": second["temperatures"],
                                                "rows": rows})
    (args.output / "report.md").write_text(report(comparison, rows), encoding="utf-8")
    write_json(args.output / "comparison.json", comparison)
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
